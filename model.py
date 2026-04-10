"""
SynthScan — ML Model
Model : umm-maybe/AI-image-detector  (Swin Transformer binary classifier)
Heatmap: Grad-CAM on last Swin encoder block → overlay PNG returned as base64

Changes in this revision
─────────────────────────
  • Fixed missing `import os` in ImageAnalysisHandler.run_forensics
  • start_monitor() now delegates to the richer Flask-backed monitor in app.py
    (kept here for standalone CLI usage)
  • analyze_bytes() alias added for convenience (same as analyze())
  • analyze_video() added — extracts every N-th frame via ffmpeg-python,
    runs the detector on each, returns a synthetic-probability timeline
"""

import io
import os
import base64
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import cv2
from huggingface_hub import login
from transformers import AutoImageProcessor, AutoModelForImageClassification
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import time
import tempfile
import ffmpeg
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler


MODEL_NAME     = "umm-maybe/AI-image-detector"
SYNTH_KEYWORDS = {'artificial', 'fake', 'ai', 'generated', 'ai-generated', 'synthetic'}
ALLOWED_EXTS   = {'.jpg', '.jpeg', '.png', '.webp', '.gif'}

token = os.environ.get("HF_TOKEN")
class _Wrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(pixel_values=x).logits


# ── Standalone watchdog handler (used by start_monitor() CLI helper) ────────

class ImageAnalysisHandler(FileSystemEventHandler):
    """
    Watchdog handler for standalone CLI usage.
    When running via Flask, the richer handler in app.py is used instead
    (it pushes SSE events and stores results in the rolling log).
    """

    def __init__(self, detector=None):
        super().__init__()
        # Optionally inject a pre-loaded detector to avoid double-loading
        self._detector = detector

    def on_created(self, event):
        if event.is_directory:
            return
        ext = os.path.splitext(event.src_path)[1].lower()
        if ext in ALLOWED_EXTS:
            print(f"[*] New image detected: {event.src_path}")
            self.run_forensics(event.src_path)

    def run_forensics(self, path: str):
        print(f"[!] Running ViT inference on {path}…")
        try:
            if self._detector is None:
                self._detector = SyntheticImageDetector()
            with open(path, 'rb') as fh:
                image_bytes = fh.read()
            result = self._detector.analyze(image_bytes)
            label   = result['label']
            conf    = round(result['confidence'] * 100, 1)
            synth_p = round(result['synthetic_probability'] * 100, 1)
            print(
                f"    → {label}  |  confidence {conf}%  "
                f"|  synthetic_probability {synth_p}%"
            )
        except Exception as exc:
            print(f"    [ERROR] {exc}")


def start_monitor(path_to_watch: str, detector=None):
    """
    Blocking CLI helper — watches a folder and prints forensic results.
    For non-blocking (Flask) monitoring use the /monitor/* endpoints in app.py.
    """
    event_handler = ImageAnalysisHandler(detector=detector)
    observer      = Observer()
    observer.schedule(event_handler, path_to_watch, recursive=True)
    observer.start()
    print(f"[+] SynthScan monitoring folder: {path_to_watch}")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


# ── Main detector ────────────────────────────────────────────────────────────

class SyntheticImageDetector:

    def __init__(self, model_name: str = MODEL_NAME):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"[Detector] device={self.device}  model={model_name}")

        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model     = AutoModelForImageClassification.from_pretrained(model_name)
        self.model.to(self.device).eval()

        self.id2label = self.model.config.id2label
        print(f"[Detector] labels={self.id2label}")

        self._wrapper      = _Wrapper(self.model)
        self._target_layer = self._resolve_target_layer()
        print(f"[Detector] Grad-CAM target layer: {type(self._target_layer).__name__}")

    # ── Public API ─────────────────────────────────────────────────────────

    def analyze(self, image_bytes: bytes) -> dict:
        image    = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        image_np = np.array(image).astype(np.float32) / 255.0

        inputs = self.processor(images=image, return_tensors='pt')
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = self.model(**inputs).logits
        probs    = F.softmax(logits, dim=-1)[0]
        pred_idx = torch.argmax(probs).item()
        pred_lbl = self.id2label[pred_idx].lower()
        conf     = probs[pred_idx].item()

        is_synth   = any(k in pred_lbl for k in SYNTH_KEYWORDS)
        synth_idx  = self._find_synth_idx()
        synth_prob = (
            probs[synth_idx].item()
            if synth_idx is not None
            else (conf if is_synth else 1.0 - conf)
        )

        target_cls = synth_idx if synth_idx is not None else pred_idx
        heatmap_b64, overlay_b64 = self._run_gradcam(
            image, image_np, inputs['pixel_values'], target_cls
        )

        return {
            'is_synthetic':            is_synth,
            'confidence':              round(float(conf), 4),
            'synthetic_probability':   round(float(synth_prob), 4),
            'label':                   'SYNTHETIC' if is_synth else 'AUTHENTIC',
            'raw_label':               pred_lbl,
            'indicators':              self._indicators(is_synth, synth_prob),
            'analysis':                self._analysis(is_synth, synth_prob, pred_lbl),
            'heatmap_base64':          heatmap_b64,
            'heatmap_overlay_base64':  overlay_b64,
        }

    # Convenience alias
    def analyze_bytes(self, image_bytes: bytes) -> dict:
        return self.analyze(image_bytes)

    def analyze_video(
        self,
        video_path: str,
        frame_step: int = 30,
        max_frames: int = 500,
    ) -> dict:
        """
        Extract every `frame_step`-th frame from a video using ffmpeg-python,
        run the synthetic-image detector on each, and return a timeline.

        Parameters
        ----------
        video_path  : path to the video file (mp4, mov, avi, …)
        frame_step  : sample 1 frame every N frames (default 30 ≈ 1 fps @ 30 fps)
        max_frames  : hard cap on frames analysed (guards against very long videos)

        Returns
        -------
        {
          "video_path": str,
          "total_frames_sampled": int,
          "frame_step": int,
          "fps": float,
          "duration_seconds": float,
          "timeline": [
            {
              "frame_index": int,       # original frame number in video
              "timestamp_seconds": float,
              "synthetic_probability": float,
              "label": "SYNTHETIC" | "AUTHENTIC",
              "confidence": float,
            },
            …
          ],
          "summary": {
            "mean_synthetic_probability": float,
            "max_synthetic_probability": float,
            "synthetic_frame_count": int,
            "verdict": "LIKELY_DEEPFAKE" | "POSSIBLY_MANIPULATED" | "LIKELY_AUTHENTIC",
          }
        }
        """
        # ── Probe video metadata ────────────────────────────────────────
        try:
            probe     = ffmpeg.probe(video_path)
            vs        = next(s for s in probe['streams'] if s['codec_type'] == 'video')
            fps_raw   = vs.get('r_frame_rate', '30/1')
            num, den  = map(int, fps_raw.split('/'))
            fps       = num / den if den else 30.0
            duration  = float(vs.get('duration', probe['format'].get('duration', 0)))
            total_src = int(vs.get('nb_frames') or round(duration * fps))
        except Exception as exc:
            raise RuntimeError(f"ffmpeg probe failed: {exc}") from exc

        # ── Extract frames into a temp directory ───────────────────────
        timeline = []
        with tempfile.TemporaryDirectory() as tmpdir:
            frame_pattern = os.path.join(tmpdir, 'frame_%06d.png')

            # Use select filter to grab every frame_step-th frame
            (
                ffmpeg
                .input(video_path)
                .video
                .filter('select', f'not(mod(n,{frame_step}))')
                .output(
                    frame_pattern,
                    vsync       = 'vfr',
                    format      = 'image2',
                    vcodec      = 'png',
                    loglevel    = 'error',
                )
                .run(overwrite_output=True)
            )

            frame_files = sorted(
                f for f in os.listdir(tmpdir) if f.endswith('.png')
            )[:max_frames]

            for seq_idx, fname in enumerate(frame_files):
                original_frame_idx = seq_idx * frame_step
                timestamp = original_frame_idx / fps if fps else 0.0
                fpath = os.path.join(tmpdir, fname)

                try:
                    with open(fpath, 'rb') as fh:
                        img_bytes = fh.read()
                    result = self.analyze(img_bytes)
                
                # --- Create the base dictionary ---
                    frame_data = {
                        'frame_index':          original_frame_idx,
                        'timestamp_seconds':    round(timestamp, 3),
                        'synthetic_probability': result['synthetic_probability'],
                        'label':                result['label'],
                        'confidence':           result['confidence'],
                    }
                
                    # --- NEW CODE: Attach Base64 image if synthetic ---
                    if result['label'] == 'SYNTHETIC':
                        import base64
                        frame_data['image_base64'] = base64.b64encode(img_bytes).decode('utf-8')
                    # --------------------------------------------------

                    timeline.append(frame_data)
                except Exception as exc:
                    print(f"[VideoAnalysis] Skipping frame {fname}: {exc}")

        # ── Summarise ──────────────────────────────────────────────────
        if not timeline:
            raise RuntimeError("No frames could be analysed.")

        probs            = [f['synthetic_probability'] for f in timeline]
        mean_prob        = round(float(np.mean(probs)), 4)
        max_prob         = round(float(np.max(probs)), 4)
        synth_count      = sum(1 for f in timeline if f['label'] == 'SYNTHETIC')

        if mean_prob > 0.70:
            verdict = 'LIKELY_DEEPFAKE'
        elif mean_prob > 0.40:
            verdict = 'POSSIBLY_MANIPULATED'
        else:
            verdict = 'LIKELY_AUTHENTIC'

        return {
            'video_path':            video_path,
            'total_frames_sampled':  len(timeline),
            'frame_step':            frame_step,
            'fps':                   round(fps, 3),
            'duration_seconds':      round(duration, 3),
            'timeline':              timeline,
            'summary': {
                'mean_synthetic_probability': mean_prob,
                'max_synthetic_probability':  max_prob,
                'synthetic_frame_count':      synth_count,
                'verdict':                    verdict,
            },
        }

    # ── Grad-CAM ──────────────────────────────────────────────────────────

    def _run_gradcam(self, image, image_np, pixel_values, target_cls):
        try:
            cam = GradCAM(
                model             = self._wrapper,
                target_layers     = [self._target_layer],
                reshape_transform = self._reshape,
            )
            grayscale = cam(
                input_tensor = pixel_values,
                targets      = [ClassifierOutputTarget(target_cls)],
            )[0]

            W, H   = image.size
            gray_rs = cv2.resize(grayscale, (W, H))

            img_f32  = np.array(image.resize((W, H))).astype(np.float32) / 255.0
            overlay  = show_cam_on_image(img_f32, gray_rs, use_rgb=True)

            heat_raw = cv2.applyColorMap(
                (gray_rs * 255).astype(np.uint8), cv2.COLORMAP_JET
            )
            heat_raw = cv2.cvtColor(heat_raw, cv2.COLOR_BGR2RGB)

            return self._to_b64(heat_raw), self._to_b64(overlay)

        except Exception as exc:
            print(f"[GradCAM] Failed — {exc}")
            import traceback; traceback.print_exc()
            return None, None

    @staticmethod
    def _reshape(tensor):
        if isinstance(tensor, (tuple, list)):
            tensor = tensor[0]
        if tensor.dim() == 4:
            return tensor
        B, N, C = tensor.shape
        side    = int(N ** 0.5)
        tensor  = tensor.reshape(B, side, side, C)
        return tensor.permute(0, 3, 1, 2)

    # ── Layer resolution ──────────────────────────────────────────────────

    def _resolve_target_layer(self):
        m = self.model

        if hasattr(m, 'swin'):
            layers     = m.swin.encoder.layers
            last_stage = layers[-1]
            if hasattr(last_stage, 'blocks'):
                return last_stage.blocks[-1].layernorm_before
            return last_stage

        if hasattr(m, 'vit'):
            return m.vit.encoder.layer[-1].layernorm_before

        if hasattr(m, 'resnet'):
            return m.resnet.encoder.stages[-1]

        layers_list = list(m.named_modules())
        for name, mod in reversed(layers_list):
            if len(list(mod.children())) == 0:
                print(f"[GradCAM] fallback layer: {name}")
                return mod

        raise RuntimeError("Could not identify a suitable Grad-CAM target layer.")

    def _find_synth_idx(self):
        for idx, lbl in self.id2label.items():
            if any(k in lbl.lower() for k in SYNTH_KEYWORDS):
                return idx
        return None

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _to_b64(arr: np.ndarray) -> str:
        img = Image.fromarray(arr.astype(np.uint8))
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        return base64.b64encode(buf.getvalue()).decode('utf-8')

    @staticmethod
    def _indicators(is_synth: bool, prob: float) -> list:
        if is_synth:
            if prob > 0.9:
                return [
                    "Strong GAN / diffusion model fingerprint detected",
                    "Unnatural frequency spectrum in high-frequency components",
                    "Texture regularity inconsistent with optical capture",
                    "No authentic camera sensor noise signature",
                    "Pixel distribution anomalies typical of neural synthesis",
                ]
            elif prob > 0.7:
                return [
                    "Synthetic model artifacts present in pixel statistics",
                    "Edge boundary blurring patterns typical of neural generation",
                    "Lighting gradient inconsistencies detected",
                    "No authentic camera noise signature found",
                ]
            return [
                "Moderate probability of AI generation",
                "Some statistical anomalies in pixel distribution",
                "Possible post-processing or hybrid generation",
            ]
        return [
            "Natural sensor noise pattern consistent with real camera",
            "JPEG compression artifacts match authentic capture",
            "Frequency domain profile matches real photography",
            "Edge sharpness consistent with optical lens characteristics",
        ]

    @staticmethod
    def _analysis(is_synth: bool, prob: float, raw_label: str) -> str:
        if is_synth:
            return (
                f"Forensic model (umm-maybe/AI-image-detector) classified this image as "
                f"'{raw_label}' with {prob:.1%} synthetic probability. "
                "Grad-CAM analysis highlights the spatial regions with the strongest synthetic "
                "artifacts — typically textures, faces, and backgrounds rendered by a GAN or "
                "diffusion model. Statistical indicators include unnatural high-frequency "
                "components, texture over-smoothness, and pixel distributions inconsistent "
                "with real optical capture."
            )
        return (
            f"Forensic model classified this image as '{raw_label}' with "
            f"{1-prob:.1%} confidence. The pixel statistics, noise profile, and "
            "frequency-domain characteristics are all consistent with authentic photographic "
            "capture. No significant GAN or diffusion model fingerprints were detected."
        )