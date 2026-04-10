"""
SynthScan — Flask Backend
Runs a real ML model (Hugging Face AI image detector) + Grad-CAM heatmaps

New in this revision
────────────────────
  • POST /analyze-batch   — accepts multiple images in one request, returns a
                            JSON array of per-image forensic results
  • POST /flag            — REST endpoint for external applications to submit a
                            single image URL or file and receive a verdict
  • GET  /monitor/start   — kick off the watchdog folder monitor in a background
                            thread (folder configurable via query-param ?path=...)
  • GET  /monitor/stop    — gracefully stop the background monitor
  • GET  /monitor/status  — return current monitor state + last N events
  • GET  /monitor/events  — SSE stream: push real-time scan events to the browser
"""

from __future__ import annotations

import io
import os
import queue
import threading
import time
import urllib.request
from pathlib import Path
from typing import Generator

from flask import Flask, Response, jsonify, request, send_from_directory, stream_with_context
from flask_cors import CORS
from groq import Groq

# ── Watchdog (optional dep) ──────────────────────────────────────────────────
try:
    from watchdog.events import FileSystemEventHandler
    from watchdog.observers import Observer
    WATCHDOG_OK = True
except ImportError:
    WATCHDOG_OK = False

    # Stubs so `class _SynthScanHandler(FileSystemEventHandler)` below
    # can always be defined, even when watchdog is not installed.
    class FileSystemEventHandler:          # noqa: F811
        def on_created(self, event): pass

    class Observer:                        # noqa: F811
        def schedule(self, *a, **kw): pass
        def start(self): pass
        def stop(self): pass
        def join(self, timeout=None): pass


app = Flask(__name__)
CORS(app)

ALLOWED_MIMES   = {'image/jpeg', 'image/png', 'image/webp', 'image/gif'}
ALLOWED_EXTS    = {'.jpg', '.jpeg', '.png', '.webp', '.gif'}
MAX_BATCH_SIZE  = 20          # hard cap on batch endpoint
MAX_EVENT_QUEUE = 500         # ring-buffer size for SSE


# ── Lazy-load model ─────────────────────────────────────────────────────────

_detector = None
_detector_lock = threading.Lock()


def get_detector():
    global _detector
    if _detector is None:
        with _detector_lock:
            if _detector is None:
                from model import SyntheticImageDetector
                print("[SynthScan] Loading AI detection model…")
                _detector = SyntheticImageDetector()
                print("[SynthScan] Model loaded.")
    return _detector


# ═══════════════════════════════════════════════════════════════════════════
# PAGE ROUTING
# ═══════════════════════════════════════════════════════════════════════════

@app.route('/')
def home():
    return send_from_directory('.', 'index.html')

@app.route('/index.html')
def index_redirect():
    return send_from_directory('.', 'index.html')

@app.route('/analyze.html')
def analyze_page():
    return send_from_directory('.', 'analyze.html')

@app.route('/batch.html')
def batch_page():
    return send_from_directory('.', 'batch.html')

@app.route('/video.html')
def video_page():
    return send_from_directory('.', 'video.html')


# ═══════════════════════════════════════════════════════════════════════════
# SINGLE-IMAGE ANALYSIS  (unchanged from original)
# ═══════════════════════════════════════════════════════════════════════════

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if file.mimetype not in ALLOWED_MIMES:
        return jsonify({'error': f'Unsupported file type: {file.mimetype}'}), 400

    try:
        result = get_detector().analyze(file.read())
        return jsonify(result)
    except Exception as e:
        print(f"[SynthScan] Analysis error: {e}")
        return jsonify({'error': str(e)}), 500


# ═══════════════════════════════════════════════════════════════════════════
# BATCH ANALYSIS  — POST /analyze-batch
# ═══════════════════════════════════════════════════════════════════════════

@app.route('/analyze-batch', methods=['POST'])
def analyze_batch():
    """
    Accept up to MAX_BATCH_SIZE images in a single multipart/form-data request.
    Field name must be 'images' (repeatable).

    Returns:
        {
          "total": N,
          "results": [
            {"filename": "…", "index": 0, <all model fields>},
            …
          ]
        }
    """
    files = request.files.getlist('images')
    if not files:
        return jsonify({'error': 'No images provided. Use field name "images".'}), 400

    if len(files) > MAX_BATCH_SIZE:
        return jsonify({
            'error': f'Batch limit is {MAX_BATCH_SIZE} images. Received {len(files)}.'
        }), 400

    detector = get_detector()
    results  = []

    for idx, file in enumerate(files):
        entry: dict = {'filename': file.filename, 'index': idx}

        if file.mimetype not in ALLOWED_MIMES:
            entry['error'] = f'Unsupported MIME type: {file.mimetype}'
            results.append(entry)
            continue

        try:
            analysis = detector.analyze(file.read())
            entry.update(analysis)
        except Exception as exc:
            entry['error'] = str(exc)

        results.append(entry)

    return jsonify({'total': len(results), 'results': results})


# ═══════════════════════════════════════════════════════════════════════════
# EXTERNAL-FLAG ENDPOINT  — POST /flag
# ═══════════════════════════════════════════════════════════════════════════

@app.route('/flag', methods=['POST'])
def flag():
    """
    Designed for external applications that want to submit an image for
    forensic analysis without a full form upload.

    Accepts (mutually exclusive, checked in order):
      1. multipart field  'image'   — raw file upload  (same as /analyze)
      2. JSON body        {"url": "https://…"}  — image fetched server-side

    Returns the standard model result dict plus:
      "source": "upload" | "url"
      "flagged_at": <unix timestamp>
    """
    source     = None
    image_bytes = None

    # ── option 1: file upload ──
    if 'image' in request.files:
        file = request.files['image']
        if file.mimetype not in ALLOWED_MIMES:
            return jsonify({'error': f'Unsupported MIME type: {file.mimetype}'}), 400
        image_bytes = file.read()
        source = 'upload'

    # ── option 2: URL ──
    elif request.is_json:
        data = request.get_json(silent=True) or {}
        url  = data.get('url', '').strip()
        if not url:
            return jsonify({'error': 'Provide either a file upload or {"url": "…"}'}), 400
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'SynthScan/1.0'})
            with urllib.request.urlopen(req, timeout=15) as resp:
                content_type = resp.headers.get('Content-Type', '').split(';')[0].strip()
                if content_type not in ALLOWED_MIMES:
                    return jsonify({'error': f'Remote URL returned unsupported MIME: {content_type}'}), 400
                image_bytes = resp.read()
        except Exception as exc:
            return jsonify({'error': f'Failed to fetch URL: {exc}'}), 400
        source = 'url'

    else:
        return jsonify({'error': 'No image supplied. Send a file or JSON {"url": "…"}'}), 400

    try:
        result = get_detector().analyze(image_bytes)
        result['source']     = source
        result['flagged_at'] = int(time.time())
        return jsonify(result)
    except Exception as exc:
        print(f"[Flag] Error: {exc}")
        return jsonify({'error': str(exc)}), 500


# ═══════════════════════════════════════════════════════════════════════════
# WATCHDOG FOLDER MONITOR
# ═══════════════════════════════════════════════════════════════════════════

# Shared state (guarded by _monitor_lock)
_monitor_lock    = threading.Lock()
_observer        = None          # watchdog Observer thread
_monitor_path    = None          # folder currently being watched
_monitor_running = False
_event_queue: queue.Queue = queue.Queue(maxsize=MAX_EVENT_QUEUE)  # SSE ring buffer
_event_log: list           = []   # last 200 events for /monitor/status


def _push_event(payload: dict):
    """Add an event to the SSE queue and the rolling log."""
    global _event_log
    payload.setdefault('ts', int(time.time()))
    # Queue: drop oldest if full
    if _event_queue.full():
        try:
            _event_queue.get_nowait()
        except queue.Empty:
            pass
    _event_queue.put(payload)
    _event_log.append(payload)
    if len(_event_log) > 200:
        _event_log = _event_log[-200:]


class _SynthScanHandler(FileSystemEventHandler):
    """Watchdog event handler: runs forensic analysis on new image files."""

    def on_created(self, event):
        if event.is_directory:
            return
        path = Path(event.src_path)
        if path.suffix.lower() not in ALLOWED_EXTS:
            return

        _push_event({'type': 'detected', 'path': str(path), 'status': 'queued'})
        # Run analysis in a worker thread so we don't block watchdog
        threading.Thread(target=self._analyze, args=(path,), daemon=True).start()

    def _analyze(self, path: Path):
        _push_event({'type': 'analyzing', 'path': str(path)})
        try:
            image_bytes = path.read_bytes()
            result      = get_detector().analyze(image_bytes)
            _push_event({
                'type':        'result',
                'path':        str(path),
                'filename':    path.name,
                'is_synthetic': result['is_synthetic'],
                'label':        result['label'],
                'confidence':   result['confidence'],
                'synthetic_probability': result['synthetic_probability'],
                'risk': 'HIGH' if result['synthetic_probability'] >= 0.85
                        else 'MED' if result['synthetic_probability'] >= 0.55
                        else 'LOW',
                'indicators':  result.get('indicators', []),
            })
        except Exception as exc:
            _push_event({'type': 'error', 'path': str(path), 'message': str(exc)})


@app.route('/monitor/start', methods=['GET'])
def monitor_start():
    """
    Start the watchdog monitor.
    Query param: ?path=/absolute/folder/path   (defaults to ./watch_folder)
    """
    global _observer, _monitor_path, _monitor_running

    if not WATCHDOG_OK:
        return jsonify({'error': 'watchdog package not installed. Run: pip install watchdog'}), 500

    folder = request.args.get('path', os.path.join(os.getcwd(), 'watch_folder'))
    folder = os.path.abspath(folder)

    with _monitor_lock:
        if _monitor_running:
            return jsonify({
                'status':  'already_running',
                'path':    _monitor_path,
                'message': 'Monitor is already active. Stop it first to change folder.'
            })

        os.makedirs(folder, exist_ok=True)
        handler   = _SynthScanHandler()
        _observer = Observer()
        _observer.schedule(handler, folder, recursive=True)
        _observer.start()
        _monitor_path    = folder
        _monitor_running = True

    _push_event({'type': 'monitor_started', 'path': folder})
    print(f"[Monitor] Watching: {folder}")
    return jsonify({'status': 'started', 'path': folder})


@app.route('/monitor/stop', methods=['GET'])
def monitor_stop():
    global _observer, _monitor_path, _monitor_running

    with _monitor_lock:
        if not _monitor_running:
            return jsonify({'status': 'not_running'})

        _observer.stop()
        _observer.join(timeout=5)
        _observer        = None
        _monitor_running = False
        old_path         = _monitor_path
        _monitor_path    = None

    _push_event({'type': 'monitor_stopped', 'path': old_path})
    return jsonify({'status': 'stopped', 'path': old_path})


@app.route('/monitor/status', methods=['GET'])
def monitor_status():
    return jsonify({
        'running':     _monitor_running,
        'path':        _monitor_path,
        'event_count': len(_event_log),
        'recent':      _event_log[-20:],   # last 20 events
    })


@app.route('/monitor/events', methods=['GET'])
def monitor_events():
    """
    Server-Sent Events stream.
    The browser connects here and receives real-time forensic events as
    JSON-encoded SSE data lines.
    """
    def _generate() -> Generator[str, None, None]:
        # Send a heartbeat every 15 s so the connection stays alive
        heartbeat_interval = 15
        last_heartbeat     = time.time()

        while True:
            try:
                event = _event_queue.get(timeout=1)
                import json
                yield f"data: {json.dumps(event)}\n\n"
                last_heartbeat = time.time()
            except queue.Empty:
                if time.time() - last_heartbeat >= heartbeat_interval:
                    yield ": heartbeat\n\n"
                    last_heartbeat = time.time()

    return Response(
        stream_with_context(_generate()),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no',
        }
    )


# ═══════════════════════════════════════════════════════════════════════════
# HEALTH
# ═══════════════════════════════════════════════════════════════════════════

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status':          'ok',
        'model':           'umm-maybe/AI-image-detector',
        'monitor_running': _monitor_running,
        'monitor_path':    _monitor_path,
    })


# ═══════════════════════════════════════════════════════════════════════════
# CHAT  (unchanged)
# ═══════════════════════════════════════════════════════════════════════════

groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

SYSTEM_PROMPT = """You are SynthScan Assistant, an expert in AI-generated image detection.
Answer in 1-3 short sentences max. Be direct and concise — no bullet points, no lengthy explanations.
The SynthScan also offers video analysis by breaking the video to frame by frame images. If asked about analysis results, explain simply what they mean. Never over-explain."""


@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    if not data or 'message' not in data:
        return jsonify({'error': 'No message provided'}), 400

    history      = data.get('history', [])
    user_message = data['message']

    try:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for m in history:
            messages.append({
                "role":    "user" if m["role"] == "user" else "assistant",
                "content": m["text"],
            })
        messages.append({"role": "user", "content": user_message})

        completion = groq_client.chat.completions.create(
            model       = "openai/gpt-oss-120b",
            messages    = messages,
            max_tokens  = 1024,
            temperature = 0.7,
        )
        return jsonify({'reply': completion.choices[0].message.content})

    except Exception as e:
        print(f"[Chat] Error: {e}")
        return jsonify({'error': str(e)}), 500

# ═══════════════════════════════════════════════════════════════════════════
# Video(deepFake)
# ═══════════════════════════════════════════════════════════════════════════

ALLOWED_VIDEO_MIMES = {'video/mp4', 'video/quicktime', 'video/x-msvideo', 'video/webm'}
ALLOWED_VIDEO_EXTS  = {'.mp4', '.mov', '.avi', '.webm'}

@app.route('/analyze-video', methods=['POST'])
def analyze_video():
    """
    Accept a video file upload, extract every N-th frame via ffmpeg,
    run the detector on each, and return a synthetic-probability timeline.

    Form fields:
      - 'video'      : the video file (required)
      - 'frame_step' : int, sample every N frames (optional, default 30)
      - 'max_frames' : int, hard cap on frames analysed (optional, default 500)
    """
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided. Use field name "video".'}), 400

    file = request.files['video']
    ext  = os.path.splitext(file.filename)[1].lower()

    if file.mimetype not in ALLOWED_VIDEO_MIMES and ext not in ALLOWED_VIDEO_EXTS:
        return jsonify({'error': f'Unsupported video type: {file.mimetype}'}), 400

    frame_step = int(request.form.get('frame_step', 30))
    max_frames = int(request.form.get('max_frames', 500))

    # Save upload to a temp file (ffmpeg needs a real path)
    import tempfile
    suffix = ext or '.mp4'
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        file.save(tmp.name)
        tmp_path = tmp.name

    try:
        result = get_detector().analyze_video(tmp_path, frame_step=frame_step, max_frames=max_frames)
        return jsonify(result)
    except Exception as exc:
        print(f"[VideoAnalysis] Error: {exc}")
        return jsonify({'error': str(exc)}), 500
    finally:
        os.unlink(tmp_path)   # always clean up

# ═══════════════════════════════════════════════════════════════════════════
# WEB PAGE IMAGE SCANNER  — GET /webscan.html  +  POST /scan-url
# ═══════════════════════════════════════════════════════════════════════════

WEBSCAN_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (compatible; SynthScan/1.0)',
    'Accept': 'text/html,application/xhtml+xml,*/*;q=0.8',
}
MAX_WEBSCAN_IMAGES = 50   # hard cap per page scan


def _fetch_url_bytes(url: str, timeout: int = 15) -> tuple[bytes, str]:
    """Fetch a URL and return (bytes, content_type)."""
    req = urllib.request.Request(url, headers=WEBSCAN_HEADERS)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read(), resp.headers.get('Content-Type', '').split(';')[0].strip()


def _absolute(src: str, base: str) -> str | None:
    """Resolve a potentially relative image src against the page base URL."""
    if not src or src.startswith('data:'):
        return None
    from urllib.parse import urljoin, urlparse
    abs_url = urljoin(base, src.strip())
    parsed  = urlparse(abs_url)
    return abs_url if parsed.scheme in ('http', 'https') else None


@app.route('/webscan.html')
def webscan_page():
    return send_from_directory('.', 'webscan.html')


@app.route('/scan-url', methods=['POST'])
def scan_url():
    """
    Scrape all <img> tags from a web page and run the AI detector on each.

    Body (JSON): { "url": "https://example.com", "max_images": 20 }

    Returns:
    {
      "page_url": str,
      "images_found": int,
      "images_scanned": int,
      "results": [
        {
          "src": str,             # absolute image URL
          "alt": str,
          "is_synthetic": bool,
          "label": str,
          "confidence": float,
          "synthetic_probability": float,
          "risk": "HIGH"|"MED"|"LOW",
          "error": str | null
        }, …
      ],
      "summary": {
        "synthetic_count": int,
        "authentic_count": int,
        "error_count": int,
        "mean_synthetic_probability": float,
        "highest_risk_src": str | null,
      }
    }
    """
    from bs4 import BeautifulSoup

    data       = request.get_json(silent=True) or {}
    page_url   = (data.get('url') or '').strip()
    max_images = min(int(data.get('max_images', 20)), MAX_WEBSCAN_IMAGES)

    if not page_url:
        return jsonify({'error': 'Provide a JSON body with {"url": "https://…"}'}), 400
    if not page_url.startswith(('http://', 'https://')):
        return jsonify({'error': 'URL must start with http:// or https://'}), 400

    # ── Fetch & parse the HTML page ──────────────────────────────────────
    try:
        html_bytes, _ = _fetch_url_bytes(page_url)
        soup          = BeautifulSoup(html_bytes, 'html.parser')
    except Exception as exc:
        return jsonify({'error': f'Could not fetch page: {exc}'}), 400

    # Collect unique absolute image URLs from <img src> and <img srcset>
    seen      = set()
    img_srcs  = []   # list of (abs_url, alt_text)

    for tag in soup.find_all('img'):
        # primary src
        for attr in ('src', 'data-src', 'data-lazy-src'):
            raw = tag.get(attr, '')
            abs_url = _absolute(raw, page_url)
            if abs_url and abs_url not in seen:
                seen.add(abs_url)
                img_srcs.append((abs_url, tag.get('alt', '')))
                break

        # srcset candidates
        srcset = tag.get('srcset', '')
        if srcset:
            for part in srcset.split(','):
                raw = part.strip().split()[0]
                abs_url = _absolute(raw, page_url)
                if abs_url and abs_url not in seen:
                    seen.add(abs_url)
                    img_srcs.append((abs_url, tag.get('alt', '')))

    images_found   = len(img_srcs)
    img_srcs       = img_srcs[:max_images]   # apply cap
    images_scanned = len(img_srcs)

    # ── Run detector on each image ────────────────────────────────────────
    detector = get_detector()
    results  = []

    for src, alt in img_srcs:
        entry = {'src': src, 'alt': alt, 'error': None}
        try:
            img_bytes, content_type = _fetch_url_bytes(src, timeout=10)
            # Skip non-image content types (icons, SVGs served as text, etc.)
            if content_type and not content_type.startswith('image/'):
                entry['error'] = f'Skipped — MIME {content_type}'
                results.append(entry)
                continue
            # Skip tiny blobs (likely tracking pixels, <1 KB)
            if len(img_bytes) < 1024:
                entry['error'] = 'Skipped — image too small (<1 KB)'
                results.append(entry)
                continue

            analysis = detector.analyze(img_bytes)
            prob     = analysis['synthetic_probability']
            entry.update({
                'is_synthetic':          analysis['is_synthetic'],
                'label':                 analysis['label'],
                'confidence':            analysis['confidence'],
                'synthetic_probability': prob,
                'risk':                  'HIGH' if prob >= 0.85 else 'MED' if prob >= 0.55 else 'LOW',
            })
        except Exception as exc:
            entry['error'] = str(exc)

        results.append(entry)

    # ── Summarise ──────────────────────────────────────────────────────────
    analysed = [r for r in results if r.get('error') is None]
    synth    = [r for r in analysed if r.get('is_synthetic')]
    errs     = [r for r in results  if r.get('error')]
    probs    = [r['synthetic_probability'] for r in analysed]
    highest  = max(analysed, key=lambda r: r['synthetic_probability'], default=None)

    return jsonify({
        'page_url':       page_url,
        'images_found':   images_found,
        'images_scanned': images_scanned,
        'results':        results,
        'summary': {
            'synthetic_count':            len(synth),
            'authentic_count':            len(analysed) - len(synth),
            'error_count':                len(errs),
            'mean_synthetic_probability': round(sum(probs) / len(probs), 4) if probs else 0.0,
            'highest_risk_src':           highest['src'] if highest else None,
        },
    })


# ═══════════════════════════════════════════════════════════════════════════
# USER REVIEWS  — GET /reviews  +  POST /reviews
# ═══════════════════════════════════════════════════════════════════════════

import json as _json

REVIEWS_FILE = os.path.join(os.path.dirname(__file__), 'reviews.json')
_reviews_lock = threading.Lock()


def _load_reviews() -> list:
    if not os.path.exists(REVIEWS_FILE):
        return []
    try:
        with open(REVIEWS_FILE, 'r', encoding='utf-8') as f:
            return _json.load(f)
    except Exception:
        return []


def _save_reviews(reviews: list) -> None:
    with open(REVIEWS_FILE, 'w', encoding='utf-8') as f:
        _json.dump(reviews, f, ensure_ascii=False, indent=2)


@app.route('/reviews', methods=['GET'])
def get_reviews():
    """Return all reviews, newest first."""
    with _reviews_lock:
        reviews = _load_reviews()
    return jsonify({'reviews': list(reversed(reviews))})


@app.route('/reviews', methods=['POST'])
def post_review():
    """
    Submit a new review.
    Body (JSON): { "name": str, "rating": int (1-5), "text": str }
    """
    data = request.get_json(silent=True) or {}
    name   = str(data.get('name', 'Anonymous')).strip()[:80]
    rating = data.get('rating')
    text   = str(data.get('text', '')).strip()[:1000]

    if not name:
        name = 'Anonymous'
    if rating is None or not isinstance(rating, int) or not (1 <= rating <= 5):
        return jsonify({'error': 'rating must be an integer between 1 and 5'}), 400
    if not text:
        return jsonify({'error': 'review text is required'}), 400

    review = {
        'id':         int(time.time() * 1000),
        'name':       name,
        'rating':     rating,
        'text':       text,
        'created_at': int(time.time()),
    }

    with _reviews_lock:
        reviews = _load_reviews()
        reviews.append(review)
        _save_reviews(reviews)

    return jsonify({'ok': True, 'review': review}), 201


# ═══════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 7860))
    print(f"[SynthScan] Starting server on http://localhost:{port}")
    app.run(debug=True, port=port, host='0.0.0.0')