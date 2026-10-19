"""
Local server for the universal-approximator p5 UI + Graphwar field capture.

Usage:
    python tools/approximator_server.py
    python tools/approximator_server.py --record-training
"""

import argparse
import json
import sys
import webbrowser
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

try:
    from flask import Flask, request, jsonify
    HAS_FLASK = True
except Exception:
    Flask = None
    request = None
    jsonify = None
    HAS_FLASK = False

# Optional symbolic engines for /api/derive.  The server must still import and
# serve captures when these are unavailable.
USE_SYMENGINE = False
se = None
try:
    import symengine as se
    USE_SYMENGINE = True
except Exception:
    se = None
    USE_SYMENGINE = False

USE_SYMPY = False
symbols = None
sympify = None
diff = None
simplify = None
try:
    from sympy import symbols as _symbols, sympify as _sympify, diff as _diff, simplify as _simplify
    symbols, sympify, diff, simplify = _symbols, _sympify, _diff, _simplify
    USE_SYMPY = True
except Exception:
    USE_SYMPY = False

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from core.game_capture import capture_game_field

APPROX_DIR = ROOT_DIR / "Visuals in p5.js" / "universal-approximator"
RECORDINGS_DIR = ROOT_DIR / "outputs" / "recordings"
HOST = "127.0.0.1"
PORT = 8765
SERVER_URL = f"http://{HOST}:{PORT}/"
MAX_RECORDING_BYTES = 256 * 1024 * 1024
RECORDING_ENABLED = False

app = Flask(__name__) if HAS_FLASK else None


class ApproximatorHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        print(f"[server] {self.address_string()} {fmt % args}")

    def _send_json(self, status, payload):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: Path):
        if not path.is_file():
            self.send_error(404, "Not found")
            return

        content = path.read_bytes()
        if path.suffix == ".html":
            ctype = "text/html; charset=utf-8"
        elif path.suffix == ".js":
            ctype = "application/javascript; charset=utf-8"
        elif path.suffix == ".css":
            ctype = "text/css; charset=utf-8"
        else:
            ctype = "application/octet-stream"

        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store, max-age=0")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def do_GET(self):
        if self.path == "/api/config":
            self._send_json(200, {"record_training_enabled": RECORDING_ENABLED})
            return

        if self.path in ("/", "/index.html"):
            self._send_file(APPROX_DIR / "index.html")
            return

        rel = self.path.lstrip("/")
        target = (APPROX_DIR / rel).resolve()
        if not str(target).startswith(str(APPROX_DIR.resolve())):
            self.send_error(403, "Forbidden")
            return
        self._send_file(target)

    def do_POST(self):
        if self.path == "/api/recordings":
            if not RECORDING_ENABLED:
                self.send_error(404, "Not found")
                return
            self._save_recording()
            return

        if self.path != "/api/capture":
            self.send_error(404, "Not found")
            return

        try:
            result = capture_game_field()
        except Exception as exc:
            self._send_json(500, {"ok": False, "error": str(exc)})
            return

        if result.get("ok"):
            stats = result.get("forbidden_stats")
            if stats:
                print(
                    "[forbidden mask] "
                    f"components={stats['components']} "
                    f"cells={stats['grid_forbidden_cells']}/"
                    f"{stats['grid_total_cells']}"
                )
            elif result.get("forbidden_error"):
                print(f"[forbidden mask] ERROR: {result['forbidden_error']}")

        status = 200 if result.get("ok") else 400
        self._send_json(status, result)

    def _save_recording(self):
        content_type = self.headers.get("Content-Type", "").lower()
        if not content_type.startswith("video/webm"):
            self._send_json(415, {"ok": False, "error": "Expected a WebM video."})
            return

        try:
            length = int(self.headers.get("Content-Length", ""))
        except ValueError:
            self._send_json(411, {"ok": False, "error": "Missing or invalid Content-Length."})
            return
        if length <= 0 or length > MAX_RECORDING_BYTES:
            self._send_json(
                413,
                {"ok": False, "error": f"Recording must be between 1 byte and {MAX_RECORDING_BYTES // (1024 * 1024)} MB."},
            )
            return

        data = self.rfile.read(length)
        if len(data) != length:
            self._send_json(400, {"ok": False, "error": "Incomplete recording upload."})
            return

        kind = self.headers.get("X-Recording-Kind", "training").lower()
        if kind not in {"training", "trajectory"}:
            self._send_json(400, {"ok": False, "error": "Unknown recording kind."})
            return

        try:
            RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
            filename = f"{kind}-{datetime.now():%Y%m%d-%H%M%S-%f}.webm"
            destination = RECORDINGS_DIR / filename
            destination.write_bytes(data)
        except OSError as exc:
            self._send_json(500, {"ok": False, "error": f"Could not save recording: {exc}"})
            return

        relative_path = destination.relative_to(ROOT_DIR).as_posix()
        print(f"[recording] saved {relative_path} ({length} bytes)")
        self._send_json(201, {"ok": True, "relative_path": relative_path})


def parse_args():
    parser = argparse.ArgumentParser(description="Serve the GraphBot formula-builder UI.")
    parser.add_argument(
        "--record-training",
        action="store_true",
        help="show the private training-animation recorder and allow saving WebM files",
    )
    return parser.parse_args()


def derive():
    try:
        data = request.get_json(silent=True) or {}
        expr = (data.get('expr') or '').strip()
        order = int(data.get('order') or 0)
        if order < 0:
            order = 0
        if order > 3:
            order = 3
        if USE_SYMENGINE and se is not None:
            try:
                x = se.Symbol('x')
                f = se.sympify(expr)
                for _ in range(order):
                    f = se.diff(f, x)
                return jsonify({'ok': True, 'formula': _to_js_formula(f)})
            except Exception as e:
                pass  # fallback
        if USE_SYMPY:
            try:
                x = symbols('x')
                f = sympify(expr, evaluate=True)
                for _ in range(order):
                    f = diff(f, x)
                try:
                    f = simplify(f)
                except Exception:
                    pass
                return jsonify({'ok': True, 'formula': _to_js_formula(f)})
            except Exception as e:
                return jsonify({'ok': False, 'error': str(e)[:200]})
        return jsonify({'ok': False, 'error': 'no symbolic engine available'})
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)[:200]})



if HAS_FLASK and app is not None:
    app.add_url_rule("/api/derive", "derive", derive, methods=["POST"])


def main():
    global RECORDING_ENABLED
    args = parse_args()
    RECORDING_ENABLED = args.record_training
    if not APPROX_DIR.is_dir():
        raise SystemExit(f"Approximator folder not found: {APPROX_DIR}")

    httpd = ThreadingHTTPServer((HOST, PORT), ApproximatorHandler)
    print(f"Approximator server: {SERVER_URL}")
    print("Graphwar must be running. Use «Capture field» in the UI.")
    if RECORDING_ENABLED:
        print("Training-video recorder is enabled; files are saved in outputs/recordings.")
    if not webbrowser.open(SERVER_URL, new=2):
        print("Could not open the default browser automatically.")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
def _to_js_formula(s):
    if s is None:
        return ''
    t = str(s)
    # normalize
    t = t.replace('**', '^')
    t = t.replace('Abs(', 'abs(')
    t = t.replace('log(', 'ln(')  # sympy log is natural; prefer ln
    return t



