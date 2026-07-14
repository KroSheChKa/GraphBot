"""
Local server for the universal-approximator p5 UI + Graphwar field capture.

Usage:
    python tools/approximator_server.py

Then open http://127.0.0.1:8765/
"""

import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from core.game_capture import capture_game_field

APPROX_DIR = ROOT_DIR / "Visuals in p5.js" / "universal-approximator"
HOST = "127.0.0.1"
PORT = 8765


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
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def do_GET(self):
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
        if self.path != "/api/capture":
            self.send_error(404, "Not found")
            return

        try:
            result = capture_game_field()
        except Exception as exc:
            self._send_json(500, {"ok": False, "error": str(exc)})
            return

        status = 200 if result.get("ok") else 400
        self._send_json(status, result)


def main():
    if not APPROX_DIR.is_dir():
        raise SystemExit(f"Approximator folder not found: {APPROX_DIR}")

    httpd = ThreadingHTTPServer((HOST, PORT), ApproximatorHandler)
    print(f"Approximator server: http://{HOST}:{PORT}/")
    print("Graphwar must be running. Use «Захват поля» in the UI.")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
