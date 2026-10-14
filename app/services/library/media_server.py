"""Loopback-only streaming for large completed videos without loading them into RAM."""

import mimetypes
import re
import secrets
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse
from uuid import UUID

from app.services.narration_task import task_directory


_server = None
_token = secrets.token_urlsafe(32)
_lock = threading.Lock()


class MediaHandler(BaseHTTPRequestHandler):
    def log_message(self, *_args):
        pass

    def do_HEAD(self):
        self._serve(False)

    def do_GET(self):
        self._serve(True)

    def _serve(self, body):
        parsed = urlparse(self.path)
        parts = parsed.path.strip("/").split("/")
        if len(parts) != 3 or parts[0] != _token or parts[2] != "final.mp4":
            self.send_error(404)
            return
        try:
            path = task_directory(str(UUID(parts[1]))) / "final.mp4"
            size = path.stat().st_size
            if size <= 0:
                raise ValueError("Empty video")
        except (ValueError, OSError):
            self.send_error(404)
            return
        start, end = 0, size - 1
        byte_range = self.headers.get("Range")
        if byte_range:
            match = re.fullmatch(r"bytes=(\d*)-(\d*)", byte_range)
            if not match or not any(match.groups()):
                self.send_error(416)
                return
            if match.group(1):
                start = int(match.group(1))
                end = min(end, int(match.group(2))) if match.group(2) else end
            else:
                start = max(0, size - int(match.group(2)))
            if start > end or start >= size:
                self.send_error(416)
                return
        self.send_response(206 if byte_range else 200)
        self.send_header("Content-Type", "video/mp4")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        if byte_range:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        if parsed.query == "download=1":
            self.send_header("Content-Disposition", 'attachment; filename="final.mp4"')
        self.end_headers()
        if body:
            try:
                with path.open("rb") as source:
                    source.seek(start)
                    remaining = end - start + 1
                    while remaining:
                        chunk = source.read(min(1024 * 1024, remaining))
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        remaining -= len(chunk)
            except (BrokenPipeError, ConnectionResetError):
                pass


def video_url(task_id):
    global _server
    task_id = str(UUID(task_id))
    with _lock:
        if _server is None:
            _server = ThreadingHTTPServer(("127.0.0.1", 0), MediaHandler)
            threading.Thread(target=_server.serve_forever, daemon=True, name="Narrato-media").start()
    return f"http://127.0.0.1:{_server.server_port}/{_token}/{task_id}/final.mp4"
