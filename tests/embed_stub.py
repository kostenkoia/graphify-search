"""In-process stand-in for an OpenAI-compatible /v1 endpoint: embeddings and redirects."""
import json
import math
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, HTTPServer


def vector(text: str, dims: int = 3) -> list[float]:
    raw = [float(len(text) % 7), float(sum(ord(c) for c in text) % 11), 1.0]
    norm = math.sqrt(sum(x * x for x in raw))
    unit = [x / norm for x in raw]
    return unit + [0.0] * (dims - len(unit))


@contextmanager
def _running(handler, calls):
    server = HTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1", calls
    finally:
        server.shutdown()
        server.server_close()


@contextmanager
def serve_redirect(location: str, status: int = 302):
    calls: list[dict] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length") or 0))
            calls.append({"path": self.path, "authorization": self.headers.get("Authorization")})
            self.send_response(status)
            self.send_header("Location", location)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def log_message(self, *args):
            pass

    with _running(Handler, calls) as served:
        yield served


@contextmanager
def serve(reverse: bool = False, drop_last: bool = False, dims: int = 3, non_finite: bool = False,
          row: list[float] | None = None):
    calls: list[dict] = []

    class Handler(BaseHTTPRequestHandler):
        def _json(self, payload):
            body = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            calls.append({**body, "path": self.path, "authorization": self.headers.get("Authorization")})
            if body.get("model") == "broken":
                self.send_response(500)
                self.end_headers()
                return
            data = [{"object": "embedding", "index": i,
                     "embedding": ([float("nan")] * dims if non_finite
                                   else list(row) if row is not None else vector(t, dims))}
                    for i, t in enumerate(body["input"])]
            if drop_last:
                data = data[:-1]
            if reverse:
                data = list(reversed(data))
            self._json({"object": "list", "data": data, "model": body.get("model")})

        def log_message(self, *args):
            pass

    with _running(Handler, calls) as served:
        yield served
