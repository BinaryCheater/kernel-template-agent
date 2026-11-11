"""Serves Scope. Standard library only, so it runs on any machine with Python.

    python3 serve.py                                              # this machine, for editing
    python3 serve.py --host 0.0.0.0 --port P --password-file F    # reachable: user `scope`

A file's `.gz` beside it is sent as is to browsers that accept gzip: a trace falls from 22 MB to
3 MB, and the server compresses nothing at run time. Only the page's own kinds of file are served,
there are no directory listings, and a wrong password waits a second.
"""

import argparse
import base64
import hmac
import os
import time
from email.utils import formatdate, parsedate_to_datetime
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

SERVED = (".html", ".js", ".css", ".json")


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, secret: str | None = None, **kw) -> None:
        self.secret = secret
        super().__init__(*a, **kw)

    def log_message(self, *_: object) -> None:
        pass

    def allowed(self) -> bool:
        if self.secret is None:
            return True
        got = self.headers.get("Authorization", "")
        want = "Basic " + base64.b64encode(f"scope:{self.secret}".encode()).decode()
        if hmac.compare_digest(got.encode(), want.encode()):
            return True
        time.sleep(1)
        self.send_response(401)
        self.send_header("WWW-Authenticate", 'Basic realm="Scope"')
        self.send_header("Content-Length", "0")
        self.end_headers()
        return False

    def do_GET(self) -> None:
        if self.allowed():
            super().do_GET()

    def do_HEAD(self) -> None:
        if self.allowed():
            super().do_HEAD()

    def send_head(self):
        path = self.translate_path(self.path)
        if os.path.isdir(path):
            path = os.path.join(path, "index.html")
        if not path.endswith(SERVED) or not os.path.isfile(path):
            self.send_error(404)
            return None
        gz = path + ".gz"
        packed = os.path.isfile(gz) and "gzip" in self.headers.get("Accept-Encoding", "")
        real = gz if packed else path
        st = os.stat(real)
        since = self.headers.get("If-Modified-Since")
        if since:
            try:
                if parsedate_to_datetime(since).timestamp() >= int(st.st_mtime):
                    self.send_response(304)
                    self.end_headers()
                    return None
            except (TypeError, ValueError):
                pass
        f = open(real, "rb")
        self.send_response(200)
        self.send_header("Content-Type", self.guess_type(path))
        self.send_header("Content-Length", str(st.st_size))
        self.send_header("Last-Modified", formatdate(st.st_mtime, usegmt=True))
        self.send_header("Cache-Control", "no-cache")  # revalidate: a new export shows on reload
        self.send_header("Vary", "Accept-Encoding")
        if packed:
            self.send_header("Content-Encoding", "gzip")
        self.end_headers()
        return f


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8792)
    ap.add_argument("--password-file", type=Path)
    a = ap.parse_args()
    if a.host != "127.0.0.1" and a.password_file is None:
        ap.error("a reachable server needs --password-file")
    secret = a.password_file.read_text().strip() if a.password_file else None
    here = Path(__file__).parent
    handler = partial(Handler, directory=str(here), secret=secret)
    ThreadingHTTPServer((a.host, a.port), handler).serve_forever()


if __name__ == "__main__":
    main()
