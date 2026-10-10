#!/usr/bin/env python3
"""A stand-in for the Hugging Face router (OpenAI-style images endpoint), for testing
StoryForge's cloud backgrounds without the internet or a real key.

    POST /{provider}/v1/images/generations   answers with a small PNG as b64_json
    POST /v1/chat/completions                 a vision model describing the hero drawing
Behaviour can be switched while running:  POST /control {"mode": "..."}  with one of:
  normal        a picture
  slow          wait 7 seconds first (longer than the test's 5 second limit)
  error500      always answer HTTP 500
and for the vision models  POST /control {"vision": "..."}:
  normal        every model answers
  first_fails   the first model is "not supported" (HTTP 400), the others answer
  down          every model answers HTTP 500
Only the key "test-token" is accepted; anything else gets HTTP 401.
"""
import base64
import json
import os
import struct
import sys
import time
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

STATE = {"mode": "normal", "requests": 0, "last_prompt": "", "last_path": "", "times": [],
         "vision": "normal", "vision_models": [], "vision_had_image": False}
DESCRIPTION = "A round orange cat with a tall green hat and a long striped tail."


def tiny_png(size: int = 48) -> bytes:
    """A noisy RGB PNG made with the standard library (noise keeps it over 1000 bytes)."""
    rows = b"".join(b"\x00" + os.urandom(size * 3) for _ in range(size))

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

    header = struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b"")


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def _json(self, status: int, payload: dict) -> None:
        data = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/state":
            return self._json(200, STATE)
        self._json(404, {})

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        if self.path == "/control":
            STATE.update(body)
            return self._json(200, STATE)
        if self.path == "/v1/chat/completions":
            return self.vision(body)
        if not self.path.endswith("/v1/images/generations"):
            return self._json(404, {"error": "not found"})
        STATE["requests"] += 1
        STATE["last_prompt"] = body.get("prompt", "")
        STATE["last_size"] = body.get("size", "")
        STATE["last_path"] = self.path
        STATE["times"].append(time.time())
        if self.headers.get("Authorization") != "Bearer test-token":
            return self._json(401, {"error": "Invalid credentials in Authorization header"})
        if STATE["mode"] == "error500":
            return self._json(500, {"error": "boom"})
        if STATE["mode"] == "slow":
            time.sleep(7)
        picture = base64.b64encode(tiny_png()).decode()
        try:
            self._json(200, {"created": int(time.time()), "data": [{"b64_json": picture}]})
        except (BrokenPipeError, ConnectionResetError):
            pass


    def vision(self, body: dict):
        model = body.get("model", "")
        STATE["vision_models"].append(model)
        STATE["vision_prompt"] = next((p["text"] for p in body["messages"][0]["content"] if p.get("type") == "text"), "")
        parts = body["messages"][0]["content"]
        STATE["vision_had_image"] = any(
            p.get("type") == "image_url" and p["image_url"]["url"].startswith("data:image/jpeg;base64,/9j/")
            for p in parts
        )
        if self.headers.get("Authorization") != "Bearer test-token":
            return self._json(401, {"error": "Invalid credentials"})
        if STATE["vision"] == "down" or (STATE["vision"] == "first_fails" and len(STATE["vision_models"]) % 2 == 1):
            return self._json(400 if STATE["vision"] == "first_fails" else 500, {"error": {"message": "nope"}})
        self._json(200, {"choices": [{"message": {"role": "assistant", "content": DESCRIPTION}}]})


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 18091
    try:
        ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
    except KeyboardInterrupt:
        pass
