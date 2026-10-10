#!/usr/bin/env python3
"""A stand-in for llama-server, for testing StoryForge without a GPU.

It streams a story page token by token like the real server. Behaviour can be switched
while running:  POST /control {"mode": "..."}  with one of:
  normal        well-formed JSON
  fenced        JSON wrapped in a ```json code fence with chatter around it
  truncated     JSON cut off in the middle of the choices (as if the token limit was hit)
  garbage       plain prose, no JSON at all
  no_schema     reject response_format json_schema with HTTP 400 (like an old build)
  error500      always answer HTTP 500
"""
import json
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

STATE = {"mode": "normal", "delay": 0.01, "requests": 0, "last_user": "", "last_format": None}

PAGES = [
    "Mira was a small purple dragon, no bigger than a watering can, and she lived at the edge of the Whispering Wood. "
    "Every morning she counted the mushrooms beside her burrow, and every morning there were exactly nine.\n\n"
    "But today there were ten. The tenth mushroom glowed a soft gold, and behind it, tucked between the roots of the "
    "oldest oak, stood a tiny wooden door that had certainly not been there yesterday. A warm light flickered under it, "
    "and something on the other side was humming a tune.",
    "Mira took a deep breath and did exactly what she had decided. The air smelled of cinnamon and rain as she crept closer, "
    "her tail curled tight with excitement.\n\n"
    "On the other side she found a round room lined with shelves, and on every shelf sat a jar holding a tiny, sleeping star. "
    "One jar was open and empty. A small voice near the ceiling whispered, \"Oh dear. You have not seen a runaway star, have you?\" "
    "Mira looked up, and a pair of silver eyes blinked back at her.",
    "The silver eyes belonged to a moth named Pim, keeper of the star jars. Pim explained that the runaway star was the youngest, "
    "and that it loved to hide in places that sparkled.\n\n"
    "Together they followed a trail of glittering dust across the mossy floor, out through a crack in the wall, and down to the "
    "edge of a moonlit pond. There, in the middle of the water, something was shining beneath a lily pad, and it was giggling.",
]
ENDING = (
    "And so, with the littlest star safely back in its jar, Mira said goodnight to Pim and stepped out through the tiny wooden door. "
    "The Whispering Wood was quiet and silver with moonlight.\n\n"
    "Beside her burrow, ten mushrooms waited, and the tenth still glowed a soft gold. Mira curled up next to it, smiled, and "
    "fell asleep knowing that tomorrow the door would still be there, and so would her new friend."
)


def build_reply(user_text: str) -> dict:
    final = "LAST page" in user_text
    first = "Write page 1 only" in user_text
    number = 1
    import re
    m = re.search(r"Write page (\d+)", user_text)
    if m:
        number = int(m.group(1))
    reply: dict = {}
    if first:
        reply["title"] = "Mira and the Tenth Mushroom"
    # The page plan comes first: setting, visual_details, action.
    if number == 1:
        reply["setting"] = "a cosy village bakery kitchen at dawn"
        reply["visual_details"] = ["a warm brick oven", "flour on the wooden table", "morning sun through the window"]
        reply["action"] = "waving"
    else:
        reply["setting"] = f"a sunny sandy beach beside a red lighthouse, scene {number}"
        reply["visual_details"] = ["seashells in the sand", "a striped beach umbrella"]
        reply["action"] = "walking"
    if first and '"hero_name"' in user_text:
        reply["hero_name"] = "Captain Whiskers"
    reply["segment"] = ENDING if final else PAGES[(number - 1) % len(PAGES)]
    if first:
        reply["character_look"] = "a small purple baby dragon with large amber eyes and tiny wings"
        reply["world"] = "an enchanted forest of ancient oaks and glowing mushrooms"
    reply["visual_prompt"] = (
        "a small purple baby dragon beside a tiny glowing wooden door between oak roots in an enchanted forest"
        if number == 1 else f"a small purple dragon and a silver moth in a magical scene, page {number}"
    )
    reply["choices"] = [] if final else [
        "Open the tiny wooden door", "Knock and wait politely", "Follow the humming around the tree",
    ]
    reply["continuity_summary"] = (
        f"Mira, a small purple dragon, found a tiny door by a golden mushroom in the Whispering Wood. Page {number} done."
    )
    return reply


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
        if self.path == "/health":
            self._json(200, {"status": "ok"})
        elif self.path == "/state":
            self._json(200, STATE)
        else:
            self._json(404, {})

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        if self.path == "/control":
            STATE.update(body)
            return self._json(200, STATE)
        if self.path != "/v1/chat/completions":
            return self._json(404, {})

        STATE["requests"] += 1
        STATE["last_user"] = body["messages"][-1]["content"]
        fmt = (body.get("response_format") or {}).get("type")
        STATE["last_format"] = fmt
        mode = STATE["mode"]
        if mode == "error500":
            return self._json(500, {"error": {"message": "boom"}})
        if mode == "no_schema" and fmt == "json_schema":
            return self._json(400, {"error": {"message": "unsupported response_format"}})

        text = json.dumps(build_reply(STATE["last_user"]), ensure_ascii=False)
        if mode == "fenced":
            text = "Sure! Here is the page:\n```json\n" + text + "\n```\nHope you like it."
        elif mode == "truncated":
            text = text[: text.index('"choices"') + 24]
        elif mode == "garbage":
            text = "I am sorry, but I would rather tell you about the weather today."

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Connection", "close")
        self.end_headers()
        pieces = [text[i : i + 6] for i in range(0, len(text), 6)]
        try:
            for piece in pieces:
                chunk = {"choices": [{"delta": {"content": piece}}]}
                self.wfile.write(b"data: " + json.dumps(chunk).encode() + b"\n\n")
                self.wfile.flush()
                time.sleep(float(STATE["delay"]))
            last = {"choices": [{"delta": {}, "finish_reason": "stop"}],
                    "timings": {"predicted_n": len(pieces), "predicted_per_second": 17.5}}
            self.wfile.write(b"data: " + json.dumps(last).encode() + b"\n\ndata: [DONE]\n\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            STATE["cancelled"] = STATE.get("cancelled", 0) + 1
        self.close_connection = True


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
    try:
        ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
    except KeyboardInterrupt:
        pass
