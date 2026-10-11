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
POST /control {"ignore_choice": n} makes the next n pages leave out the reader's choice (the
"event" the schema fills in); otherwise a page starts by acting it out, as a good model would.
POST /control {"short_pages": n, "short_words": k} makes the next n pages only k words long.
POST /control {"cut_off_pages": n} cuts the next n replies off a few words into the page text,
with finish_reason "length", as if the token limit was hit.
Tap-a-word definition requests (system prompt about learning new words) are answered separately and never
change the page counters above: a definition for each word in the schema (the words StoryForge
chose), plus a word it did not ask for, which StoryForge must leave out.
POST /control {"define_delay": seconds per piece, "define_mode": "normal" | "garbage"}.
Story outline requests (system prompt about planning adventures) are answered with OUTLINE and
counted separately too; POST /control {"outline_mode": "garbage"} makes them unusable.
POST /control {"rough_pages": n} adds a not-gentle sentence to the next n pages.
"""
import json
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

STATE = {"mode": "normal", "delay": 0.01, "requests": 0, "last_user": "", "last_format": None,
         "ignore_choice": 0, "last_schema": None, "short_pages": 0, "short_words": 8, "cut_off_pages": 0,
         "last_system": "", "define_requests": 0, "define_cancelled": 0, "define_delay": 0.002,
         "define_mode": "normal", "last_define_user": "",
         "outline_requests": 0, "outline_mode": "normal", "last_outline_user": "", "rough_pages": 0}
OUTLINE = {
    "hero_name": "Mira", "hero_kind": "baby dragon",
    "goal": "find the lost moon pearl and bring it home",
    "obstacle": "a sleepy river has washed over the stepping stones",
    "friends": [{"name": "Pim", "kind": "silver moth"}, {"name": "Olo", "kind": "old owl"}],
    "goal_done": "Mira finds the moon pearl hidden in a shell and names it Shimmer",
    "ending": "Mira returns the moon pearl and the whole wood glows with silver light",
}


def definitions_reply(schema: dict, user_text: str) -> dict:
    """A definition for each word StoryForge chose (the schema's keys, or the prompt's list when an
    older build gets no schema), with the fields the level asks for, plus a word it did not ask for,
    which StoryForge must leave out."""
    import re
    entries = ((schema.get("properties") or {}).get("words") or {}).get("properties") or {}
    if not entries:
        listed = re.search(r"Define these words from the page: (.*?)\. Say", user_text)
        fields = {"definition": {}}
        if "one emoji" in user_text:
            fields["emoji"] = {}
        if "tiny example" in user_text:
            fields["example"] = {}
        entries = {w: {"properties": fields} for w in re.findall(r'"([^"]+)"', listed.group(1) if listed else "")}
    words = {}
    for word, item in entries.items():
        fields = item.get("properties") or {}
        words[word] = {"definition": f"a thing in the story called {word}. More words here."}
        if "emoji" in fields:
            words[word]["emoji"] = "🌟"
        if "example" in fields:
            words[word]["example"] = f"We saw the {word} today."
    words["zeppelin"] = {"definition": "a big airship"}  # not asked for
    return {"words": words}

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


def extra_page(number: int) -> str:
    """A page after the written ones. Each is different (and repeatable), since StoryForge drops
    sentences an earlier page already had."""
    import random
    rng = random.Random(number)
    who = ["Mira", "Pim", "The old owl", "A sleepy frog", "The tiny star", "A kind beetle"]
    did = ["tiptoed past", "painted", "counted", "hummed beside", "polished", "climbed over", "peeked under", "carried"]
    what = ["a mossy bridge", "seven blue pebbles", "a hollow log", "the singing reeds", "a basket of berries",
            "a crooked signpost", "the silver puddle", "a pile of maple leaves", "a lantern made of glass", "a velvet curtain"]
    how = ["very slowly", "with a giggle", "before the moon rose", "while the crickets sang", "on wobbly toes",
           "as the breeze whistled", "without a single sound", "for the very first time"]
    sentences = [f"{rng.choice(who)} {rng.choice(did)} {rng.choice(what)} {rng.choice(how)}, and {rng.choice(who).lower()} "
                 f"{rng.choice(did)} {rng.choice(what)} {rng.choice(how)}." for _ in range(5)]
    return " ".join(sentences[:3]) + "\n\n" + " ".join(sentences[3:])


def build_reply(user_text: str, event: str | None = None) -> dict:
    final = "LAST page" in user_text
    first = "Write page 1 only" in user_text
    number = 1
    import re
    m = re.search(r"Write page (\d+)", user_text)
    if m:
        number = int(m.group(1))
    reply: dict = {}
    if event:
        reply["event"] = event  # the grammar writes the const first, before anything the model chooses
    if STATE.get("last_beat"):
        reply["beat"] = STATE["last_beat"]
    if first:
        reply["title"] = "Mira and the Tenth Mushroom"
    # The page plan comes first: setting, visual_details, action.
    if number == 1:
        reply["setting"] = "a cosy village bakery kitchen at dawn"
        reply["visual_details"] = ["a warm brick oven", "flour on the wooden table", "morning sun through the window"]
        reply["props"] = ["a ripe strawberry", "glimmering zorb"]  # the second has no emoji
        reply["action"] = "waving"
    else:
        reply["setting"] = f"a sunny sandy beach beside a red lighthouse, scene {number}"
        reply["visual_details"] = ["seashells in the sand", "a striped beach umbrella"]
        reply["props"] = ["a seashell"]
        reply["action"] = "walking"
    if first and '"hero_name"' in user_text:
        reply["hero_name"] = "Captain Whiskers"
    reply["segment"] = ENDING if final else PAGES[number - 1] if number <= len(PAGES) else extra_page(number)
    if event:
        if STATE["ignore_choice"] > 0:
            STATE["ignore_choice"] -= 1
        else:
            reply["segment"] = f"Mira chose to {event[0].lower() + event[1:]}. " + reply["segment"]
    if STATE["rough_pages"] > 0:
        STATE["rough_pages"] -= 1
        reply["segment"] = "Then the two dragons began to fight in the meadow. " + reply["segment"]
    if STATE["short_pages"] > 0:
        STATE["short_pages"] -= 1
        reply["segment"] = " ".join(reply["segment"].split()[: STATE["short_words"]])
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

        # Recognised by the prompt, not the schema: after "no_schema" StoryForge sends json_object.
        if "learn new words" in body["messages"][0]["content"]:
            return self.definitions(body)
        if "plan gentle picture-book adventures" in body["messages"][0]["content"]:
            STATE["outline_requests"] += 1
            STATE["last_outline_user"] = body["messages"][-1]["content"]
            text = "not an outline" if STATE["outline_mode"] == "garbage" else json.dumps(OUTLINE)
            return self.stream_text(text)
        STATE["requests"] += 1
        STATE["last_user"] = body["messages"][-1]["content"]
        STATE["last_system"] = body["messages"][0]["content"]
        fmt = (body.get("response_format") or {}).get("type")
        schema = ((body.get("response_format") or {}).get("json_schema") or {}).get("schema") or {}
        STATE["last_schema"] = schema
        event = (schema.get("properties") or {}).get("event", {}).get("const")
        STATE["last_beat"] = (schema.get("properties") or {}).get("beat", {}).get("const")
        if not event:  # no grammar (an old build): a model would still read the prompt's last line
            import re
            found = re.search(r"This page must show: (.+)$", body["messages"][-1]["content"])
            event = found.group(1) if found else None
        STATE["last_format"] = fmt
        mode = STATE["mode"]
        if mode == "error500":
            return self._json(500, {"error": {"message": "boom"}})
        if mode == "no_schema" and fmt == "json_schema":
            return self._json(400, {"error": {"message": "unsupported response_format"}})

        text = json.dumps(build_reply(STATE["last_user"], event), ensure_ascii=False)
        if mode == "fenced":
            text = "Sure! Here is the page:\n```json\n" + text + "\n```\nHope you like it."
        elif mode == "truncated":
            text = text[: text.index('"choices"') + 24]
        elif mode == "garbage":
            text = "I am sorry, but I would rather tell you about the weather today."
        finish = "stop"
        if STATE["cut_off_pages"] > 0:
            STATE["cut_off_pages"] -= 1
            text = text[: text.index('"segment"') + 60]
            finish = "length"

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
            last = {"choices": [{"delta": {}, "finish_reason": finish}],
                    "timings": {"predicted_n": len(pieces), "predicted_per_second": 17.5}}
            self.wfile.write(b"data: " + json.dumps(last).encode() + b"\n\ndata: [DONE]\n\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            STATE["cancelled"] = STATE.get("cancelled", 0) + 1
        self.close_connection = True


def _definitions(handler, body):
    STATE["define_requests"] += 1
    STATE["last_define_user"] = body["messages"][-1]["content"]
    if STATE["define_mode"] == "garbage":
        text = "Sorry, I can't help with words today."
    else:
        schema = ((body.get("response_format") or {}).get("json_schema") or {}).get("schema") or {}
        text = json.dumps(definitions_reply(schema, STATE["last_define_user"]))
    handler.send_response(200)
    handler.send_header("Content-Type", "text/event-stream")
    handler.send_header("Connection", "close")
    handler.end_headers()
    try:
        for i in range(0, len(text), 6):
            chunk = {"choices": [{"delta": {"content": text[i : i + 6]}}]}
            handler.wfile.write(b"data: " + json.dumps(chunk).encode() + b"\n\n")
            handler.wfile.flush()
            time.sleep(float(STATE["define_delay"]))
        handler.wfile.write(b'data: {"choices": [{"delta": {}, "finish_reason": "stop"}]}\n\ndata: [DONE]\n\n')
        handler.wfile.flush()
    except (BrokenPipeError, ConnectionResetError):
        STATE["define_cancelled"] += 1
    handler.close_connection = True


def _stream_text(handler, text):
    handler.send_response(200)
    handler.send_header("Content-Type", "text/event-stream")
    handler.send_header("Connection", "close")
    handler.end_headers()
    try:
        for i in range(0, len(text), 8):
            chunk = {"choices": [{"delta": {"content": text[i : i + 8]}}]}
            handler.wfile.write(b"data: " + json.dumps(chunk).encode() + b"\n\n")
            handler.wfile.flush()
            time.sleep(0.002)
        handler.wfile.write(b'data: {"choices": [{"delta": {}, "finish_reason": "stop"}]}\n\ndata: [DONE]\n\n')
        handler.wfile.flush()
    except (BrokenPipeError, ConnectionResetError):
        pass
    handler.close_connection = True


Handler.definitions = _definitions
Handler.stream_text = _stream_text


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
    try:
        ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
    except KeyboardInterrupt:
        pass
