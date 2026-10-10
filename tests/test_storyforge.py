#!/usr/bin/env python3
"""End-to-end tests for StoryForge using stand-ins for Qwen and sd-cli.

Run with:   python3 tests/test_storyforge.py
Nothing here touches the real models, the GPU, or your real stories and pictures:
everything is written to a temporary folder that is deleted afterwards.
(The fake picture maker needs Pillow:  pip install pillow)
"""
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
APP_PORT, LLM_PORT = 18000, 18080
CLOUD_PORT, HF_PORT = 18001, 18091  # a second StoryForge in Hugging Face mode, and a fake router
BASE = f"http://127.0.0.1:{APP_PORT}"
LLM = f"http://127.0.0.1:{LLM_PORT}"
CLOUD = f"http://127.0.0.1:{CLOUD_PORT}"
HF = f"http://127.0.0.1:{HF_PORT}"
TMP = Path(tempfile.mkdtemp(prefix="storyforge_test_"))
FAIL_FILE = TMP / "sd_should_fail"
SD_LOG = TMP / "sd_calls.log"
FAST_DIR = TMP / "fast"
FAST_FAIL = TMP / "fast_should_fail"
procs: list[subprocess.Popen] = []


def call(method: str, url: str, payload=None, raw: bytes | None = None, timeout=30):
    data = raw if raw is not None else (json.dumps(payload).encode() if payload is not None else None)
    request = urllib.request.Request(url, data=data, method=method)
    if payload is not None:
        request.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read()
            kind = response.headers.get("Content-Type", "")
            return response.status, (json.loads(body) if "json" in kind else body)
    except urllib.error.HTTPError as exc:
        body = exc.read()
        try:
            return exc.code, json.loads(body)
        except ValueError:
            return exc.code, body


def llm_mode(mode: str, delay: float = 0.004) -> None:
    call("POST", f"{LLM}/control", {"mode": mode, "delay": delay})


def wait_for(check, timeout=40, every=0.15):
    deadline = time.time() + timeout
    while time.time() < deadline:
        value = check()
        if value:
            return value
        time.sleep(every)
    raise AssertionError("timed out waiting")


def segment(story_id: str, index: int, base=BASE) -> dict:
    status, body = call("GET", f"{base}/api/stories/{story_id}/segments/{index}")
    assert status == 200, (status, body)
    return body


def wait_ready(story_id: str, index: int, base=BASE) -> dict:
    return wait_for(lambda: (s := segment(story_id, index, base))["status"] in ("ready", "failed") and s)


def wait_image(story_id: str, index: int, base=BASE) -> dict:
    return wait_for(
        lambda: (s := segment(story_id, index, base))["image"]["status"] in ("completed", "failed", "skipped") and s
    )


def new_story(premise="A small purple dragon discovers a mysterious door in an enchanted forest", base=BASE) -> str:
    status, body = call("POST", f"{base}/api/stories", {"premise": premise})
    assert status == 201, (status, body)
    return body["id"]


def setUpModule():
    env = dict(
        os.environ,
        # Neither the real settings.env nor the real .env (with the real key) is ever read here.
        STORYFORGE_SETTINGS=str(TMP / "none.env"), STORYFORGE_ENV_FILE=str(TMP / "none.env"), HF_TOKEN="",
        IMAGE_MODE="local",
        LLM_URL=LLM, PORT=str(APP_PORT), HTTPS_PORT="0",
        SD_CLI=str(ROOT / "tests" / "fake_sd_cli"), SD_MODEL=str(ROOT / "tests" / "fake_sd_cli"),
        IMAGE_DIR=str(TMP / "images"), DATA_DIR=str(TMP / "data"), LOG_DIR=str(TMP / "logs"),
        CERT_DIR=str(TMP / "certs"), SD_STEPS="8", SD_MIN_FREE_MB="0", TTS_MODE="browser",
        FAKE_SD_STEP_SECONDS="0.05", FAKE_SD_FAIL_FILE=str(FAIL_FILE), FAKE_SD_LOG=str(SD_LOG),
        LLM_TIMEOUT_SECONDS="20",
        SD_LCM_LORA=str(FAST_DIR / "lcm-lora-sdv1-5.safetensors"), SD_TAESD=str(FAST_DIR / "taesd.safetensors"),
        FAKE_SD_FAIL_FAST_FILE=str(FAST_FAIL),
    )
    # The cloud instance gets its key from a .env-style file, the way the real app does.
    (TMP / "cloud.env").write_text("# test secrets\nHF_TOKEN=test-token\n")
    cloud_env = dict(
        env, IMAGE_MODE="huggingface", STORYFORGE_ENV_FILE=str(TMP / "cloud.env"),
        HF_URL=HF, HF_TIMEOUT_SECONDS="5", HF_FALLBACK_LOCAL="1", PORT=str(CLOUD_PORT),
        IMAGE_DIR=str(TMP / "cloud_images"), DATA_DIR=str(TMP / "cloud_data"), LOG_DIR=str(TMP / "cloud_logs"),
    )
    del cloud_env["HF_TOKEN"]
    procs.append(subprocess.Popen([sys.executable, str(ROOT / "tests" / "fake_llama.py"), str(LLM_PORT)]))
    procs.append(subprocess.Popen([sys.executable, str(ROOT / "tests" / "fake_hf.py"), str(HF_PORT)]))
    for port, app_env in ((APP_PORT, env), (CLOUD_PORT, cloud_env)):
        procs.append(subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "app:app", "--host", "127.0.0.1", "--port", str(port),
             "--log-level", "warning"],
            cwd=ROOT, env=app_env, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT,
        ))
    wait_for(lambda: call("GET", f"{LLM}/health")[0] == 200 if _up(LLM_PORT) else False)
    wait_for(lambda: _up(HF_PORT))
    wait_for(lambda: call("GET", f"{BASE}/api/config")[0] == 200 if _up(APP_PORT) else False)
    wait_for(lambda: call("GET", f"{CLOUD}/api/config")[0] == 200 if _up(CLOUD_PORT) else False)


def _up(port: int) -> bool:
    import socket
    with socket.socket() as sock:
        sock.settimeout(0.3)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def tearDownModule():
    for proc in procs:
        proc.send_signal(signal.SIGINT)
    for proc in procs:
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
    shutil.rmtree(TMP, ignore_errors=True)


class Parsing(unittest.TestCase):
    def test_partial_extraction_while_streaming(self):
        from storyforge.llm import extract_string_field
        raw = '{"title": "T", "segment": "Once upon a \\"time\\" there\\nwas'
        text, complete = extract_string_field(raw, "segment")
        self.assertEqual(text, 'Once upon a "time" there\nwas')
        self.assertFalse(complete)
        text, complete = extract_string_field(raw + ' a dragon.", "visual_prompt": "x"}', "segment")
        self.assertTrue(complete)
        self.assertTrue(text.endswith("a dragon."))
        self.assertEqual(extract_string_field('{"title": "T"', "segment"), (None, False))
        # an escape sequence that has only half arrived must not break anything
        self.assertEqual(extract_string_field('{"segment": "caf\\u00e', "segment")[0], "caf")

    def test_validation_rejects_bad_pages(self):
        from storyforge.llm import WriterFailed, parse_page
        with self.assertRaises(WriterFailed):
            parse_page("no json here", True, False)
        with self.assertRaises(WriterFailed):
            parse_page('{"segment": "Too short."}', True, False)
        ten = json.dumps({"segment": " ".join(["word"] * 10) + "."})
        with self.assertRaises(WriterFailed):
            parse_page(ten, True, False)  # under 15 words: not a usable page
        for words in (15, 37):  # short, but a fine children's page: kept
            reply = json.dumps({"segment": " ".join(["word"] * words) + "."})
            self.assertEqual(len(parse_page(reply, True, False).segment.split()), words)
        from storyforge.vision import clean_description
        self.assertEqual(clean_description("A smiling man wearing a hat, drawn in dark blue, with a big nose."),
                         "a smiling man wearing a hat, with a big nose")

    def test_validation_repairs_choices(self):
        from storyforge.llm import parse_page
        words = " ".join(["word"] * 60) + "."
        page = parse_page(json.dumps({"segment": words, "choices": ["1. Open the door.", "open the door", {"text": "Run"}, 7]}), False, False)
        self.assertEqual(page.choices, ["Open the door", "Run"])
        page = parse_page(json.dumps({"segment": words, "choices": "none"}), False, False)
        self.assertEqual(len(page.choices), 3)
        self.assertTrue(page.used_fallback_choices)
        page = parse_page(json.dumps({"segment": words, "choices": ["a b c"]}), False, True)
        self.assertEqual(page.choices, [])

    def test_page_plan_comes_first_and_is_parsed(self):
        from storyforge.llm import HERO_ACTIONS, SYSTEM_PROMPT, build_messages, parse_page, response_schema
        plan = ["setting", "visual_details", "props", "action"]
        self.assertEqual(list(response_schema(False, False)["properties"])[:5], plan + ["segment"])
        self.assertEqual(list(response_schema(True, False)["properties"])[:6], ["title"] + plan + ["segment"])
        self.assertEqual(response_schema(False, False)["properties"]["action"]["enum"], list(HERO_ACTIONS))
        story = {"premise": "p", "segments": []}
        text = build_messages(story, None, False)[1]["content"]
        self.assertLess(text.index('"action"'), text.index('"segment"'))
        self.assertIn("Do not default to a forest", text)
        self.assertIn("written from the plan above", text)
        self.assertIn("Keep it gentle: nobody is hurt", SYSTEM_PROMPT)
        self.assertNotIn("sword", SYSTEM_PROMPT)  # naming weapons makes the model add them
        words = " ".join(["word"] * 60) + "."
        page = parse_page(json.dumps({"setting": "a windy hilltop at sunset", "visual_details": ["a red kite", "a red kite", 3],
                                      "action": "Jumping", "segment": words}), False, False)
        self.assertEqual((page.setting, page.visual_details, page.hero_action), ("a windy hilltop at sunset", ["a red kite"], "jumping"))
        page = parse_page(json.dumps({"segment": words, "action": "attacking"}), False, False)
        self.assertEqual(page.hero_action, HERO_ACTIONS[0])  # anything off the list becomes the default
        # broken JSON: the plan is still salvaged
        page = parse_page('{"setting": "a castle hall", "visual_details": ["a long table", "candles"], "action": "dancing", '
                          '"segment": "' + words + '", "choices": ["Open', False, False)
        self.assertEqual((page.setting, page.visual_details, page.hero_action), ("a castle hall", ["a long table", "candles"], "dancing"))

    def test_every_hero_action_is_animated(self):
        import re as regex
        from storyforge.llm import HERO_ACTIONS
        hero_js = (ROOT / "static" / "hero.js").read_text()
        css = (ROOT / "static" / "style.css").read_text()
        listed = regex.search(r"const ACTIONS = \[(.*?)\];", hero_js, regex.S).group(1)
        self.assertEqual(tuple(regex.findall(r'"(\w+)"', listed)), HERO_ACTIONS)  # same list, same order
        for action in HERO_ACTIONS[1:]:  # "standing" is breathing only
            self.assertIn(f'[data-action="{action}"]', css, action)
        self.assertIn("prefers-reduced-motion", hero_js)
        reduced = css[css.index("/* Reduce motion: the hero just stands there. */"):]
        self.assertIn(".hero-stage, .hero-stage *, .hero-fx { animation: none !important; }", reduced)
        page = (ROOT / "static" / "index.html").read_text()
        for needle in ('id="line-boil-noise"', "feDisplacementMap", 'src="/static/hero.js"', 'class="hero-shadow"', 'id="hero-fx"'):
            self.assertIn(needle, page)
        # props sit inside the hero's layers, so they move and breathe with it
        breath = page[page.index('<div class="hero-breath">'):]
        self.assertLess(breath.index('class="hero-props"'), breath.index("</div>"))
        self.assertIn("set(next, props = [])", hero_js)
        self.assertIn('heroAnimator.set(page.hero_action || "standing", page.props || [])',
                      (ROOT / "static" / "app.js").read_text())
        self.assertIn(".hero-prop {", css)
        # whenever the hero turns around, its props' emoji turn back the same way at the same time
        import re as regex2
        flips = regex2.findall(r'\[data-action="(\w+)"\] \.hero-flip \{ animation: (face [^;]+); \}', css)
        counters = regex2.findall(r'\[data-action="(\w+)"\] \.hero-prop-face \{ animation: (face [^;]+); \}', css)
        self.assertTrue(flips)
        self.assertEqual(sorted(flips), sorted(counters))
        self.assertIn('face.className = "hero-prop-face"', hero_js)

    def test_no_weapon_words_unless_the_drawing_has_one(self):
        from storyforge import vision
        from storyforge.llm import _WEAPON, build_messages
        def prompts(description: str) -> list[str]:
            story = {"premise": "A story starring a small blue fish", "hero_description": description, "segments": []}
            first = build_messages(story, None, False)
            story.update(title="T", hero_name="Blue", character_look=description, summary="s",
                         segments=[{"status": "ready", "text": "Blue swam.", "setting": "a reef"}])
            later = build_messages(story, "Swim to the cave", False)
            return [m["content"] for m in first + later]
        calm = "a small blue fish with a yellow crown, one black eye, and a red tail"
        for text in prompts(calm) + [vision.PROMPT]:
            self.assertIsNone(_WEAPON.search(text), _WEAPON.search(text) and _WEAPON.search(text).group(0))
        first_user, later_user = prompts(calm)[1], prompts(calm)[3]
        self.assertIn("The hero only holds or uses things that are in these notes.", first_user)
        self.assertIn("The hero only holds or uses things from how it looks", later_user)
        # a weapon in the drawing itself: one line, naming it, saying it is a toy or a tool
        armed = prompts("a green dragon with a horn and a wooden sword with a yellow handle")
        for user in (armed[1], armed[3]):
            self.assertEqual(user.count("toy or a tool"), 1)
            self.assertIn("The sword the hero was drawn with is only a toy or a tool here", user)
        self.assertNotIn("toy or a tool", armed[0])  # the system prompt never mentions it

    def test_choice_helpers(self):
        from storyforge.llm import bridge_sentence, choice_followed, response_schema, short_summary, soften_choice
        for scary, gentle in (("fight the dragon", "make friends with the dragon"),
                              ("Attack the castle with a big sword", "Make friends with the castle"),
                              ("scare the ghost away", "make friends with the ghost"),
                              ("steal the golden key", "borrow the golden key"),
                              ("Open the tiny wooden door", "Open the tiny wooden door"),
                              ("eat the cake", "eat the cake")):
            self.assertEqual(soften_choice(scary), gentle)
        self.assertTrue(choice_followed("Knock and wait politely", "Mira knocked twice and waited."))
        self.assertFalse(choice_followed("Open the tiny wooden door", "Mira flew to the moon."))
        # words the previous page already had prove nothing: "Pim" alone doesn't count
        self.assertFalse(choice_followed("Follow Pim to the pond", "Pim smiled.", known="Pim the moth waved."))
        self.assertTrue(choice_followed("Follow Pim to the pond", "They reached the pond.", known="Pim the moth waved."))
        self.assertEqual(bridge_sentence("Open the tiny wooden door", "Mira"), "So Mira decided to open the tiny wooden door.")
        self.assertEqual(bridge_sentence("the jars wake up and sing", None), "And then, the jars wake up and sing.")
        self.assertEqual(short_summary("One. Two. Three. Four."), "One. Two.")
        props = list(response_schema(False, False, event="Open the door")["properties"])
        self.assertEqual(props[:2], ["event", "setting"])
        self.assertNotIn("event", response_schema(False, False)["properties"])

    def test_hero_notes_and_name_field(self):
        from storyforge.llm import hero_notes, response_schema
        self.assertEqual(hero_notes("a small blue fish with a yellow crown, one black eye, and a red triangular tail"),
                         ("a small blue fish", ["a yellow crown", "one black eye", "a red triangular tail"]))
        self.assertEqual(hero_notes("a brave little hero in bright colours"), ("a brave little hero in bright colours", []))
        # real descriptions: a kind, then 2-3 features, never a lone word
        self.assertEqual(hero_notes("a smiling person wearing a hat and glasses, with a vest, has a rounded face, a small nose"),
                         ("a smiling person", ["wearing a hat and glasses", "a vest", "a rounded face"]))
        self.assertEqual(hero_notes("a round, yellow creature with a cream belly, pink cheeks, and dark eyes"),
                         ("a round, yellow creature", ["a cream belly", "pink cheeks", "dark eyes"]))
        self.assertEqual(hero_notes("a rabbit-like creature with a dark purple body, featuring two long ears, and a nose"),
                         ("a rabbit-like creature", ["a dark purple body", "two long ears", "a nose"]))
        drawn = list(response_schema(True, False, drawn_hero=True)["properties"])
        self.assertLess(drawn.index("hero_name"), drawn.index("segment"))
        self.assertNotIn("hero_name", response_schema(True, False)["properties"])
        self.assertNotIn("hero_name", response_schema(False, False, drawn_hero=True)["properties"])

    def test_plan_is_only_used_once_complete(self):
        from storyforge.llm import extract_plan
        full = '{"setting": "a space station window", "visual_details": ["a blue planet", "tiny stars"], "action": "flying", "segment": "Once'
        self.assertIsNone(extract_plan(full[:40]))
        self.assertIsNone(extract_plan(full[: full.index('"action"')]))
        self.assertIsNone(extract_plan(full[: full.index('flying') + 3]))
        self.assertEqual(extract_plan(full), {"setting": "a space station window",
                                              "visual_details": ["a blue planet", "tiny stars"], "props": [],
                                              "action": "flying"})

    def test_preset_matches_the_place(self):
        from pathlib import Path as P
        from storyforge.images import pick_preset
        presets = [P(f"preset-{n}.jpg") for n in ("beach", "castle", "space", "underwater", "village")]
        for place, want in (("a sunny sandy beach", "beach"), ("the castle throne room", "castle"),
                            ("a rocket near the moon", "space"), ("a coral reef deep under the sea", "underwater"),
                            ("grandma's kitchen in the village", "village")):
            self.assertEqual(pick_preset(presets, place, 0).stem, f"preset-{want}", place)
        self.assertEqual(pick_preset(presets, "a mysterious meadow", 7), presets[7 % 5])  # no match: steady choice

    def test_background_prompt_has_no_characters(self):
        from storyforge.images import NO_CHARACTERS, ImageWorker
        story = {"premise": "p", "world": "a forest", "character_look": "a purple dragon with wings"}
        prompt = ImageWorker.build_background_prompt(
            story, {"setting": "a snowy village square at dusk", "visual_details": ["a glowing lantern", "falling snow"]})
        self.assertIn("a snowy village square at dusk, a glowing lantern, falling snow", prompt)
        self.assertIn(NO_CHARACTERS, prompt)
        self.assertNotIn("dragon", prompt)
        self.assertIn("a forest", ImageWorker.build_background_prompt(story, {}))  # no setting yet
        # characters that slip into the plan are never painted
        prompt = ImageWorker.build_background_prompt(
            story, {"setting": "Robot's Kitchen", "visual_details": ["Robot", "a ticking clock", "a sleepy cat"]})
        self.assertIn("Kitchen, a ticking clock, " + NO_CHARACTERS, prompt)
        self.assertNotIn("Robot", prompt)
        self.assertNotIn("cat", prompt)
        prompt = ImageWorker.build_background_prompt(story, {"setting": "a village square", "visual_details": [
            "Pogo, the purple stick figure", "a kite for Mia", "Round sun in the sky", "shadows of trees"]})
        self.assertIn("a village square, Round sun in the sky, shadows of trees, " + NO_CHARACTERS, prompt)

    def test_background_leaves_out_the_hero_and_what_it_wears(self):
        from storyforge.images import COMPOSITION, ImageWorker, hero_words
        story = {"hero_name": "Cowboy Bill", "hero_who": "my cowboy",
                 "hero_description": "a stick figure with a brown cowboy hat, blue jeans and a lasso",
                 "character_look": "a stick figure with a brown cowboy hat, blue jeans and a lasso"}
        words = hero_words(story)
        self.assertTrue({"cowboy", "bill", "hat", "jean", "lasso"} <= words)
        self.assertFalse({"brown", "blue", "with", "a"} & words)  # colours and filler stay allowed
        prompt = ImageWorker.build_background_prompt(story, {
            "setting": "Cowboy Bill's sunny ranch", "visual_details": [
                "a big cowboy hat", "Hats on a fence", "a red scarf", "a wooden barn", "blue sky with clouds"]})
        self.assertIn(f"{COMPOSITION}, sunny ranch, a wooden barn, blue sky with clouds, ", prompt)
        for word in ("owboy", "Bill", "hat", "Hat", "scarf"):
            self.assertNotIn(word, prompt)
        self.assertLess(prompt.index(COMPOSITION), prompt.index("ranch"))  # part of the style, up front

    def test_sentences_from_earlier_pages_are_dropped(self):
        from storyforge.llm import drop_repeats, parse_page
        earlier = ["Mira ran down the hill. A warm light flickered under the old wooden door.\n\nShe wondered what was inside."]
        page = ("A warm light flickered under the old wooden door! Mira knocked twice and the door creaked open. "
                "Mira smiled.\n\nInside, a warm light was flickering under the old wooden door.\n\nShe wondered what was inside.")
        text, dropped = drop_repeats(page, earlier)
        # nearly the same (one word or the ending changed), or the same: dropped; an empty paragraph goes too
        self.assertEqual(text, "Mira knocked twice and the door creaked open. Mira smiled.")
        self.assertEqual(dropped, 3)
        # very short sentences may come back, and a sentence still streaming in is left alone
        self.assertEqual(drop_repeats("Mira smiled. A warm light flickered under the old", ["Mira smiled."])[0],
                         "Mira smiled. A warm light flickered under the old")
        self.assertEqual(drop_repeats(page, []), (page, 0))
        # parse_page drops them, but counts the words before it does
        new = " ".join(f"Brand new sentence number {n} is here." for n in range(8))
        reply = json.dumps({"segment": f"A warm light flickered under the old wooden door. {new}"})
        parsed = parse_page(reply, False, False, earlier_pages=earlier)
        self.assertTrue(parsed.segment.startswith("Brand new"))
        self.assertEqual(parsed.repeats_dropped, 1)

    def test_repeat_filter_never_leaves_a_page_too_short(self):
        from storyforge.llm import drop_repeats, parse_page
        earlier = ["Mira ran all the way down the grassy hill. A warm light flickered under the old wooden door. "
                   "She wondered what was waiting inside."]
        # 21 words, of which the filter would leave 6: the original sentences are kept instead
        page = ("Mira ran all the way down the grassy hill! A warm light flickered under the old wooden door. "
                "Then she knocked twice.")
        self.assertEqual(drop_repeats(page, earlier, floor=15), (page, 0))
        self.assertEqual(drop_repeats(page, earlier)[1], 2)  # without a floor they would go
        parsed = parse_page(json.dumps({"segment": page}), False, False, earlier_pages=earlier)
        self.assertEqual((parsed.segment, parsed.repeats_dropped), (page, 0))

    def test_only_finished_and_checked_sentences_stream_in(self):
        from storyforge.llm import drop_repeats
        earlier = ["A warm light flickered under the old wooden door."]
        shown = lambda text: drop_repeats(text, earlier, finished_only=True)[0]
        self.assertEqual(shown("Mira knocked twice. Then the do"), "Mira knocked twice.")
        # a repeated sentence never shows up, even for a moment
        self.assertEqual(shown("A warm light flickered under the old wood"), "")
        self.assertEqual(shown("A warm light flickered under the old wooden door. Mira smi"), "")
        self.assertEqual(shown("Mira knocked.\n\nA warm light flickered under the old wooden door. She"), "Mira knocked.")
        # with nothing earlier (page 1), still only finished sentences
        self.assertEqual(drop_repeats("One. Two thr", [], finished_only=True)[0], "One.")

    def test_props_become_emoji_and_stay_out_of_the_background(self):
        from storyforge.images import ImageWorker
        from storyforge.llm import clean_props, extract_plan, prop_emoji, response_schema
        self.assertEqual(prop_emoji(["a ripe strawberry", "Seashells"]),
                         [{"name": "a ripe strawberry", "emoji": "🍓"}, {"name": "Seashells", "emoji": "🐚"}])
        self.assertEqual(prop_emoji(["ice cream cone", "glimmering zorb"]), [{"name": "ice cream cone", "emoji": "🍦"}])
        self.assertEqual(prop_emoji(["a teddy bear", "golden keys", "a kite"]),  # two at most
                         [{"name": "a teddy bear", "emoji": "🧸"}, {"name": "golden keys", "emoji": "🔑"}])
        self.assertEqual(clean_props(["a sword", "a shiny key.", "A shiny key", "x", "a kite", "a ball"]),
                         ["a shiny key", "a kite"])  # no weapons, no repeats, two at most
        self.assertEqual(clean_props("a kite"), [])
        # in the plan, before the action, so they are known when the background starts
        schema = response_schema(False, False)["properties"]
        self.assertEqual(schema["props"], {"type": "array", "items": {"type": "string"}, "minItems": 0, "maxItems": 2})
        self.assertLess(list(schema).index("props"), list(schema).index("action"))
        raw = ('{"setting": "a sunny beach", "visual_details": ["soft waves", "a sandcastle"], '
               '"props": ["a pink seashell"], "action": "walking", "segment": "Mi')
        self.assertEqual(extract_plan(raw)["props"], ["a pink seashell"])
        self.assertEqual(extract_plan(raw.replace('"props": ["a pink seashell"], ', ""))["props"], [])  # optional
        # never painted: not in the place, and a detail with one is dropped
        prompt = ImageWorker.build_background_prompt({}, {
            "setting": "a strawberry field by the sea", "prop_names": ["a ripe strawberry", "a pink seashell"],
            "visual_details": ["seashells in the sand", "a striped windmill", "pink clouds"]})
        self.assertIn("a field by the sea, a striped windmill, pink clouds, ", prompt)  # "pink" is only a colour
        self.assertNotIn("strawberr", prompt)
        self.assertNotIn("shell", prompt)

    def test_prompt_stays_compact(self):
        from storyforge.llm import build_messages
        long_page = "word " * 400
        story = {"premise": "p", "title": "t", "character_look": "c", "world": "w", "summary": "s",
                 "segments": [{"status": "ready", "text": long_page} for _ in range(30)]}
        text = build_messages(story, "Open the door", False)[1]["content"]
        # Only the latest page (trimmed) is sent, not all thirty. With the page-plan instructions a
        # realistic worst case measured 176 + 717 prompt tokens + 480 reply = ~1400 of llama-server's 2048.
        self.assertLess(len(text), 2800)
        self.assertIn("Write page 31 only", text)
        self.assertIn("Open the door", text)


class Flow(unittest.TestCase):
    def setUp(self):
        llm_mode("normal")
        FAIL_FILE.unlink(missing_ok=True)

    def test_01_pages_load(self):
        status, body = call("GET", f"{BASE}/")
        self.assertEqual(status, 200)
        self.assertIn(b"Start Journey", body)
        for asset in ("app.js", "style.css"):
            self.assertEqual(call("GET", f"{BASE}/static/{asset}")[0], 200)
        status, health = call("GET", f"{BASE}/api/health")
        self.assertTrue(health["story_writer"]["ok"])
        self.assertTrue(health["pictures"]["ok"])
        self.assertTrue(health["pictures"]["options_checked_against_help"])

    def test_02_only_the_opening_is_written(self):
        llm_mode("normal", delay=0.02)
        story_id = new_story()
        # text streams in before the page is finished
        partial = wait_for(lambda: (s := segment(story_id, 0))["text"] and s)
        self.assertIn(partial["status"], ("writing", "ready"))
        # narration can start (text_complete) before choices exist and before the picture
        early = wait_for(lambda: (s := segment(story_id, 0))["text_complete"] and s)
        self.assertNotEqual(early["image"]["status"], "completed")
        done = wait_ready(story_id, 0)
        self.assertEqual(done["status"], "ready")
        self.assertEqual(done["title"], "Mira and the Tenth Mushroom")
        self.assertEqual(len(done["choices"]), 3)
        self.assertNotIn("visual_prompt", done["text"])
        self.assertNotIn("{", done["text"])
        status, story = call("GET", f"{BASE}/api/stories/{story_id}")
        self.assertEqual(len(story["segments"]), 1)  # one page, not a whole story
        self.assertEqual(call("GET", f"{LLM}/state")[1]["last_format"], "json_schema")
        image = wait_image(story_id, 0)["image"]
        self.assertEqual(image["status"], "completed")
        status, png = call("GET", BASE + image["url"])
        self.assertEqual(status, 200)
        self.assertEqual(png[:8], b"\x89PNG\r\n\x1a\n")

    def test_03_choices_continue_the_story_with_context(self):
        story_id = new_story()
        wait_ready(story_id, 0)
        status, story = call("POST", f"{BASE}/api/stories/{story_id}/continue",
                             {"kind": "choice", "action": "Open the tiny wooden door", "after": 0})
        self.assertEqual(status, 201)
        second = wait_ready(story_id, 1)
        self.assertEqual(second["status"], "ready")
        sent = call("GET", f"{LLM}/state")[1]["last_user"]
        self.assertIn("Open the tiny wooden door", sent)       # the decision
        self.assertIn("Story so far: Mira", sent)               # a short summary
        # nothing of the previous page's text: it tempted the model to retell how that page ended
        self.assertNotIn("Where the last page ended", sent)
        self.assertNotIn("A warm light flickered under it", sent)
        self.assertNotIn("ten. The tenth mushroom", sent)
        self.assertTrue(sent.endswith("This page must show: Open the tiny wooden door"))  # the choice comes last
        schema = call("GET", f"{LLM}/state")[1]["last_schema"]
        self.assertEqual(list(schema["properties"])[:2], ["event", "setting"])  # ...and opens the plan
        self.assertEqual(schema["properties"]["event"]["const"], "Open the tiny wooden door")
        self.assertIn("Mira chose to open the tiny wooden door", second["text"])
        self.assertIn("purple baby dragon", sent)               # the character's look
        # custom idea, then plain continue, then the ending
        call("POST", f"{BASE}/api/stories/{story_id}/continue",
             {"kind": "custom", "action": "Mira sneezes sparkles and the jars wake up", "after": 1})
        wait_ready(story_id, 2)
        self.assertIn("sneezes sparkles", call("GET", f"{LLM}/state")[1]["last_user"])
        call("POST", f"{BASE}/api/stories/{story_id}/continue", {"kind": "continue", "after": 2})
        wait_ready(story_id, 3)
        call("POST", f"{BASE}/api/stories/{story_id}/continue", {"kind": "finish", "after": 3})
        last = wait_ready(story_id, 4)
        self.assertTrue(last["final"])
        self.assertEqual(last["choices"], [])
        self.assertTrue(last["ended"])
        status, body = call("POST", f"{BASE}/api/stories/{story_id}/continue", {"kind": "continue", "after": 4})
        self.assertEqual(status, 409)
        # every page gets its own picture, one at a time, each exactly once
        for index in range(5):
            self.assertEqual(wait_image(story_id, index)["image"]["status"], "completed")
        calls = [row for row in SD_LOG.read_text().splitlines() if f"sf_{story_id}_" in row]
        self.assertEqual(len(calls), 5)
        self.assertTrue(all("--seed" in row and "--negative-prompt" in row and "--vae-tiling" in row for row in calls))
        self.assertTrue(all("-p children's storybook watercolor" in row for row in calls))  # style comes first
        self.assertTrue(all("--steps 8 " in row and "--taesd" not in row for row in calls))  # classic mode

    def test_04_double_clicks_do_not_create_extra_pages(self):
        llm_mode("normal", delay=0.01)
        story_id = new_story()
        wait_ready(story_id, 0)
        before = call("GET", f"{LLM}/state")[1]["requests"]
        body = {"kind": "choice", "action": "Knock and wait politely", "after": 0}
        results = [call("POST", f"{BASE}/api/stories/{story_id}/continue", body) for _ in range(4)]
        self.assertEqual(results[0][0], 201)
        for status, payload in results[1:]:
            self.assertEqual(status, 200)
            self.assertTrue(payload["duplicate"])
        wait_ready(story_id, 1)
        status, story = call("GET", f"{BASE}/api/stories/{story_id}")
        self.assertEqual(len(story["segments"]), 2)
        self.assertEqual(call("GET", f"{LLM}/state")[1]["requests"], before + 1)
        # asking again for a picture that exists does nothing
        wait_image(story_id, 1)
        count = len(SD_LOG.read_text().splitlines())
        call("POST", f"{BASE}/api/stories/{story_id}/segments/1/image/retry")
        time.sleep(0.5)
        self.assertEqual(len(SD_LOG.read_text().splitlines()), count)

    def test_05_messy_model_output_is_handled(self):
        for mode in ("fenced", "truncated", "no_schema"):
            with self.subTest(mode=mode):
                llm_mode(mode)
                story_id = new_story()
                page = wait_ready(story_id, 0)
                self.assertEqual(page["status"], "ready", mode)
                self.assertGreater(len(page["text"].split()), 40)
                self.assertGreaterEqual(len(page["choices"]), 2)
                self.assertNotIn("```", page["text"])
        self.assertEqual(call("GET", f"{LLM}/state")[1]["last_format"], "json_object")

    def test_06_unusable_output_fails_politely_and_can_be_retried(self):
        llm_mode("garbage")
        story_id = new_story()
        page = wait_ready(story_id, 0)
        self.assertEqual(page["status"], "failed")
        self.assertEqual(page["error"]["code"], "writer_failed")
        self.assertNotIn("Traceback", json.dumps(page))
        llm_mode("normal")
        call("POST", f"{BASE}/api/stories/{story_id}/segments/0/retry")
        self.assertEqual(wait_ready(story_id, 0)["status"], "ready")

    def test_07_failed_picture_does_not_break_the_story(self):
        FAIL_FILE.write_text("fail")
        story_id = new_story()
        wait_ready(story_id, 0)
        image = wait_image(story_id, 0)["image"]
        # nothing could paint it, so a ready-made forest is shown instead of no picture
        self.assertEqual(image["status"], "completed")
        self.assertTrue(image["preset"])
        status, jpg = call("GET", BASE + image["url"])
        self.assertEqual((status, jpg[:3]), (200, b"\xff\xd8\xff"))
        # the preset that fits the page's plan ("a cosy village bakery kitchen")
        self.assertEqual(jpg, (ROOT / "static" / "art" / "preset-village.jpg").read_bytes())
        # the story carries on regardless
        status, _ = call("POST", f"{BASE}/api/stories/{story_id}/continue", {"kind": "continue", "after": 0})
        self.assertEqual(status, 201)
        self.assertEqual(wait_ready(story_id, 1)["status"], "ready")
        # and the picture can be retried once the problem is gone
        wait_image(story_id, 1)
        FAIL_FILE.unlink()
        call("POST", f"{BASE}/api/stories/{story_id}/segments/0/image/retry")
        image = wait_for(lambda: (s := segment(story_id, 0))["image"]["status"] == "completed"
                         and not s["image"]["preset"] and s)["image"]
        self.assertEqual(call("GET", BASE + image["url"])[1][:8], b"\x89PNG\r\n\x1a\n")  # painted for real

    def test_08_server_stays_responsive_while_generating(self):
        llm_mode("normal", delay=0.03)
        story_id = new_story()
        slowest = 0.0
        while segment(story_id, 0)["status"] == "writing":
            started = time.time()
            self.assertEqual(call("GET", f"{BASE}/api/config")[0], 200)
            slowest = max(slowest, time.time() - started)
        self.assertLess(slowest, 0.5)
        while segment(story_id, 0)["image"]["status"] in ("queued", "generating"):
            started = time.time()
            self.assertEqual(call("GET", f"{BASE}/static/style.css")[0], 200)
            slowest = max(slowest, time.time() - started)
        self.assertLess(slowest, 0.5)

    def test_09_starting_over_cancels_the_old_story(self):
        llm_mode("normal", delay=0.05)
        old = new_story()
        wait_for(lambda: segment(old, 0)["text"])
        cancelled_before = call("GET", f"{LLM}/state")[1].get("cancelled", 0)
        self.assertEqual(call("POST", f"{BASE}/api/stories/{old}/abandon")[0], 200)
        llm_mode("normal")
        fresh = new_story("A robot learns to bake bread")
        self.assertEqual(wait_ready(fresh, 0)["status"], "ready")
        self.assertEqual(segment(old, 0)["status"], "failed")
        self.assertEqual(call("POST", f"{BASE}/api/stories/{old}/continue", {"kind": "continue", "after": 0})[0], 409)
        wait_for(lambda: call("GET", f"{LLM}/state")[1].get("cancelled", 0) > cancelled_before, timeout=10)
        time.sleep(0.5)
        self.assertFalse([r for r in SD_LOG.read_text().splitlines() if f"sf_{old}_" in r])

    def test_10_bad_input_is_rejected(self):
        self.assertEqual(call("POST", f"{BASE}/api/stories", {"premise": ""})[0], 422)
        self.assertEqual(call("POST", f"{BASE}/api/stories", {"premise": "x" * 5000})[0], 422)
        self.assertEqual(call("POST", f"{BASE}/api/stories", raw=b"not json")[0], 422)
        self.assertEqual(call("GET", f"{BASE}/api/stories/../../etc/passwd")[0], 404)
        self.assertEqual(call("GET", f"{BASE}/api/stories/0123456789abcdef")[0], 404)
        self.assertEqual(call("GET", f"{BASE}/api/stories/..%2f..%2fsettings/segments/0/image")[0], 404)
        self.assertEqual(call("GET", f"{BASE}/static/..%2fsettings.env")[0], 404)
        story_id = new_story()
        wait_ready(story_id, 0)
        self.assertEqual(call("GET", f"{BASE}/api/stories/{story_id}/segments/99")[0], 404)
        self.assertEqual(call("POST", f"{BASE}/api/stories/{story_id}/continue",
                              {"kind": "custom", "action": "", "after": 0})[0], 422)
        self.assertEqual(call("POST", f"{BASE}/api/stories/{story_id}/continue",
                              {"kind": "explode", "action": "x", "after": 0})[0], 422)
        self.assertEqual(call("POST", f"{BASE}/api/transcribe", raw=b"RIFF")[0], 404)  # whisper not installed here

    def test_11_story_is_saved_to_disk(self):
        story_id = new_story()
        wait_ready(story_id, 0)
        wait_image(story_id, 0)
        saved = json.loads((TMP / "data" / "sessions" / f"{story_id}.json").read_text())
        self.assertEqual(saved["segments"][0]["status"], "ready")
        self.assertEqual(saved["segments"][0]["image"]["status"], "completed")
        self.assertTrue(saved["summary"])
        self.assertTrue(saved["character_look"])
        status, metrics = call("GET", f"{BASE}/api/metrics")
        self.assertGreater(metrics["pages_measured"], 0)
        self.assertIsNotNone(metrics["avg_image_seconds"])

    def test_13_fast_mode_and_fallback(self):
        # (runs last) once the fast-mode files exist they are used automatically
        FAST_DIR.mkdir(exist_ok=True)
        for name in ("lcm-lora-sdv1-5.safetensors", "taesd.safetensors"):
            (FAST_DIR / name).write_bytes(b"x")
        story_id = new_story()
        wait_ready(story_id, 0)
        self.assertEqual(wait_image(story_id, 0)["image"]["status"], "completed")
        row = [r for r in SD_LOG.read_text().splitlines() if f"sf_{story_id}_0" in r][-1]
        self.assertIn("--steps 6 ", row)
        self.assertIn("--cfg-scale 1 --sampling-method lcm", row)
        self.assertIn("<lora:lcm-lora-sdv1-5:1>", row)
        self.assertIn("--taesd", row)
        self.assertNotIn("--vae-tiling", row)
        self.assertNotIn("--negative-prompt", row)
        self.assertIn("fast (lcm+taesd), 6 steps", call("GET", f"{BASE}/api/health")[1]["pictures"]["mode"])
        # if fast mode fails on this machine, the picture is repainted the classic way
        FAST_FAIL.write_text("x")
        call("POST", f"{BASE}/api/stories/{story_id}/continue", {"kind": "continue", "after": 0})
        wait_ready(story_id, 1)
        self.assertEqual(wait_image(story_id, 1)["image"]["status"], "completed")
        rows = [r for r in SD_LOG.read_text().splitlines() if f"sf_{story_id}_1" in r]
        self.assertEqual(len(rows), 2)
        self.assertIn("--taesd", rows[0])
        self.assertIn("--vae-tiling", rows[1])
        self.assertIn("--steps 8 ", rows[1])
        self.assertIn("classic", call("GET", f"{BASE}/api/health")[1]["pictures"]["mode"])

    def test_14_the_choice_is_followed_or_bridged(self):
        story_id = new_story()
        wait_ready(story_id, 0)
        log_file = TMP / "logs" / "storyforge.log"
        # followed: nothing added
        call("POST", f"{BASE}/api/stories/{story_id}/continue", {"kind": "choice", "action": "Knock and wait politely", "after": 0})
        page = wait_ready(story_id, 1)
        self.assertTrue(page["text"].startswith("Mira chose to knock and wait politely."), page["text"][:80])
        self.assertIn("choice='Knock and wait politely'", log_file.read_text())
        self.assertIn("result=passed", log_file.read_text())
        # ignored: never written again (the reader is watching it); a bridge sentence carries the choice
        # out instead (none of these words is on the fake's page)
        call("POST", f"{LLM}/control", {"ignore_choice": 1})
        before = call("GET", f"{LLM}/state")[1]["requests"]
        call("POST", f"{BASE}/api/stories/{story_id}/continue", {"kind": "choice", "action": "Climb the tallest oak tree", "after": 1})
        page = wait_ready(story_id, 2)
        self.assertEqual(call("GET", f"{LLM}/state")[1]["requests"] - before, 1)
        self.assertTrue(page["text"].startswith("So our hero decided to climb the tallest oak tree. "), page["text"][:80])
        self.assertIn("result=bridged", log_file.read_text())
        self.assertRegex(log_file.read_text(), r"page check failed  story=\w+ page=3  check=choice  choice='Climb the tallest oak tree'")
        # a scary choice is made gentle before the storyteller sees it
        call("POST", f"{BASE}/api/stories/{story_id}/continue", {"kind": "custom", "action": "fight the dragon with a sword", "after": 2})
        page = wait_ready(story_id, 3)
        sent = call("GET", f"{LLM}/state")[1]["last_user"]
        self.assertTrue(sent.endswith("This page must show: make friends with the dragon"))
        self.assertNotIn("fight", sent)
        self.assertNotIn("sword", sent)
        self.assertIn("Mira chose to make friends with the dragon", page["text"])

    def test_15_short_and_cut_off_pages(self):
        llm_mode("normal")
        log_file = TMP / "logs" / "storyforge.log"
        story_id = new_story()
        wait_ready(story_id, 0)
        # 20 words: short, but kept as it is, at the first try
        call("POST", f"{LLM}/control", {"short_pages": 1, "short_words": 20})
        before = call("GET", f"{LLM}/state")[1]["requests"]
        call("POST", f"{BASE}/api/stories/{story_id}/continue", {"kind": "continue", "action": "", "after": 0})
        page = wait_ready(story_id, 1)
        self.assertEqual((page["status"], len(page["text"].split())), ("ready", 20))
        self.assertEqual(call("GET", f"{LLM}/state")[1]["requests"] - before, 1)
        self.assertRegex(log_file.read_text(), rf"page check failed  story={story_id} page=2  check=length  words=20")
        # cut off by the token limit: the log says so, and the second try is used
        call("POST", f"{LLM}/control", {"cut_off_pages": 1})
        call("POST", f"{BASE}/api/stories/{story_id}/continue", {"kind": "continue", "action": "", "after": 1})
        page = wait_ready(story_id, 2)
        self.assertEqual(page["status"], "ready")
        self.assertRegex(log_file.read_text(), rf"page={3}  check=unusable  attempt=1  .*finish=length \(cut off\)"
                                               r".*text_closed=False.*words_before_filter=\d+")
        # both tries too short: the longer text is shown, with the bridge sentence for the choice it misses
        call("POST", f"{LLM}/control", {"short_pages": 2, "short_words": 8, "ignore_choice": 2})
        before = call("GET", f"{LLM}/state")[1]["requests"]
        call("POST", f"{BASE}/api/stories/{story_id}/continue", {"kind": "choice", "action": "Climb the tallest oak tree", "after": 2})
        page = wait_ready(story_id, 3)
        self.assertEqual(call("GET", f"{LLM}/state")[1]["requests"] - before, 2)
        self.assertEqual(page["status"], "ready")
        self.assertTrue(page["text"].startswith("So our hero decided to climb the tallest oak tree. "), page["text"])
        self.assertGreaterEqual(len(page["choices"]), 2)
        self.assertIn("kept the longest text", log_file.read_text())
        # no text at all from either try: only then the friendly error (test_06 has the garbage case)
        call("POST", f"{LLM}/control", {"short_pages": 2, "short_words": 0})
        call("POST", f"{BASE}/api/stories/{story_id}/continue", {"kind": "continue", "action": "", "after": 3})
        page = wait_ready(story_id, 4)
        self.assertEqual((page["status"], page["error"]["code"]), ("failed", "writer_failed"))

    def test_12_writer_offline(self):
        llm_mode("error500")
        story_id = new_story()
        page = wait_ready(story_id, 0)
        self.assertEqual(page["status"], "failed")
        self.assertIn("Try again", page["error"]["message"])


def hf_mode(mode: str) -> None:
    call("POST", f"{HF}/control", {"mode": mode})


class CloudBackgrounds(unittest.TestCase):
    """The second instance: IMAGE_MODE=huggingface against the fake router, falling back to the
    fake local sd-cli. Its key comes from a .env-style file (STORYFORGE_ENV_FILE)."""

    def setUp(self):
        llm_mode("normal")
        hf_mode("normal")
        FAIL_FILE.unlink(missing_ok=True)

    def sd_rows(self, story_id: str) -> list[str]:
        return [r for r in SD_LOG.read_text().splitlines() if f"sf_{story_id}_" in r] if SD_LOG.exists() else []

    def test_1_background_starts_while_the_page_is_written(self):
        llm_mode("normal", delay=0.03)
        before = call("GET", f"{HF}/state")[1]["requests"]
        story_id = new_story(base=CLOUD)
        # the setting streams in first, so the cloud is asked before the page is finished
        wait_for(lambda: call("GET", f"{HF}/state")[1]["requests"] > before)
        self.assertEqual(segment(story_id, 0, CLOUD)["status"], "writing")
        state = call("GET", f"{HF}/state")[1]
        self.assertEqual(state["last_path"], "/nscale/v1/images/generations")
        self.assertIn("a cosy village bakery kitchen at dawn, a warm brick oven, flour on the wooden table", state["last_prompt"])
        self.assertIn("no characters", state["last_prompt"])
        self.assertNotIn("dragon", state["last_prompt"])  # the hero is drawn by the child, never painted
        self.assertNotIn("strawberr", state["last_prompt"])  # nor what it holds
        page = wait_image(story_id, 0, CLOUD)
        self.assertEqual(page["image"]["status"], "completed")
        self.assertIn("bakery kitchen", page["image"]["alt"])
        self.assertEqual(page["hero_action"], "waving")
        self.assertEqual(page["props"], [{"name": "a ripe strawberry", "emoji": "🍓"}])  # "glimmering zorb": no emoji
        status, png = call("GET", CLOUD + page["image"]["url"])
        self.assertEqual((status, png[:8]), (200, b"\x89PNG\r\n\x1a\n"))
        self.assertEqual(self.sd_rows(story_id), [])  # nothing painted on the Jetson
        # the hero layer: the placeholder until /draw exists
        status, story = call("GET", f"{CLOUD}/api/stories/{story_id}")
        status, hero = call("GET", CLOUD + story["hero_url"])
        self.assertEqual((status, hero[:8]), (200, b"\x89PNG\r\n\x1a\n"))
        self.assertEqual(hero, (ROOT / "static" / "art" / "hero-placeholder.png").read_bytes())

    def test_2_slow_cloud_gives_up_and_paints_locally(self):
        hf_mode("slow")
        story_id = new_story(base=CLOUD)
        wait_ready(story_id, 0, CLOUD)
        started = time.time()
        page = wait_for(lambda: (s := segment(story_id, 0, CLOUD))["image"]["status"] == "completed" and s, timeout=30)
        self.assertLess(time.time() - started, 15)
        self.assertEqual(len(self.sd_rows(story_id)), 1)  # repainted by the local fallback
        self.assertIn("no characters", self.sd_rows(story_id)[0])
        hf_mode("normal")

    def test_3_cloud_error_falls_back_and_retry_asks_the_cloud_again(self):
        hf_mode("error500")
        FAIL_FILE.write_text("fail")  # and the local painter fails too
        story_id = new_story(base=CLOUD)
        wait_ready(story_id, 0, CLOUD)
        image = wait_image(story_id, 0, CLOUD)["image"]
        self.assertEqual(image["status"], "completed")
        self.assertTrue(image["preset"])  # neither painter worked: a preset forest
        self.assertEqual(image["alt"], "a storybook village scene")
        self.assertEqual(len(self.sd_rows(story_id)), 1)
        FAIL_FILE.unlink()
        hf_mode("normal")
        before = call("GET", f"{HF}/state")[1]["requests"]
        call("POST", f"{CLOUD}/api/stories/{story_id}/segments/0/image/retry")
        image = wait_for(lambda: (s := segment(story_id, 0, CLOUD))["image"]["status"] == "completed"
                         and not s["image"]["preset"] and s)["image"]
        self.assertIn("bakery kitchen", image["alt"])
        self.assertGreater(call("GET", f"{HF}/state")[1]["requests"], before)
        self.assertEqual(len(self.sd_rows(story_id)), 1)  # the cloud painted it this time

    def test_4_key_is_loaded_from_env_file_and_never_shown(self):
        status, health = call("GET", f"{CLOUD}/api/health")
        self.assertTrue(health["pictures"]["hf_key_set"])
        self.assertIn("huggingface", health["pictures"]["mode"])
        story_id = new_story(base=CLOUD)
        wait_image(story_id, 0, CLOUD)
        pages = [call("GET", f"{CLOUD}/api/{p}")[1] for p in ("health", "config", f"stories/{story_id}")]
        self.assertNotIn("test-token", json.dumps(pages))
        for log_file in (TMP / "cloud_logs").glob("*"):
            self.assertNotIn("test-token", log_file.read_text(errors="replace"), log_file.name)


def drawing_payload(idea: str = "") -> dict:
    import base64
    from tests.fake_hf import tiny_png
    hero = tiny_png(64)
    jpeg = b"\xff\xd8\xff\xe0" + os.urandom(600)  # only the start of a JPEG is checked
    return {
        "hero_png": "data:image/png;base64," + base64.b64encode(hero).decode(),
        "preview_jpeg": "data:image/jpeg;base64," + base64.b64encode(jpeg).decode(),
        "idea": idea,
    }, hero


class DrawnHero(unittest.TestCase):
    """/draw: the child's drawing becomes the hero, described by a vision model, then page 1 starts."""

    def setUp(self):
        llm_mode("normal")
        hf_mode("normal")
        call("POST", f"{HF}/control", {"vision": "normal", "vision_models": []})

    def test_1_draw_page_loads(self):
        status, body = call("GET", f"{CLOUD}/draw")
        self.assertEqual(status, 200)
        self.assertIn(b"Draw your hero", body)
        for asset in ("draw.js", "draw.css"):
            self.assertEqual(call("GET", f"{CLOUD}/static/{asset}")[0], 200)
        # Without this rule the loading overlay (display: grid) shows from the start and covers the paper.
        self.assertIn(b"[hidden] { display: none !important; }", call("GET", f"{CLOUD}/static/draw.css")[1])
        self.assertIn(b"GIVE_UP_MS = 45000", call("GET", f"{CLOUD}/static/draw.js")[1])
        self.assertIn(b'href="/draw"', call("GET", f"{CLOUD}/")[1])

    def test_2_drawing_becomes_the_hero_of_a_new_story(self):
        payload, hero = drawing_payload()
        status, story = call("POST", f"{CLOUD}/api/stories/drawn", payload)
        self.assertEqual(status, 201, story)
        description = "a round orange cat with a tall green hat and a long striped tail"
        self.assertEqual(story["hero_description"], description)
        self.assertEqual(story["premise"], "A story starring a round orange cat")
        self.assertEqual(len(story["segments"]), 1)
        vision = call("GET", f"{HF}/state")[1]
        self.assertEqual(vision["vision_models"], ["meta-llama/Llama-4-Scout-17B-16E-Instruct"])
        self.assertTrue(vision["vision_had_image"])
        # the saved drawing is served as this story's hero, byte for byte
        status, png = call("GET", CLOUD + story["hero_url"])
        self.assertEqual((status, png), (200, hero))
        self.assertNotEqual(story["hero_url"].split("v=")[1], "0")
        # the storyteller is told about the drawn hero, and keeps its look
        page = wait_ready(story["id"], 0, CLOUD)
        self.assertEqual(page["status"], "ready")
        first_prompt = call("GET", f"{LLM}/state")[1]["last_user"]
        # notes to use, not a sentence to copy; and a name chosen before the page is written
        self.assertIn("The hero is a character the reader drew: a round orange cat. "
                      "Notes: a tall green hat; a long striped tail.", first_prompt)
        self.assertNotIn(description, first_prompt)
        self.assertLess(first_prompt.index('"hero_name"'), first_prompt.index('"segment"'))
        saved = json.loads((TMP / "cloud_data" / "sessions" / f"{story['id']}.json").read_text())
        self.assertEqual(saved["character_look"], description)
        self.assertEqual(saved["hero_description"], description)
        self.assertTrue((TMP / "cloud_data" / "heroes" / f"{story['id']}.png").is_file())
        call("POST", f"{CLOUD}/api/stories/{story['id']}/continue", {"kind": "continue", "after": 0})
        wait_ready(story["id"], 1, CLOUD)
        later_prompt = call("GET", f"{LLM}/state")[1]["last_user"]
        self.assertIn("Main character's name: Captain Whiskers", later_prompt)
        self.assertIn(f"Main character looks like: {description}", later_prompt)
        self.assertEqual(json.loads((TMP / "cloud_data" / "sessions" / f"{story['id']}.json").read_text())["hero_name"],
                         "Captain Whiskers")

    def test_3_next_vision_model_is_tried_and_the_story_starts_even_without_one(self):
        call("POST", f"{HF}/control", {"vision": "first_fails", "vision_models": []})
        status, story = call("POST", f"{CLOUD}/api/stories/drawn", drawing_payload("a trip to the moon")[0])
        self.assertEqual(status, 201)
        self.assertEqual(call("GET", f"{HF}/state")[1]["vision_models"],
                         ["meta-llama/Llama-4-Scout-17B-16E-Instruct", "google/gemma-3-27b-it"])
        self.assertTrue(story["hero_description"].startswith("a round orange cat"))
        self.assertEqual(story["premise"], "a trip to the moon")  # the reader's own idea is kept
        call("POST", f"{HF}/control", {"vision": "down", "vision_models": []})
        status, story = call("POST", f"{CLOUD}/api/stories/drawn", drawing_payload()[0])
        self.assertEqual(status, 201)
        self.assertEqual(story["hero_description"], "a brave little hero in bright colours")
        self.assertEqual(wait_ready(story["id"], 0, CLOUD)["status"], "ready")

    def test_5_who_is_this_beats_the_guess_and_everything_is_logged(self):
        payload = dict(drawing_payload()[0], who='my bunny "Ricky"')
        status, story = call("POST", f"{CLOUD}/api/stories/drawn", payload)
        self.assertEqual(status, 201, story)
        # the vision model is told what the child said ...
        self.assertIn("The child says this is: \"my bunny 'Ricky'\"", call("GET", f"{HF}/state")[1]["vision_prompt"])
        # ... and so is the storyteller, ahead of the model's notes
        self.assertEqual(story["premise"], "A story starring my bunny 'Ricky'")
        wait_ready(story["id"], 0, CLOUD)
        first = call("GET", f"{LLM}/state")[1]["last_user"]
        self.assertLess(first.index("The reader says their hero is \"my bunny 'Ricky'\". That comes first"),
                        first.index("The hero is a character the reader drew"))
        saved = json.loads((TMP / "cloud_data" / "sessions" / f"{story['id']}.json").read_text())
        self.assertEqual(saved["hero_who"], "my bunny 'Ricky'")
        call("POST", f"{CLOUD}/api/stories/{story['id']}/continue", {"kind": "continue", "after": 0})
        wait_ready(story["id"], 1, CLOUD)
        self.assertIn("The reader calls the hero: \"my bunny 'Ricky'\"", call("GET", f"{LLM}/state")[1]["last_user"])
        # each drawing's vision request is traceable: image size, exact prompt, model, raw answer
        log_text = (TMP / "cloud_logs" / "storyforge.log").read_text()
        self.assertIn("hero vision request  image=?", log_text)  # the test JPEG is not a real one
        self.assertIn("The child says this is", log_text)
        self.assertIn("hero vision answer  model=meta-llama/Llama-4-Scout-17B-16E-Instruct  raw='A round orange cat", log_text)
        # without a "who", nothing about it reaches either prompt
        status, story = call("POST", f"{CLOUD}/api/stories/drawn", drawing_payload()[0])
        self.assertNotIn("The child says", call("GET", f"{HF}/state")[1]["vision_prompt"])
        self.assertNotIn("hero_who", story)
        self.assertIn('id="who"', call("GET", f"{CLOUD}/draw")[1].decode())

    def test_4_bad_drawings_are_rejected(self):
        payload = drawing_payload()[0]
        for broken in (
            dict(payload, hero_png="data:image/png;base64," + "QUJD" * 40),  # not a PNG
            dict(payload, preview_jpeg=payload["hero_png"]),                  # not a JPEG
            dict(payload, hero_png="!!!not base64!!!" * 10),
            {"idea": "no pictures"},
        ):
            status, body = call("POST", f"{CLOUD}/api/stories/drawn", broken)
            self.assertEqual(status, 422, body)
            self.assertIn("message", body)


if __name__ == "__main__":
    unittest.main(verbosity=2)
