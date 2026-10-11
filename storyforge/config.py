"""Settings. Values come from settings.env, then .env (secrets such as HF_TOKEN), and
environment variables override both. .env is never committed."""
from __future__ import annotations

import os
import shlex
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _read_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip()
    return values


_FILE = {
    **_read_env_file(Path(os.environ.get("STORYFORGE_SETTINGS", ROOT / "settings.env"))),
    **_read_env_file(Path(os.environ.get("STORYFORGE_ENV_FILE", ROOT / ".env"))),
}


def _get(key: str, default: str) -> str:
    return os.environ.get(key, _FILE.get(key, default))


def _int(key: str, default: int, low: int, high: int) -> int:
    try:
        value = int(_get(key, str(default)))
    except ValueError:
        value = default
    return max(low, min(high, value))


def _float(key: str, default: float, low: float, high: float) -> float:
    try:
        value = float(_get(key, str(default)))
    except ValueError:
        value = default
    return max(low, min(high, value))


@dataclass(frozen=True)
class Settings:
    llm_url: str = _get("LLM_URL", "http://127.0.0.1:8080").rstrip("/")
    llm_max_tokens: int = _int("LLM_MAX_TOKENS", 420, 150, 1200)
    llm_temperature: float = _float("LLM_TEMPERATURE", 0.8, 0.0, 1.5)
    llm_timeout: int = _int("LLM_TIMEOUT_SECONDS", 180, 20, 900)
    max_segments: int = _int("MAX_SEGMENTS", 40, 2, 200)

    sd_cli: Path = Path(_get("SD_CLI", "/home/akw/jetson-ai/stable-diffusion.cpp/build/bin/sd-cli"))
    sd_model: Path = Path(_get("SD_MODEL", "/home/akw/jetson-ai/models/images/v1-5-pruned_Q4_0.gguf"))
    image_dir: Path = Path(_get("IMAGE_DIR", "/home/akw/jetson-ai/generated_images"))
    sd_width: int = _int("SD_WIDTH", 512, 128, 1024)
    sd_height: int = _int("SD_HEIGHT", 512, 128, 1024)
    sd_steps: int = _int("SD_STEPS", 15, 1, 60)
    sd_extra_args: tuple[str, ...] = tuple(
        shlex.split(_get("SD_EXTRA_ARGS", "--diffusion-fa --vae-tiling --backend te=cpu --max-vram 0"))
    )
    sd_style: str = _get("SD_STYLE", "children's storybook watercolor illustration, soft bright colors")
    sd_negative: str = _get("SD_NEGATIVE", "text, watermark, signature, blurry, deformed, extra limbs, scary, photo")
    # Fast mode: used automatically when these files exist (scripts/get_fast_models.sh).
    sd_fast: bool = _get("SD_FAST", "1") not in ("0", "false", "no", "off")
    sd_lcm_lora: Path = Path(_get("SD_LCM_LORA", "/home/akw/jetson-ai/models/images/fast/lcm-lora-sdv1-5.safetensors"))
    sd_lcm_steps: int = _int("SD_LCM_STEPS", 6, 2, 12)
    sd_taesd: Path = Path(_get("SD_TAESD", "/home/akw/jetson-ai/models/images/fast/taesd.safetensors"))
    sd_prompt_words: int = _int("SD_PROMPT_WORDS", 34, 10, 60)
    sd_timeout: int = _int("SD_TIMEOUT_SECONDS", 600, 30, 3600)
    sd_min_free_mb: int = _int("SD_MIN_FREE_MB", 1200, 0, 16000)
    # Where page backgrounds come from: huggingface | local | off. IMAGES_ENABLED=0 still means off.
    image_mode: str = (
        "off" if _get("IMAGES_ENABLED", "1").lower() in ("0", "false", "no", "off")
        else _get("IMAGE_MODE", "huggingface").lower()
    )
    hf_token: str = field(default=_get("HF_TOKEN", ""), repr=False)  # secret: lives in .env, never logged
    hf_url: str = _get("HF_URL", "https://router.huggingface.co").rstrip("/")
    hf_provider: str = _get("HF_PROVIDER", "nscale")
    hf_image_model: str = _get("HF_IMAGE_MODEL", "black-forest-labs/FLUX.1-schnell")
    hf_timeout: int = _int("HF_TIMEOUT_SECONDS", 20, 5, 120)
    hf_fallback_local: bool = _get("HF_FALLBACK_LOCAL", "1") not in ("0", "false", "no", "off")
    # Describes the child's hero drawing; tried in order through the same router and key.
    vision_models: tuple[str, ...] = tuple(
        m.strip() for m in _get(
            "VISION_MODELS", "meta-llama/Llama-4-Scout-17B-16E-Instruct,google/gemma-3-27b-it"
        ).split(",") if m.strip()
    )
    vision_timeout: int = _int("VISION_TIMEOUT_SECONDS", 20, 5, 120)

    tts_mode: str = _get("TTS_MODE", "auto").lower()
    piper_bin: Path = Path(_get("PIPER_BIN", "/home/akw/jetson-ai/.venv/bin/piper"))
    piper_voice: Path = Path(_get("PIPER_VOICE", "/home/akw/jetson-ai/models/voices/en_US-lessac-medium.onnx"))
    elevenlabs_api_key: str = field(default=_get("ELEVENLABS_API_KEY", ""), repr=False)  # secret: .env only
    elevenlabs_voice_id: str = _get("ELEVENLABS_VOICE_ID", "JBFqnCBsd6RMkjVDRZzb") or "JBFqnCBsd6RMkjVDRZzb"
    elevenlabs_model: str = _get("ELEVENLABS_MODEL", "eleven_flash_v2_5")
    elevenlabs_url: str = _get("ELEVENLABS_URL", "https://api.elevenlabs.io").rstrip("/")
    elevenlabs_timeout: int = _int("ELEVENLABS_TIMEOUT_SECONDS", 10, 2, 60)

    whisper_cli: Path = Path(_get("WHISPER_CLI", "/home/akw/jetson-ai/whisper.cpp/build/bin/whisper-cli"))
    whisper_model: Path = Path(_get("WHISPER_MODEL", "/home/akw/jetson-ai/whisper.cpp/models/ggml-base.en.bin"))

    host: str = _get("HOST", "0.0.0.0")
    port: int = _int("PORT", 8000, 1, 65535)
    https_port: int = _int("HTTPS_PORT", 8443, 0, 65535)

    data_dir: Path = Path(_get("DATA_DIR", str(ROOT / "data")))
    log_dir: Path = Path(_get("LOG_DIR", str(ROOT / "logs")))
    cert_dir: Path = Path(_get("CERT_DIR", str(ROOT / "certs")))
    static_dir: Path = field(default=ROOT / "static")

    # Input limits (characters).
    max_premise_chars: int = 600
    max_action_chars: int = 300


settings = Settings()
