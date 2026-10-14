"""Xiaomi MiMo speech synthesis using its chat completions API."""

import base64
import io
import os
import unicodedata
import wave

import requests

from app.utils.openai_base_url_security import validate_openai_compatible_base_url


DEFAULT_BASE_URL = "https://api.xiaomimimo.com/v1"
MODELS = (
    "mimo-v2.5-tts",
    "mimo-v2.5-tts-voicedesign",
    "mimo-v2.5-tts-voiceclone",
)
VOICES = ("冰糖", "茉莉", "苏打", "白桦", "Mia", "Chloe", "Milo", "Dean", "mimo_default")
MAX_REFERENCE_BASE64_BYTES = 10 * 1024 * 1024
MAX_TTS_HAN = 12000


def tts_han_count(text):
    return sum(unicodedata.name(char, "").startswith(("CJK UNIFIED IDEOGRAPH", "CJK COMPATIBILITY IDEOGRAPH")) for char in text)


def validate_tts_text(text):
    if not text or not text.strip():
        raise ValueError("配音文本不能为空。")
    count = tts_han_count(text)
    if count > MAX_TTS_HAN:
        raise ValueError(f"单次配音含{count}个汉字，最多{MAX_TTS_HAN}个，请拆分此配音块。")
    return count


def synthesize(text: str, settings: dict, proxies: dict | None = None) -> tuple[bytes, float]:
    """Return WAV bytes and their actual duration; never rewrite narration text."""
    api_key = str(settings.get("api_key", "")).strip()
    if not api_key:
        raise ValueError("MiMo TTS: API Key is required.")
    validate_tts_text(text)
    base_url = validate_openai_compatible_base_url(settings.get("base_url") or DEFAULT_BASE_URL)
    model = str(settings.get("model", MODELS[0])).strip().lower()
    if model not in MODELS:
        raise ValueError("MiMo TTS: unsupported speech model.")

    instruction = str(settings.get("instruction", "")).strip()
    audio = {"format": "wav"}
    if model == "mimo-v2.5-tts-voiceclone":
        reference = str(settings.get("reference_audio", ""))
        extension = os.path.splitext(reference)[1].lower()
        if extension not in {".wav", ".mp3"} or not os.path.isfile(reference):
            raise ValueError("MiMo TTS: choose an existing WAV or MP3 reference audio file.")
        with open(reference, "rb") as source:
            reference_bytes = source.read(MAX_REFERENCE_BASE64_BYTES * 3 // 4 + 1)
        encoded = base64.b64encode(reference_bytes).decode("ascii")
        if not encoded or len(encoded) > MAX_REFERENCE_BASE64_BYTES:
            raise ValueError("MiMo TTS: reference audio must be nonempty and below 10 MB after Base64 encoding.")
        mime = "audio/wav" if extension == ".wav" else "audio/mpeg"
        audio["voice"] = f"data:{mime};base64,{encoded}"
    elif model == "mimo-v2.5-tts-voicedesign":
        if not instruction:
            raise ValueError("MiMo TTS: describe the voice before using voice design.")
        audio["optimize_text_preview"] = False
    else:
        selected_voice = str(settings.get("voice", "冰糖"))
        if selected_voice not in VOICES:
            raise ValueError("MiMo TTS: unsupported preset voice.")
        audio["voice"] = selected_voice

    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": text.strip()},
        ],
        "audio": audio,
        "stream": False,
    }
    try:
        response = requests.post(
            f"{base_url.rstrip('/')}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=payload,
            proxies=proxies,
            timeout=(15, 300),
        )
    except requests.RequestException:
        raise RuntimeError("MiMo TTS: connection failed; check the API endpoint and proxy.") from None
    if not response.ok:
        # Avoid echoing a request body containing an API key or reference audio.
        raise RuntimeError(
            f"MiMo TTS: HTTP {response.status_code}; check the API key, model access, reference audio and quota."
        )
    try:
        result = response.json()
        reason = result["choices"][0].get("finish_reason")
        if reason == "length":
            raise RuntimeError("MiMo TTS: 单次语音生成达到输出上限，请拆分当前配音块；截断音频不会作为成功缓存。")
        if reason == "content_filter" or result["choices"][0]["message"].get("refusal"):
            raise RuntimeError("MiMo TTS: 当前配音文本被内容审核拒绝。")
        encoded_audio = result["choices"][0]["message"]["audio"]["data"]
        output = base64.b64decode(encoded_audio, validate=True)
        with wave.open(io.BytesIO(output), "rb") as wav:
            duration = wav.getnframes() / wav.getframerate()
        if duration <= 0:
            raise ValueError("Empty audio")
    except (KeyError, IndexError, TypeError, ValueError, wave.Error, EOFError):
        raise RuntimeError("MiMo TTS: the API did not return valid WAV audio.") from None
    return output, duration
