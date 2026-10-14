#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
Subtitle text utilities.

This module provides a shared, cross-platform way to read and normalize subtitle
content. Both Short Drama Editing (混剪) and Short Drama Narration (解说) should
consume subtitle content through this module to avoid platform-specific parsing
issues (e.g. Windows UTF-16 SRT, timestamp separators, etc.).
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Iterable, Optional


_SRT_TIME_RE = re.compile(
    r"\b\d{2}:\d{2}:\d{2}(?:[,.]\d{3})?\s*-->\s*\d{2}:\d{2}:\d{2}(?:[,.]\d{3})?\b"
)
_SRT_MS_DOT_RE = re.compile(r"(\b\d{2}:\d{2}:\d{2})\.(\d{3}\b)")


@dataclass(frozen=True)
class DecodedSubtitle:
    text: str
    encoding: str


def has_timecodes(text: str) -> bool:
    """Return True if the subtitle text contains at least one SRT timecode."""
    if not text:
        return False
    return _SRT_TIME_RE.search(text) is not None


def has_ass_events(text: str) -> bool:
    """Recognize readable ASS/SSA content when choosing a text encoding."""
    return bool(
        re.search(r"^\s*\[Events\]\s*$", text, re.MULTILINE | re.IGNORECASE)
        and re.search(r"^\s*Dialogue\s*:", text, re.MULTILINE | re.IGNORECASE)
    )


def convert_ass_to_srt(text: str) -> str:
    """Convert decoded ASS/SSA dialogue to SRT using the configured FFmpeg."""
    if not has_ass_events(text):
        raise ValueError("ASS/SSA subtitle has no readable dialogue events.")
    ffmpeg = os.environ.get("IMAGEIO_FFMPEG_EXE") or shutil.which("ffmpeg")
    if not ffmpeg:
        raise ValueError("FFmpeg is required to convert ASS/SSA subtitles.")
    with tempfile.TemporaryDirectory(prefix="narrato-ass-") as temp_dir:
        source_path = os.path.join(temp_dir, "source.ass")
        output_path = os.path.join(temp_dir, "converted.srt")
        with open(source_path, "w", encoding="utf-8") as source:
            source.write(text)
        try:
            result = subprocess.run(
                [ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
                 "-i", source_path, "-map", "0:s:0", "-c:s", "srt", output_path],
                capture_output=True, timeout=60, check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise ValueError("ASS/SSA conversion failed; check FFmpeg and the subtitle file.") from exc
        if result.returncode != 0 or not os.path.isfile(output_path):
            raise ValueError("FFmpeg could not convert the ASS/SSA subtitle; check its format.")
        with open(output_path, "r", encoding="utf-8-sig") as output:
            converted = normalize_subtitle_text(output.read())
    # FFmpeg may retain basic HTML styling or ASS alignment overrides in SRT.
    converted = re.sub(r"</?(?:b|i|u|s|font)(?:\s+[^>]*)?>", "", converted, flags=re.IGNORECASE)
    converted = re.sub(r"\{\\[^}]*\}", "", converted)
    if not has_timecodes(converted):
        raise ValueError("ASS/SSA conversion produced no timed text subtitles.")
    return converted


def normalize_subtitle_text(text: str) -> str:
    """
    Normalize subtitle text to improve cross-platform reliability.

    - Unifies line endings to LF
    - Removes BOM and NUL bytes
    - Normalizes millisecond separators from '.' to ',' in timecodes
    """
    if text is None:
        return ""

    normalized = str(text)

    # Strip BOM.
    if normalized.startswith("\ufeff"):
        normalized = normalized.lstrip("\ufeff")

    # Remove NUL bytes (common when UTF-16 is mis-decoded elsewhere).
    normalized = normalized.replace("\x00", "")

    # Normalize newlines.
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")

    # Normalize timestamp millisecond separator: 00:00:01.000 -> 00:00:01,000
    normalized = _SRT_MS_DOT_RE.sub(r"\1,\2", normalized)

    return normalized.strip()


def decode_subtitle_bytes(
    data: bytes,
    *,
    encodings: Optional[Iterable[str]] = None,
) -> DecodedSubtitle:
    """
    Decode subtitle bytes using a small set of common encodings.

    Prefer decodings with recognizable SRT timecodes or ASS/SSA dialogue events.
    """
    if data is None:
        return DecodedSubtitle(text="", encoding="utf-8")

    candidates = list(encodings) if encodings else [
        "utf-8",
        "utf-8-sig",
        "utf-16",
        "utf-16-le",
        "utf-16-be",
        "gbk",
        "gb2312",
    ]

    decoded_results: list[DecodedSubtitle] = []
    for encoding in candidates:
        try:
            decoded_text = data.decode(encoding)
        except UnicodeDecodeError:
            continue
        decoded_results.append(
            DecodedSubtitle(text=normalize_subtitle_text(decoded_text), encoding=encoding)
        )

        # Fast path: if we already see timecodes, keep the first such decode.
        if has_timecodes(decoded_results[-1].text) or has_ass_events(decoded_results[-1].text):
            return decoded_results[-1]

    if decoded_results:
        # Fall back to the first successful decoding.
        return decoded_results[0]

    # Last resort: replace undecodable bytes.
    return DecodedSubtitle(text=normalize_subtitle_text(data.decode("utf-8", errors="replace")), encoding="utf-8")


def read_subtitle_text(file_path: str) -> DecodedSubtitle:
    """Read subtitle file from disk, decode and normalize its text."""
    if not file_path or not str(file_path).strip():
        return DecodedSubtitle(text="", encoding="utf-8")

    normalized_path = os.path.abspath(str(file_path))
    with open(normalized_path, "rb") as f:
        data = f.read()

    decoded = decode_subtitle_bytes(data)
    if os.path.splitext(normalized_path)[1].lower() in {".ass", ".ssa"}:
        return DecodedSubtitle(text=convert_ass_to_srt(decoded.text), encoding=decoded.encoding)
    return decoded

