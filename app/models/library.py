"""Persistent material-library records; paths refer to the original media."""

from dataclasses import dataclass
import hashlib
import json


LIBRARY_SCHEMA = 2
ANALYSIS_PROTOCOL = "content-observations-v1"
RETRIEVAL_PROTOCOL = "event-shot-match-v1"


def media_id(kind, *identity):
    """Stable IDs describe media/ranges, never a prompt, model, path or library."""
    return kind + hashlib.sha256(json.dumps(identity, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()[:20]


@dataclass(frozen=True)
class ShotRange:
    id: str
    shot_id: str
    start: float
    end: float
    natural_start: float
    natural_end: float


@dataclass(frozen=True)
class SearchPage:
    candidates: list
    cursor: int
    exhausted: bool
    reason: str = ""


@dataclass(frozen=True)
class VideoInspection:
    path: str
    size: int
    mtime_ns: int
    fingerprint: str
    duration: float
    width: int
    height: int
    fps: float


@dataclass(frozen=True)
class ImportMapping:
    video_path: str
    subtitle_path: str = ""
    episode: int = 1

