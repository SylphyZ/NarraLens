"""Persistent material-library records; paths refer to the original media."""

from dataclasses import dataclass


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


@dataclass(frozen=True)
class CandidateClip:
    scene_id: str
    asset_id: str
    video_path: str
    source_start: float
    source_end: float
    description: str
    episode: int
    score: float = 0.0

