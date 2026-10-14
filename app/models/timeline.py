"""Independent narration and visual timelines for manuscript-driven editing."""

from dataclasses import asdict, dataclass, field


@dataclass
class NarrationSegment:
    id: str
    text_start: int
    text_end: int
    text: str
    requirement: dict = field(default_factory=dict)
    audio_path: str = ""
    audio_start: float = 0.0
    audio_end: float = 0.0


@dataclass
class VisualShot:
    asset_id: str
    scene_id: str
    video_path: str
    source_start: float
    source_end: float
    timeline_start: float
    timeline_end: float
    segment_id: str
    reason: str = ""
    weak_match: bool = False


def timeline_dict(segments, shots):
    return {"version": 1, "segments": [asdict(item) for item in segments], "shots": [asdict(item) for item in shots]}

