"""Persistent material-library records; paths refer to the original media."""

from dataclasses import dataclass
from bisect import bisect_left
from fractions import Fraction
import hashlib
import json
import math


LIBRARY_SCHEMA = 2
FRAME_PROTOCOL = "source-frames-v1"
ANALYSIS_PROTOCOL = "content-observations-v2"
RETRIEVAL_PROTOCOL = "event-shot-match-v2"


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
    start_frame: int
    end_frame: int
    shot_start_frame: int
    shot_end_frame: int


SOURCE_INDEX_FIELDS = ("index_protocol", "index_key", "stream_index", "time_base", "origin_pts",
                       "source_frame_count", "source_duration")


def range_input(item):
    """One projection for a range submitted to either analysis producer."""
    required = (*ShotRange.__dataclass_fields__, *SOURCE_INDEX_FIELDS, "frames", "cues",
                "fingerprint", "asset_id", "subtitle_identity", "generation")
    return {**{key: item[key] for key in required},
            **{key: item[key] for key in ("analysis_baseline", "evidence_images") if key in item}}


class SourceFrameIndex:
    """Presentation-order source frames; output-video FPS never enters this clock."""

    def __init__(self, data):
        self.data = data
        if data.get("protocol") != FRAME_PROTOCOL or data.get("complete") is not True:
            raise ValueError("需要按2.9.1重新建立完整源帧索引。")
        self.pts, self.durations, self.cuts = data["pts"], data["durations"], data["cuts"]
        clock = data["time_base"]
        if (not isinstance(clock, list) or len(clock) != 2 or any(type(v) is not int or v <= 0 for v in clock)
                or type(data["stream_index"]) is not int or data["stream_index"] < 0):
            raise ValueError("源帧索引的视频流或时间基无效。")
        self.clock = Fraction(*clock)
        if (not self.pts or len(self.pts) != len(self.durations) or
                any(type(v) is not int for v in self.pts) or
                any(type(v) is not int or v < 0 for v in self.durations) or
                any(a > b for a, b in zip(self.pts, self.pts[1:])) or self.durations[-1] <= 0):
            raise ValueError("源帧PTS缺失、倒退或尾帧结束时间未确认，不能建立精确索引。")
        self.count = len(self.pts)
        if (data.get("frame_count") != self.count or data.get("origin_pts") != self.pts[0] or
                not self.cuts or self.cuts[0] != 0 or self.cuts[-1] != self.count or
                any(type(v) is not int for v in self.cuts) or
                any(a >= b for a, b in zip(self.cuts, self.cuts[1:]))):
            raise ValueError("源帧索引的数量或镜头边界无效。")
        self.end_pts = self.pts[-1] + self.durations[-1]
        self.duration = self.time(self.count)
        if not math.isfinite(self.duration) or self.duration <= 0:
            raise ValueError("源视频显示时长无效。")
        if any(self.time(a) >= self.time(b) for a, b in zip(self.cuts, self.cuts[1:])):
            raise ValueError("物理镜头边界没有正显示时长。")
        for a, b, duration in zip(self.pts, self.pts[1:], self.durations):
            if duration and b-a-duration > max(duration, int(1/self.clock)):
                raise ValueError("源视频存在无法解释的PTS跳变，未生成可剪辑索引。")

    def time(self, number):
        if type(number) is not int or not 0 <= number <= len(self.pts):
            raise ValueError("源帧序号越界。")
        pts = self.end_pts if number == len(self.pts) else self.pts[number]
        return float((pts-self.pts[0])*self.clock)

    def nearest(self, seconds, start=0, end=None):
        """Nearest legal frame/boundary, with earlier display order breaking ties."""
        end = self.count if end is None else end
        if not math.isfinite(seconds) or not 0 <= start < end <= self.count+1:
            raise ValueError("取帧时间或帧区间无效。")
        target = Fraction(str(seconds))/self.clock + self.pts[0]
        pos = bisect_left(self.pts, target, min(start, self.count), min(end, self.count))
        choices = {max(start, min(end-1, pos)), max(start, min(end-1, pos-1))}
        return min(choices, key=lambda n: (abs((self.end_pts if n == self.count else self.pts[n])-target), n))

    def frame(self, number):
        if type(number) is not int or not 0 <= number < self.count:
            raise ValueError("帧号须指向实际画面，不能使用范围排他终点。")
        return {"id": media_id("F", FRAME_PROTOCOL, self.data["fingerprint"], self.data["stream_index"], number),
                "frame_index": number, "time": self.time(number),
                "source_clock": {"pts": self.pts[number], "time_base": self.data["time_base"]}}

    def binding(self):
        return {"index_protocol": FRAME_PROTOCOL, "index_key": self.data["key"],
                "stream_index": self.data["stream_index"], "time_base": self.data["time_base"],
                "origin_pts": self.pts[0], "source_frame_count": self.count, "source_duration": self.duration}


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
    stream_index: int


@dataclass(frozen=True)
class ImportMapping:
    video_path: str
    subtitle_path: str = ""
    episode: int = 1

