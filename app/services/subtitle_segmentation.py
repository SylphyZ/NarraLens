"""Word-aware Chinese subtitle segmentation without a model API call."""

import logging
import math
import re
import unicodedata
from bisect import bisect_left, bisect_right
from functools import lru_cache
from pathlib import Path

import jieba
from PIL import ImageFont

from app.services.library.runner import check_cancel


_TITLE_RE = re.compile(r"(《[^》]+》|〈[^〉]+〉)")
_WEAK_ENDINGS = {"的", "地", "得", "把", "被", "在", "向", "与", "和", "而", "因为", "虽然", "如果", "但", "以及"}
_DEPENDENT_STARTS = {"的", "地", "得", "了", "着", "过", "吗", "呢", "吧"}
_LAYOUT_WEAK_ENDINGS = _WEAK_ENDINGS | {
    "用", "来", "比", "为", "从", "以", "将", "让", "对", "这件", "那件", "一个",
    "不再", "没有", "不会", "并不", "不能", "并未", "不在于", "真正", "逐渐", "始终",
    "最终", "究竟", "却", "仍然", "只是", "主要", "甚至", "完全", "更", "最", "可能", "已经",
}
_SENTENCE_END = re.compile(r"[。！？.!?][”’」』】》）\)\]\}]*$")
_CLAUSE_END = re.compile(r"[，；：、,;:…][”’」』】》）\)\]\}]*$")
_QUANTIFIER_END = re.compile(r"^(?:第[一二三四五六七八九十百\d]+|[一二三四五六七八九十百两\d]+)(?:次|个|位|套|件|种|部|段|条|份|场|章|集)$")


@lru_cache(maxsize=1)
def _tokenizer():
    tokenizer = jieba.Tokenizer()
    cache_dir = Path(__file__).resolve().parents[2] / "storage" / "cache" / "jieba"
    cache_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.tmp_dir = str(cache_dir)
    jieba.setLogLevel(logging.WARNING)
    tokenizer.initialize()
    return tokenizer


def _visible_length(text: str) -> int:
    return sum(not char.isspace() and not unicodedata.category(char).startswith("P") for char in text)


def word_boundary_parts(text: str) -> list[str]:
    """Keep contiguous words/titles with trailing punctuation and whitespace."""
    ends, offset = [], 0
    for piece in _TITLE_RE.split(text):
        tokens = [piece] if _TITLE_RE.fullmatch(piece) else _tokenizer().cut(piece)
        for token in tokens:
            offset += len(token)
            if re.search(r"\w", token):
                trailing = re.match(r"[\s，。！？；：、,.!?;:…）】》〉」』”’\)\]\}]*", text[offset:])
                ends.append(offset + trailing.end())
    ends = sorted(set(ends + [len(text)]))
    parts, start = [], 0
    for end in ends:
        fragment = text[start:end]
        if re.search(r"\w", fragment):
            parts.append(fragment)
            start = end
    if parts and start < len(text):
        parts[-1] += text[start:]
    return parts


def split_by_words(text: str, target_chars: int, *, max_chars: int | None = None) -> list[str]:
    """Balance cues at word boundaries; max_chars counts all non-whitespace chars."""
    if max_chars is not None:
        tokens = word_boundary_parts(text)
    else:
        tokens = []
        for piece in _TITLE_RE.split(text):
            if not piece:
                continue
            if _TITLE_RE.fullmatch(piece):
                tokens.append(piece)
            else:
                tokens.extend(_tokenizer().cut(piece))
    if not tokens:
        return []
    target = max(1, int(target_chars))
    limit = max(1, int(max_chars)) if max_chars is not None else target + max(2, target // 3)
    target = min(target, limit)
    count = len(tokens)
    costs = [float("inf")] * (count + 1)
    next_boundary = [count] * (count + 1)
    costs[count] = 0.0
    for start in range(count - 1, -1, -1):
        content = ""
        for end in range(start + 1, count + 1):
            content += tokens[end - 1]
            length = sum(not char.isspace() for char in content) if max_chars is not None else _visible_length(content)
            if length > limit and end > start + 1:
                break
            if not content.strip():
                continue
            cost = ((length - target) / target) ** 2
            if length < min(4, target):
                cost += 8.0
            if end < count:
                if max_chars is not None and not re.search(r"[，。！？；：、,.!?;:…][”’」』】》）\)\]\}]*\s*$", tokens[end - 1]):
                    cost += 0.5
                if tokens[end - 1].strip() in _WEAK_ENDINGS:
                    cost += 3.0
                if tokens[end].strip() in _DEPENDENT_STARTS:
                    cost += 3.0
            total = cost + costs[end]
            if total < costs[start]:
                costs[start] = total
                next_boundary[start] = end
    chunks = []
    start = 0
    while start < count:
        end = next_boundary[start]
        chunk = "".join(tokens[start:end]).strip()
        if chunk:
            chunks.append(chunk)
        start = end
    return chunks


def narration_weight(text: str) -> float:
    """Approximate duration weight; this is not forced audio alignment."""
    chinese = len(re.findall(r"[\u3400-\u4dbf\u4e00-\u9fff]", text))
    words = len(re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*", text))
    return max(1.0, chinese + words * 2.0)


class SubtitleLayout:
    """One font/width implementation for timed cue planning and single-line drawing."""

    def __init__(self, font_path, font_size, max_width, minimum_seconds=1.0):
        self.font_path, self.font_size = font_path, int(font_size)
        self.max_width, self.minimum_seconds = float(max_width) - 6, float(minimum_seconds)
        if self.font_size < 1 or not math.isfinite(self.max_width) or self.max_width <= 0:
            raise ValueError("字幕字号或单行像素宽度无效。")
        if not math.isfinite(self.minimum_seconds) or self.minimum_seconds <= 0:
            raise ValueError("字幕阅读目标必须是大于0的有效秒数。")
        self._fonts, self._widths = {}, {}
        self._font(self.font_size)

    def _font(self, size):
        if size not in self._fonts:
            self._fonts[size] = ImageFont.truetype(self.font_path, size)
        return self._fonts[size]

    def width(self, text, size=None):
        key = (text, size or self.font_size)
        if key not in self._widths:
            font = self._font(key[1])
            box = font.getbbox(text)
            self._widths[key] = max(font.getlength(text), box[2] - box[0])
        return self._widths[key]

    def fit(self, text):
        text = " ".join(text.split())
        size, width = self.font_size, self.width(text)
        if width > self.max_width:
            size = max(1, int(size * self.max_width / width))
            while size > 1 and self.width(text, size) > self.max_width:
                size -= 1
        if self.width(text, size) > self.max_width:
            raise ValueError("不可拆词缩为1号字仍超宽，请增加字幕行宽。")
        return text, size

    @staticmethod
    def _spoken_chars(text):
        return sum(unicodedata.category(char)[0] in {"L", "N"} for char in text)

    @staticmethod
    def _groups(text, text_start, words):
        starts, ends = [word["text_start"] for word in words], [word["text_end"] for word in words]
        parts = word_boundary_parts(text)
        if "".join(parts) != text:
            raise ValueError("字幕词边界没有完整覆盖原文。")
        groups, offset, pending = [], text_start, ""
        for raw in parts:
            end = offset + len(raw)
            first, after = bisect_right(ends, offset), bisect_left(starts, end)
            if first == after:
                if SubtitleLayout._spoken_chars(raw):
                    raise ValueError(f"字幕文字没有对应字词时间：{raw.strip()!r}")
                if groups:
                    groups[-1].update(b=end, raw=groups[-1]["raw"] + raw)
                else:
                    pending += raw
                offset = end
                continue
            group = {"a": offset - len(pending), "b": end, "raw": pending + raw, "first": first, "last": after - 1,
                     "start": words[first]["start_time"], "end": words[after - 1]["end_time"]}
            if groups and groups[-1]["last"] == first:
                previous = groups[-1]
                previous.update(b=end, raw=previous["raw"] + group["raw"], last=group["last"], end=group["end"])
            else:
                groups.append(group)
            pending, offset = "", end
        if not groups:
            raise ValueError("字幕段落没有可匹配的发声字词。")
        return groups

    @staticmethod
    def _cut_cost(groups, after, semantic_ends):
        if after == len(groups):
            return 0.0
        raw = groups[after - 1]["raw"].strip()
        boundary = groups[after - 1]["b"] - (len(groups[after - 1]["raw"]) - len(groups[after - 1]["raw"].rstrip()))
        cost = 0.0 if _SENTENCE_END.search(raw) else .2 if _CLAUSE_END.search(raw) else .3 if boundary in semantic_ends else 1.0
        if raw in _LAYOUT_WEAK_ENDINGS or _QUANTIFIER_END.fullmatch(raw):
            cost += 3.0
        if groups[after]["raw"].strip() in _DEPENDENT_STARTS:
            cost += 3.0
        return cost

    def _solve(self, groups, semantic_ends, strict, stop):
        count = len(groups)
        costs, next_end = [math.inf] * (count + 1), [None] * count
        costs[count] = 0.0
        for begin in range(count - 1, -1, -1):
            check_cancel(stop)
            raw = ""
            for after in range(begin + 1, count + 1):
                raw += groups[after - 1]["raw"]
                text = " ".join(raw.split())
                width = self.width(text)
                if width > self.max_width + .01 and after > begin + 1:
                    break
                duration = groups[after - 1]["end"] - groups[begin]["start"]
                if duration <= 0 or not math.isfinite(costs[after]):
                    continue
                chars = self._spoken_chars(text)
                short_time, short_text = duration < self.minimum_seconds - .001, chars < 4
                if strict and (short_time or short_text):
                    continue
                _, size = self.fit(text)
                fill = self.width(text, size) / self.max_width
                cost = .25 + 2.0 * (fill - .82) ** 2 + .1 * ((duration - 2.2) / 2.2) ** 2
                cost += self._cut_cost(groups, after, semantic_ends)
                if short_time:
                    cost += 100 + 80 * (1 - duration / self.minimum_seconds) ** 2
                if short_text:
                    cost += 20 + 30 * (1 - chars / 4) ** 2
                if duration > 5:
                    cost += 2 * (duration - 5) ** 2
                if cost + costs[after] < costs[begin]:
                    costs[begin], next_end[begin] = cost + costs[after], after
        if not math.isfinite(costs[0]):
            return None
        result, begin = [], 0
        while begin < count:
            after = next_end[begin]
            text, size = self.fit("".join(group["raw"] for group in groups[begin:after]))
            duration = groups[after - 1]["end"] - groups[begin]["start"]
            result.append({"text_start": groups[begin]["a"], "text_end": groups[after - 1]["b"],
                           "font_size": size, "pixel_width": round(self.width(text, size), 2),
                           "short": duration < self.minimum_seconds - .001 or self._spoken_chars(text) < 4})
            begin = after
        return result

    def plan(self, text, text_start, semantic_ranges, words, stop=None):
        groups = self._groups(text, text_start, words)
        semantic_ends = {row["text_end"] for row in semantic_ranges}
        result, region_start = [], 0
        for after, group in enumerate(groups, 1):
            check_cancel(stop)
            end = group["b"] - (len(group["raw"]) - len(group["raw"].rstrip()))
            paragraph_end = bool(re.search(r"\n\s*\n", group["raw"]))
            if after != len(groups) and not paragraph_end and end not in semantic_ends and not _SENTENCE_END.search(group["raw"].strip()):
                continue
            region = groups[region_start:after]
            if after != len(groups) and not paragraph_end and (
                region[-1]["end"] - region[0]["start"] < self.minimum_seconds - .001
                or self._spoken_chars("".join(item["raw"] for item in region)) < 4
            ):
                continue
            rows = self._solve(region, semantic_ends, True, stop)
            if rows is None:
                rows = self._solve(region, semantic_ends, False, stop)
            if rows is None:
                raise ValueError("字幕区域没有有效发声时长，请检查已保存的字词对齐结果。")
            result.extend(rows)
            region_start = after
        reconstructed = "".join(text[row["text_start"] - text_start:row["text_end"] - text_start] for row in result)
        if "".join(reconstructed.split()) != "".join(text.split()):
            raise ValueError("字幕动态排版没有完整保留原文。")
        return result
