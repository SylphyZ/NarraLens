"""Select Chinese dialogue before ASS/SSA styling and layout are discarded."""

import re
from bisect import bisect_left
from dataclasses import dataclass


_HAN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\U00020000-\U0002fa1f]")
_OTHER_SCRIPT = re.compile(r"[\u3040-\u30ff\u31f0-\u31ff\uff66-\uff9f\uac00-\ud7af\u1100-\u11ff\u0400-\u052f\u0600-\u06ff\u0e00-\u0e7f]")
_LATIN = re.compile(r"[A-Za-z\u00c0-\u024f]")
_CHINESE_STYLE = re.compile(r"(?:^|[^a-z])(?:chs|cht|chn|cn|zh|zho|chi|chinese|sc|tc)(?:$|[^a-z])", re.IGNORECASE)
_DEFAULT_FIELDS = ("layer", "start", "end", "style", "name", "marginl", "marginr", "marginv", "effect", "text")


@dataclass(frozen=True)
class AssEvent:
    line_index: int
    fields: tuple[str, ...]
    text_index: int
    style: str
    start: str
    end: str
    text: str


@dataclass(frozen=True)
class ChineseAssResult:
    text: str
    original: tuple[AssEvent, ...]
    selected: tuple[AssEvent, ...]
    removed_events: int
    removed_lines: int
    duplicate_events: int
    uncertain_events: int


def plain_ass_lines(text: str) -> list[str]:
    """Remove override tags and vector drawings, keeping explicit line breaks."""
    parts = []
    drawing = False
    for part in re.split(r"(\{[^}]*\})", text):
        if part.startswith("{") and part.endswith("}"):
            mode = re.search(r"\\p(\d+)\b", part)
            if mode:
                drawing = int(mode.group(1)) != 0
        elif not drawing:
            parts.append(part)
    visible = "".join(parts).replace(r"\N", "\n").replace(r"\n", "\n").replace(r"\h", " ")
    return [line.strip() for line in visible.splitlines() if line.strip()]


def parse_ass_events(text: str) -> list[AssEvent]:
    events = []
    in_events = False
    columns = _DEFAULT_FIELDS
    for line_index, line in enumerate(text.splitlines()):
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            in_events = stripped.casefold() == "[events]"
            continue
        if not in_events or ":" not in stripped:
            continue
        kind, body = stripped.split(":", 1)
        if kind.casefold() == "format":
            columns = tuple(field.strip().casefold() for field in body.split(","))
            continue
        if kind.casefold() != "dialogue":
            continue
        if not {"start", "end", "style", "text"}.issubset(columns):
            raise ValueError("ASS/SSA Events Format must contain Start, End, Style and Text.")
        text_index = columns.index("text")
        before = body.lstrip().split(",", text_index)
        after_count = len(columns) - text_index - 1
        after = before[-1].rsplit(",", after_count) if after_count else [before[-1]]
        fields = tuple(before[:-1] + after)
        if len(fields) != len(columns):
            raise ValueError(f"Invalid ASS/SSA dialogue at line {line_index + 1}.")
        events.append(AssEvent(
            line_index, fields, text_index, fields[columns.index("style")].strip(),
            fields[columns.index("start")].strip(), fields[columns.index("end")].strip(), fields[text_index],
        ))
    if not events:
        raise ValueError("ASS/SSA subtitle has no dialogue events.")
    return events


def is_chinese_style(style: str) -> bool:
    return bool(_CHINESE_STYLE.search(style) or any(word in style for word in ("中文", "简体", "繁体", "简中", "繁中", "簡體", "繁體", "簡中")))


def _chinese_candidate(line: str) -> bool:
    # Han characters alone do not prove a language; explicit styles take priority.
    return bool(_HAN.search(line) and not _OTHER_SCRIPT.search(line))


def _time_ms(value: str) -> int:
    hours, minutes, seconds = value.split(":")
    return round((int(hours) * 3600 + int(minutes) * 60 + float(seconds)) * 1000)


def _parallel(left: AssEvent, right: AssEvent) -> bool:
    start, end = _time_ms(left.start), _time_ms(left.end)
    other_start, other_end = _time_ms(right.start), _time_ms(right.end)
    shortest = min(end - start, other_end - other_start)
    overlap = min(end, other_end) - max(start, other_start)
    return shortest > 0 and overlap / shortest >= 0.7


def select_chinese_ass(text: str, styles: list[str] | None = None, line_mode: str = "auto") -> ChineseAssResult:
    """Use explicit styles/line selection, otherwise conservatively remove parallel translations."""
    if line_mode not in {"auto", "first", "last"}:
        raise ValueError("Unsupported Chinese subtitle line selection.")
    events = parse_ass_events(text)
    available = {event.style for event in events}
    chosen = set(styles or [])
    if chosen and not chosen.intersection(available):
        raise ValueError("Selected Chinese styles do not exist in this subtitle file.")
    if not chosen:
        chosen = {style for style in available if is_chinese_style(style)}
    candidates = [event for event in events if any(_chinese_candidate(line) for line in plain_ass_lines(event.text))]
    candidates.sort(key=lambda event: _time_ms(event.start))
    candidate_starts = [_time_ms(event.start) for event in candidates]
    max_candidate_duration = max((_time_ms(event.end) - _time_ms(event.start) for event in candidates), default=0)
    source_lines = text.splitlines()
    replacements = {}
    selected = []
    removed_events = removed_lines = duplicates = uncertain = 0
    seen = set()

    for event in events:
        lines = plain_ass_lines(event.text)
        if (chosen and event.style not in chosen) or not lines:
            replacements[event.line_index] = None
            removed_events += 1
            continue
        chinese = [line for line in lines if _chinese_candidate(line)]
        if not chosen and not chinese and any(_OTHER_SCRIPT.search(line) or _LATIN.search(line) for line in lines):
            low = bisect_left(candidate_starts, _time_ms(event.start) - max_candidate_duration)
            high = bisect_left(candidate_starts, _time_ms(event.end))
            if any(_parallel(event, candidate) for candidate in candidates[low:high]):
                replacements[event.line_index] = None
                removed_events += 1
                continue
            uncertain += 1  # Standalone foreign text may be a name or a line with no Chinese translation.
        kept = list(lines)
        if line_mode != "auto":
            kept = [lines[0] if line_mode == "first" else lines[-1]]
        else:
            if chinese:
                # A separate foreign-script/Latin line beside Chinese is a translation candidate.
                kept = [line for line in lines if _chinese_candidate(line) or not (_OTHER_SCRIPT.search(line) or _LATIN.search(line))]
                if not chosen:
                    uncertain += 1  # Japanese kanji-only text cannot be reliably distinguished.
        removed_lines += len(lines) - len(kept)
        kept = list(dict.fromkeys(kept))
        visible = r"\N".join(kept)
        identity = (_time_ms(event.start), _time_ms(event.end), visible)
        if identity in seen:
            replacements[event.line_index] = None
            duplicates += 1
            continue
        seen.add(identity)
        fields = list(event.fields)
        fields[event.text_index] = visible
        replacements[event.line_index] = "Dialogue: " + ",".join(fields)
        selected.append(AssEvent(event.line_index, tuple(fields), event.text_index, event.style, event.start, event.end, visible))

    if not selected:
        raise ValueError("No subtitles remain after Chinese selection; check the selected styles and lines.")
    filtered = "\n".join(
        replacements.get(index, line) for index, line in enumerate(source_lines)
        if replacements.get(index, line) is not None
    )
    return ChineseAssResult(filtered, tuple(events), tuple(selected), removed_events, removed_lines, duplicates, uncertain)
