"""External editors own visual decisions; local tools own source and timeline validity."""

import argparse
import hashlib
import hmac
import json
import os
import secrets
import sys
import time
import wave
from bisect import bisect_right
from contextlib import contextmanager
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path
from uuid import UUID, uuid4

from app.models.library import SOURCE_INDEX_FIELDS
from app.models.timeline import NarrationSegment, VisualShot, timeline_dict
from app.services.library import runner
from app.services.library.store import LibraryStore, library_ids, read_json, stable_hash, write_json

PROJECT = Path(__file__).resolve().parents[2]
TASK_ROOT = PROJECT / "storage" / "narration_tasks"
PROTOCOL = "agent-project-visual-v1"
FPS = 30
LEVELS = {"exact", "related", "atmosphere"}
ASSET_FIELDS = ("id", "library_id", "path", "episode", "size", "mtime_ns", "fingerprint",
                "duration", "fps", "subtitle_identity")


def visual_backend(options):
    value = options.get("visual_backend", "internal")
    if value not in {"internal", "agent"}:
        raise ValueError("画面匹配方式应为内置API或外部Agent。")
    return value


def _owned(root, relative=""):
    root, relative = Path(root), Path(relative)
    path = root / relative
    if relative.is_absolute() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("画面任务路径越界。")
    for entry in (path, *path.parents):
        if entry.is_symlink() or entry.is_junction():
            raise ValueError("画面任务目录包含链接，拒绝访问。")
        if entry == PROJECT:
            break
    return path


def _project(identifier):
    return _owned(TASK_ROOT, str(UUID(str(identifier))))


@contextmanager
def project_lock(identifier):
    """Cross-process project lock, also used by run, delete and configuration commits."""
    path = _owned(TASK_ROOT, ".locks/" + str(UUID(str(identifier))) + ".visual.lock")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as stream:
        stream.seek(0, 2)
        if not stream.tell():
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise ValueError("项目正在生成或提交画面方案，请结束后重试。") from exc
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)


@lru_cache(maxsize=64)
def _digest_file(path, size, modified):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    stat = Path(path).stat()
    if (stat.st_size, stat.st_mtime_ns) != (size, modified):
        raise ValueError("读取期间文件改变，请稍后重试。")
    return digest.hexdigest()


def _identity(path):
    stat = Path(path).stat()
    return _digest_file(str(path), stat.st_size, stat.st_mtime_ns)


def _secret(directory):
    return read_json(_owned(directory, "agent_visual/registry.json"))["secret"]


def _sign(value, key):
    return hmac.new(bytes.fromhex(key), stable_hash(value).encode("ascii"), hashlib.sha256).hexdigest()


def _save(path, value, key):
    write_json(path, {"value": value, "signature": _sign(value, key)})


def _signed(path, key):
    record = read_json(path)
    if not hmac.compare_digest(record.get("signature", ""), _sign(record["value"], key)):
        raise ValueError("画面任务记录已改变，请通过提交工具保存，不能手改回执。")
    return record["value"]


def _motion(directory, settings, text, segments):
    """Reconstruct existing reservations without rendering or probing any media."""
    from app.services.narration_motion import load_motion_assets, motion_config_key, motion_options
    options = motion_options(settings["options"])
    binding = stable_hash({k: v for k, v in options.items() if k.startswith("motion_") or
                           k in {"aspect", "font", "font_size", "subtitles"}})
    if not options["motion_enabled"]:
        return {"binding": binding, "plan": None, "assets": [], "reservations": []}
    plan = read_json(_owned(directory, "motion/plan.json"), default={})
    manifest = read_json(_owned(directory, "motion/manifest.json"), default={})
    if (not plan or plan.get("text_hash") != stable_hash(text) or
            abs(plan.get("duration", 0) - segments[-1].audio_end) > .01 or
            plan.get("manual_config_key") != motion_config_key(options) or
            manifest.get("plan_key") != plan.get("key")):
        raise ValueError("解释动画尚未准备完成，请先从3. 寻找画面继续；本按钮不调用动画模型。")
    reservations = []
    for row in plan["animations"]:
        if row["id"] in options["motion_disabled_ids"]:
            continue
        spans = row.get("spans") or [{"segment_id": row["segment_id"], "start": row["start"], "end": row["end"],
                                     "source_start": 0, "source_end": row["end"] - row["start"]}]
        asset = next((a for a in manifest.get("assets", []) if a["id"] == row["id"]), None)
        if not asset:
            raise ValueError("固定动画缺少已渲染素材，请从3. 寻找画面继续准备。")
        for span in spans:
            reservations.append(VisualShot(asset["id"], asset["id"], asset["path"], span["source_start"], span["source_end"],
                span["start"], span["end"], span["segment_id"], "解释动画：" + row["reason"]))
    assets = load_motion_assets(directory, reservations, plan_key=plan["key"])
    return {"binding": binding, "plan": {k: plan[k] for k in
            ("key", "timing_method", "manual_config_key", "placement") if k in plan},
            "assets": [{**a, "file_hash": _identity(_owned(directory, Path(a["path"]).relative_to(directory)))} for a in assets],
            "reservations": [asdict(s) for s in reservations]}


def _snapshot(directory):
    from app.services.narration_task import _resume_checkpoint
    settings = read_json(_owned(directory, "settings.json"))
    if visual_backend(settings["options"]) != "agent":
        raise ValueError("请先保存画面匹配方式为外部Agent。")
    with _owned(directory, "input.txt").open("r", encoding="utf-8", newline="") as source:
        text = source.read()
    checkpoint = _resume_checkpoint(directory, settings, text, 3)
    audio = _owned(directory, "narration.wav")
    with wave.open(str(audio), "rb") as source:
        rate, samples = source.getframerate(), source.getnframes()
    total_frames = round(samples * FPS / rate)
    if total_frames < 1:
        raise ValueError("完整配音不足一帧。")
    return {"text": text, "text_hash": stable_hash(text), "overview": checkpoint["overview"], "library_ids": library_ids(settings),
            "segments": [asdict(s) for s in checkpoint["segments"]], "audio_manifest": checkpoint["audio"],
            "audio": {"sha256": _identity(audio), "rate": rate, "samples": samples},
            "fps": FPS, "total_frames": total_frames, "target_shot_seconds": settings["options"].get("shot_seconds", 7),
            "motion": _motion(directory, settings, text, checkpoint["segments"])}


def _intervals(snapshot):
    reservations = sorted((round(s["timeline_start"] * FPS), round(s["timeline_end"] * FPS))
                          for s in snapshot["motion"]["reservations"])
    result, position = [], 0
    for start, end in reservations:
        if start < position or end <= start or end > snapshot["total_frames"]:
            raise ValueError("固定动画时间范围无效，请先完成动画准备。")
        if start > position:
            result.append([position, start])
        position = end
    if position < snapshot["total_frames"]:
        result.append([position, snapshot["total_frames"]])
    return result


def list_tasks(project_id):
    directory = _project(project_id)
    result = []
    registry = _owned(directory, "agent_visual/registry.json")
    if not registry.is_file():
        return result
    key = _secret(directory)
    for path in _owned(directory, "agent_visual").glob("*/manifest.json"):
        manifest = _signed(_owned(path.parent, "manifest.json"), key)
        head = _signed(_owned(path.parent, "head.json"), key)
        result.append({**manifest, "head": head, "directory": str(path.parent)})
    return sorted(result, key=lambda row: row["created"], reverse=True)


def _load(project_id, task_id):
    directory = _project(project_id)
    root = _owned(directory, "agent_visual/" + str(UUID(str(task_id))))
    key = _secret(directory)
    manifest = _signed(_owned(root, "manifest.json"), key)
    snapshot = read_json(_owned(root, "inputs/project.json"))
    catalogue = read_json(_owned(root, "inputs/catalogue.json"))
    if (manifest["protocol"] != PROTOCOL or manifest["project_id"] != directory.name or manifest["task_id"] != root.name or
            manifest["input_hash"] != stable_hash([snapshot, catalogue])):
        raise ValueError("画面任务输入与固定快照不一致，请重新创建任务。")
    return directory, root, key, manifest, snapshot, catalogue


def create_task(project_id, *, locked=False):
    if not locked:
        with project_lock(project_id), runner.library_operation(project_id, prefix="narration:"):
            return create_task(project_id, locked=True)
    directory = _project(project_id)
    snapshot = _snapshot(directory)
    with runner.library_operation(snapshot["library_ids"]):
        catalogue = LibraryStore.catalogue_snapshot(snapshot["library_ids"])
    catalogue["assets"] = [{k: a.get(k) for k in ASSET_FIELDS} for a in catalogue["assets"]]
    signature = stable_hash([snapshot, catalogue])
    for task in list_tasks(project_id):
        if task["input_hash"] == signature:
            return task
    registry = _owned(directory, "agent_visual/registry.json")
    if not registry.exists():
        write_json(registry, {"secret": secrets.token_hex(32)})
    key, identifier = _secret(directory), str(uuid4())
    root = _owned(directory, "agent_visual/" + identifier)
    for folder in ("inputs", "proposals", "results", "receipts", "previews"):
        _owned(root, folder).mkdir(parents=True, exist_ok=True)
    write_json(_owned(root, "inputs/project.json"), snapshot)
    write_json(_owned(root, "inputs/catalogue.json"), catalogue, compact=True)
    manifest = {"protocol": PROTOCOL, "project_id": directory.name, "task_id": identifier,
                "input_hash": signature, "created": time.time(), "fps": FPS, "total_frames": snapshot["total_frames"],
                "original_intervals": _intervals(snapshot), "library_ids": snapshot["library_ids"]}
    _save(_owned(root, "manifest.json"), manifest, key)
    _save(_owned(root, "head.json"), {"revision": 0, "complete": False, "clips": 0, "covered_frames": 0,
                                    "levels": {}, "workset_revisions": {}, "updated": manifest["created"]}, key)
    _owned(root, "TASK.md").write_text(instructions(manifest, root), encoding="utf-8")
    previous = _owned(directory, "timeline.json")
    if previous.is_file():
        value = read_json(previous)
        write_json(_owned(root, "inputs/previous-visual.json"),
                   {"note": "仅供参考，可重新组织整篇画面", "shots": value.get("shots", [])})
    return {**manifest, "directory": str(root), "head": _signed(_owned(root, "head.json"), key)}


def instructions(manifest, root):
    executable = str(Path(sys.executable).resolve())
    command = f'& "{executable}" -X utf8 -m app.services.narration_agent --project {manifest["project_id"]} --task {manifest["task_id"]}'
    return f'''# 整篇文案的画面编辑任务

在代码目录 `{PROJECT}` 中用 PowerShell 运行下列命令。使用当前客户端模型，不读取私有config.toml或API Key。

任务：{manifest["task_id"]}；协议：{PROTOCOL}；input_hash：{manifest["input_hash"]}。

你负责全文理解、视觉分段、全库检索、看图、选镜头、源帧范围、顺序、时长和全片节奏。默认覆盖全部原片区间，不仅补旧缺口。
原文/配音/字幕与已渲染动画占位固定。允许exact准确对应、related主题/人物/关系配图、atmosphere氛围与过渡；三种均是成功结果。
以整段论述表达为核心，可跨句跨段保持一个镜头。具体人物事件尽量准确，避免误导；抽象观点无需逐句直接剧情证明。
不要重复源帧或重复自然镜头，避开OP/ED名单/黑场、几帧短切，通常按参考镜头长度编排；不循环、不改语速、不长时间定格补时。
实际查看镜头图片，身份不确定时如实说明，不以自报信心代替看图。工具只检查执行结构，不替你判断语义。

## 查询与按需取证

```powershell
Set-Location -LiteralPath "{PROJECT}"
{command} describe --full
{command} search --query "家庭 交谈" --kind ranges --offset 0 --limit 20
{command} search --query "" --kind contents --episode 1 --offset 0 --limit 20
{command} view --library <library_id> --range <range_id>
{command} view --library <library_id> --range <range_id> --capture
{command} fit --library <library_id> --range <range_id> --start-frame 0 --target-frames 210
{command} occupancy
```

search是本地分页工具，可换查询、按集浏览所有已选库，没有固定候选上限。view默认仅返回观察、字幕、目录和已有图路径。
只有显式--capture才会提取该范围的3张图片；也可加--frames指定实际呈现帧序号（逗号分隔，最多12）。取证后打开返回的图片。
fit的start-frame必须使用view返回的源帧范围，按真实PTS计算可用终点；若不够长，换范围或调整目标。源/输出帧均为半开区间。
多个连续range_ids只允许同一素材、同一自然镜头内连续范围。相邻不同自然镜头应作为不同clip，不能隐藏切点。

## 草稿、分工与完成版

完整输入在inputs/project.json；需要原片的目标帧区间见describe。输出固定30fps，总帧数{manifest["total_frames"]}，不能改。
先理解全文，再按章节分工。可用worksets生成互不重叠的连续时间分工；是否实际启动子Agent由当前客户端决定。

```powershell
{command} worksets --workers 4
{command} submit --file proposals/part.json
{command} check
```

提案JSON结构（示意数字须换成真实范围）：

```json
{{"protocol":"{PROTOCOL}","project_id":"{manifest["project_id"]}","task_id":"{manifest["task_id"]}",
"input_hash":"{manifest["input_hash"]}","expected_revision":0,"complete":false,
"sections":[{{"id":"C01","text_start":0,"text_end":100,"target_start_frame":0,"target_end_frame":210,"goal":"表达主题"}}],
"clips":[{{"clip_id":"C01-001","section_id":"C01","library_id":"登记库ID","range_ids":["登记范围ID"],
"source_start_frame":0,"source_end_frame":210,"target_start_frame":0,"target_end_frame":210,
"match_level":"related","reason":"画面怎样辅助本段论述"}}]}}
```

文件必须位于本任务proposals下，不能手改inputs、results、receipts、head、SQLite、正式timeline或源媒体。
主Agent提交用describe中最新全局expected_revision（对应head.revision），提交替换整条草稿；重复内容可幂等返回。complete=true须覆盖所有原片区间且保留动画。
全文只需首次describe --full，后续describe不带--full仅查询简要状态，避免每次重发长文案。
子Agent在提案增加workset_id，用对应workset的expected_revision，只替换本分工；其他分工会合并保留。分工提交只能是草稿。
冲突时先describe与occupancy读取已保存进度，再合并修订；不要重新跑已完成的付费分析。
主Agent必须汇总所有分工，核对跨章节重复与节奏，然后提交完整sections/clips并complete=true，check通过后交回WebUI导入。
check失败按具体clip/范围修正；related/atmosphere不会因不够精确而被当成缺口。
工具提交不等于正式导入。用户在WebUI点击“检查并导入完成版”，随后从第4/5阶段推进。旧成片保留。
'''


def describe(project_id, task_id, *, full=False):
    _, root, key, manifest, snapshot, catalogue = _load(project_id, task_id)
    return {**manifest, "directory": str(root), "head": _signed(_owned(root, "head.json"), key),
            **({"text": snapshot["text"], "overview": snapshot["overview"], "segments": snapshot["segments"],
                "cues": snapshot["audio_manifest"]["cues"], "audio": snapshot["audio"],
                "motion": snapshot["motion"], "assets": catalogue["assets"]} if full else {}),
            "target_shot_seconds": snapshot["target_shot_seconds"], "libraries": catalogue["libraries"],
            "range_count": len(catalogue["ranges"]),
            "content_count": len(catalogue["contents"]), "worksets": (
                _signed(_owned(root, "worksets-record.json"), key) if _owned(root, "worksets-record.json").is_file() else [])}


def search(project_id, task_id, query="", *, kind="ranges", episode=None, library=None, offset=0, limit=20):
    _, _, _, _, _, catalogue = _load(project_id, task_id)
    if kind not in {"ranges", "contents"} or type(offset) is not int or offset < 0 or not 1 <= limit <= 100:
        raise ValueError("查询类型或分页无效，limit应为1至100。")
    import jieba
    terms = set(t.lower() for t in jieba.lcut(query) if t.strip())
    for entry in catalogue["libraries"]:
        for person in entry.get("work", {}).get("characters", []):
            values = [person.get("name", ""), *person.get("aliases", [])]
            if any(value and value.lower() in query.lower() for value in values):
                terms.update(value.lower() for value in values if value)
    rows = []
    for row in catalogue[kind]:
        if (episode is not None and row["episode"] != episode) or (library and row["library_id"] != library):
            continue
        if row.get("snapshot_stale"):
            continue
        body = json.dumps(row, ensure_ascii=False).lower()
        score = sum(body.count(term) for term in terms)
        if terms and not score:
            continue
        rows.append((score, row))
    rows.sort(key=lambda value: (-value[0], value[1]["episode"], value[1].get("start", 0), value[1]["id"]))
    fields = ("id", "library_id", "asset_id", "episode", "start", "end", "start_frame", "end_frame", "index_key",
              "summary", "title", "visible_people", "mentioned_people", "tags", "period", "location", "uncertainties",
              "content_kind", "analysis_state", "range_ids", "duplicate_group")
    return {"items": [{k: row[k] for k in fields if k in row} for _, row in rows[offset:offset+limit]], "offset": offset,
            "next_offset": offset + min(limit, max(0, len(rows)-offset)), "total": len(rows),
            "exhausted": offset+limit >= len(rows)}


def _range(catalogue, library, identifier):
    item = next((r for r in catalogue["ranges"] if r["id"] == identifier and r["library_id"] == library), None)
    if not item or item.get("snapshot_stale"):
        raise ValueError(f"范围{identifier}未登记在本任务的有效快照中。")
    asset = next(a for a in catalogue["assets"] if a["id"] == item["asset_id"] and a["library_id"] == library)
    return item, asset


def _index(asset, item):
    from app.services.library.frame_extractor import _read_source_index
    path = _owned(PROJECT / "storage" / "libraries", asset["library_id"] + "/frame_index/" + asset["fingerprint"] + ".json")
    stat = path.stat()
    index = _read_source_index(str(path), stat.st_mtime_ns, stat.st_size)
    if index.data["fingerprint"] != asset["fingerprint"] or any(item.get(k) != v for k, v in index.binding().items()):
        raise ValueError(f"范围{item['id']}的源帧索引已改变。")
    return index


def _source_current(asset):
    from app.services.library.frame_extractor import sampled_fingerprint
    path = Path(asset["path"])
    stat = path.stat()
    if stat.st_size != asset["size"] or stat.st_mtime_ns != asset["mtime_ns"] or sampled_fingerprint(path) != asset["fingerprint"]:
        raise ValueError(f"第{asset['episode']}集源视频已改变，请重新登记并创建任务。")


def view(project_id, task_id, library, range_id, *, capture=False, frames=None):
    directory, root, _, _, _, catalogue = _load(project_id, task_id)
    item, asset = _range(catalogue, library, range_id)
    result = {"range": item, "asset": asset, "contents": [c for c in catalogue["contents"]
              if c["library_id"] == library and range_id in c.get("range_ids", [])]}
    images = {}
    for frame in [*item.get("frames", []), *item.get("evidence_images", [])]:
        if not frame.get("image") or frame.get("crop") is not None:
            continue
        path = _owned(PROJECT / "storage" / "libraries" / library, frame["image"])
        if path.is_file() and (not frame.get("sha256") or _identity(path) == frame["sha256"]):
            images[frame["frame_index"]] = {**frame, "path": str(path)}
    result["images"] = list(images.values())
    if capture:
        from app.config import config
        from app.services.library.frame_extractor import extract_frames
        with project_lock(project_id), runner.library_operation(project_id, prefix="narration:"), runner.library_operation(library):
            if not directory.is_dir():
                raise ValueError("项目已删除。")
            _source_current(asset)
            index = _index(asset, item)
            start, end = item["start_frame"], item["end_frame"]
            numbers = frames if frames is not None else sorted({start, (start+end-1)//2, end-1})
            if not numbers or len(numbers) > 12 or any(type(n) is not int or not start <= n < end for n in numbers):
                raise ValueError("取证帧必须位于登记范围，单次最多12帧。")
            missing = [n for n in numbers if n not in images]
            if missing:
                config.apply_ffmpeg_path(config.app.get("ffmpeg_path", ""))
                images.update({f["frame_index"]: f for f in extract_frames(asset, index, missing, _owned(root, "previews/" + asset["id"]))})
            result["images"] = [images[n] for n in numbers]
            _source_current(asset)
    return result


def fit(project_id, task_id, library, range_id, start_frame, target_frames):
    *_, catalogue = _load(project_id, task_id)
    item, asset = _range(catalogue, library, range_id)
    index = _index(asset, item)
    if (type(start_frame) is not int or not item["start_frame"] <= start_frame < item["end_frame"] or
            type(target_frames) is not int or target_frames < FPS):
        raise ValueError("起点需在源范围内，目标至少30输出帧。")
    target = index.time(start_frame) + target_frames / FPS
    end = index.nearest(target, start_frame+1, item["end_frame"]+1)
    seconds = index.time(end) - index.time(start_frame)
    tolerance = max(1/FPS, float(index.clock)) + 1e-7
    if abs(seconds-target_frames/FPS) > tolerance:
        raise ValueError("当前范围容量或帧量化不适合目标时长，请换范围或调整目标帧数。")
    return {"source_start_frame": start_frame, "source_end_frame": end, "source_seconds": seconds,
            "target_frames": target_frames, "index_key": index.data["key"]}


def _revision(root, key, number=None):
    head = _signed(_owned(root, "head.json"), key)
    number = head["revision"] if number is None else number
    if type(number) is not int or number < 0:
        raise ValueError("修订号无效。")
    if not number:
        return {"sections": [], "clips": [], "complete": False}, head
    value = _signed(_owned(root, f"results/revision-{number}.json"), key)
    receipt = _signed(_owned(root, f"receipts/revision-{number}.json"), key)
    manifest = _signed(_owned(root, "manifest.json"), key)
    if (number > head["revision"] or value["revision"] != number or receipt["result_hash"] != stable_hash(value) or
            any(value.get(field) != manifest[field] for field in ("protocol", "project_id", "task_id", "input_hash")) or
            receipt.get("input_hash") != manifest["input_hash"]):
        raise ValueError("画面修订与受控提交回执不一致。")
    return value, head


def _span(row, name, maximum):
    start, end = row.get(name + "_start_frame"), row.get(name + "_end_frame")
    if type(start) is not int or type(end) is not int or not 0 <= start < end <= maximum:
        raise ValueError(f"{name}帧范围应为范围内的整数半开区间。")
    return start, end


def _compile(snapshot, catalogue, proposal, *, complete):
    from app.services.timeline_planner import validate_timeline_structure
    segments = [NarrationSegment(**s) for s in snapshot["segments"]]
    sections, clips = proposal.get("sections"), proposal.get("clips")
    if (not isinstance(sections, list) or not isinstance(clips, list) or
            any(not isinstance(row, dict) for row in [*sections, *clips])):
        raise ValueError("sections与clips应为对象列表。")
    section_map, original_shots, used, fingerprints, assets, natural_used, groups_used = {}, [], {}, {}, {}, set(), set()
    for section in sections:
        identifier = section.get("id", "")
        if not isinstance(identifier, str) or not identifier or identifier in section_map:
            raise ValueError("视觉段落ID应非空且唯一。")
        a, b = _span(section, "target", snapshot["total_frames"])
        left, right = section.get("text_start"), section.get("text_end")
        if type(left) is not int or type(right) is not int or not 0 <= left < right <= len(snapshot["text"]):
            raise ValueError(f"视觉段落{identifier}原文字符范围无效。")
        owners = [s for s in segments if s.text_end > left and s.text_start < right]
        if not owners or a < round(owners[0].audio_start*FPS) or b > round(owners[-1].audio_end*FPS):
            raise ValueError(f"视觉段落{identifier}目标时间超出所绑定的原文段落。")
        if not isinstance(section.get("goal"), str) or not section["goal"].strip():
            raise ValueError(f"视觉段落{identifier}缺少表达目标。")
        if any(min(b, _span(s, "target", snapshot["total_frames"])[1]) >
               max(a, _span(s, "target", snapshot["total_frames"])[0]) for s in section_map.values()):
            raise ValueError(f"视觉段落{identifier}目标时间重叠。")
        section_map[identifier] = section
    identifiers = set()
    for clip in clips:
        label = clip.get("clip_id", "")
        try:
            if not isinstance(label, str) or not label or label in identifiers:
                raise ValueError("clip_id应非空且唯一。")
            identifiers.add(label)
            section = section_map.get(clip.get("section_id"))
            a, b = _span(clip, "target", snapshot["total_frames"])
            if not section or not section["target_start_frame"] <= a < b <= section["target_end_frame"]:
                raise ValueError("镜头未完整位于对应视觉段落。")
            if b-a < FPS:
                raise ValueError("镜头少于1秒，请调整邻镜头；不能用几帧填洞。")
            if not any(left <= a < b <= right for left, right in _intervals(snapshot)):
                raise ValueError("镜头跨越固定动画或超出原片区间。")
            if clip.get("match_level") not in LEVELS or not isinstance(clip.get("reason"), str) or not clip["reason"].strip():
                raise ValueError("需填写exact/related/atmosphere与选择理由。")
            ranges = clip.get("range_ids")
            if not isinstance(ranges, list) or not ranges or len(ranges) != len(set(ranges)):
                raise ValueError("range_ids需包含唯一的已登记范围。")
            items = [_range(catalogue, clip.get("library_id"), rid) for rid in ranges]
            item, asset = items[0]
            index = _index(asset, item)
            if any(a2["id"] != asset["id"] or a2["library_id"] != asset["library_id"] or
                    any(row.get(k) != item.get(k) for k in SOURCE_INDEX_FIELDS) for row, a2 in items):
                raise ValueError("一个镜头只能引用同源同索引的连续范围。")
            for row, _ in items:
                if row.get("content_kind") in {"opening", "credits", "black"}:
                    raise ValueError("镜头包含已登记的片头、片尾名单或黑场。")
            groups = {row["duplicate_group"] for row, _ in items if row.get("duplicate_group")}
            if groups & groups_used:
                raise ValueError("与已登记的重复镜头组冲突，请换未用镜头。")
            groups_used.update(groups)
            if any(left[0]["end_frame"] != right[0]["start_frame"] for left, right in zip(items, items[1:])):
                raise ValueError("多个range_ids必须按源帧顺序连续。")
            start, end = _span(clip, "source", index.count)
            if not item["start_frame"] <= start < end <= items[-1][0]["end_frame"]:
                raise ValueError("源帧不在所引用范围内。")
            shot_index = bisect_right(index.cuts, start)-1
            natural = (index.cuts[shot_index], index.cuts[shot_index+1])
            if end > natural[1]:
                raise ValueError("跨越自然镜头切点，请拆成独立镜头。")
            identity = (asset["fingerprint"], index.data["stream_index"])
            natural_key = (*identity, *natural)
            if natural_key in natural_used:
                raise ValueError("重复使用同一自然镜头，请换未用镜头。")
            natural_used.add(natural_key)
            if any(min(end, old_end) > max(start, old_start) for old_start, old_end in used.get(identity, [])):
                raise ValueError("源帧与其他镜头重复。")
            used.setdefault(identity, []).append((start, end))
            for forbidden in catalogue["ranges"]:
                if (forbidden.get("fingerprint") == asset["fingerprint"] and forbidden.get("index_key") == item["index_key"] and
                        forbidden.get("content_kind") in {"opening", "credits", "black"} and
                        min(end, forbidden["end_frame"]) > max(start, forbidden["start_frame"])):
                    raise ValueError("源帧与已登记禁用范围重叠。")
            source_start, source_end = index.time(start), index.time(end)
            if abs(source_end-source_start-(b-a)/FPS) > max(1/FPS, float(index.clock)) + 1e-7:
                raise ValueError("原速源时长与目标不符，请用fit确定合法帧范围；不允许循环或拉伸补时。")
            owner = next(s for s in segments if round(s.audio_end*FPS) > a)
            fingerprints.setdefault(asset["id"], (asset["fingerprint"], asset["path"]))
            if fingerprints[asset["id"]] != (asset["fingerprint"], asset["path"]):
                raise ValueError("跨库素材ID冲突，请重新登记。")
            assets[asset["id"]] = {**asset, "duration": index.duration}
            original_shots.append(VisualShot(asset["id"], item["id"], asset["path"], source_start, source_end,
                a/FPS, b/FPS, owner.id, f"{clip['match_level']}：{clip['reason']}", False,
                section["id"], source_stream_index=index.data["stream_index"],
                source_origin_seconds=float(index.data["origin_pts"]*index.clock), source_index_key=index.data["key"]))
        except (ValueError, KeyError, TypeError, StopIteration) as exc:
            raise ValueError(f"镜头{label or '?'}：{exc}") from None
    original_shots.sort(key=lambda shot: shot.timeline_start)
    for left, right in zip(original_shots, original_shots[1:]):
        if left.timeline_end > right.timeline_start + 1e-7:
            raise ValueError("原片镜头目标时间重叠。")
    motions = [VisualShot(**row) for row in snapshot["motion"]["reservations"]]
    # Round only the output clock. Source animation slices remain continuous and unchanged.
    for shot in motions:
        shot.timeline_start = round(shot.timeline_start*FPS)/FPS
        shot.timeline_end = round(shot.timeline_end*FPS)/FPS
    shots = sorted([*original_shots, *motions], key=lambda shot: shot.timeline_start)
    all_assets = list(assets.values()) + snapshot["motion"]["assets"]
    if complete:
        position = 0
        for shot in shots:
            start, end = round(shot.timeline_start*FPS), round(shot.timeline_end*FPS)
            if start != position or end <= start:
                raise ValueError(f"目标帧{position}至{start}有缺口或重叠，请调整邻接镜头；不能留一帧空洞。")
            position = end
        if position != snapshot["total_frames"]:
            raise ValueError(f"画面仅覆盖到帧{position}，目标为{snapshot['total_frames']}；请完成全部原片区间。")
        validate_timeline_structure(segments, shots, all_assets, global_track=True, tolerance=1/FPS+1e-7)
    return segments, shots, all_assets


def _current(directory, snapshot, catalogue, proposal):
    if stable_hash(_snapshot(directory)) != stable_hash(snapshot):
        raise ValueError("原文、配音、动画、库选择或画面偏好已改变，请创建新的整篇任务。")
    current = LibraryStore.catalogue_snapshot(snapshot["library_ids"])
    for library, rid in {(c["library_id"], r) for c in proposal["clips"] for r in c["range_ids"]}:
        old, old_asset = _range(catalogue, library, rid)
        now, asset = _range(current, library, rid)
        if stable_hash(old) != stable_hash(now) or any(old_asset.get(k) != asset.get(k) for k in ASSET_FIELDS):
            raise ValueError(f"范围{rid}或来源登记已更新，请创建新任务；未覆盖旧方案。")
        _index(asset, now)
    checked = set()
    for clip in proposal["clips"]:
        _, asset = _range(catalogue, clip["library_id"], clip["range_ids"][0])
        if (asset["library_id"], asset["id"]) not in checked:
            _source_current(asset)
            checked.add((asset["library_id"], asset["id"]))
    # New prohibitions on a used source must apply even when their range was not cited.
    for clip in proposal["clips"]:
        item, asset = _range(catalogue, clip["library_id"], clip["range_ids"][0])
        for row in current["ranges"]:
            if (row.get("fingerprint") == asset["fingerprint"] and row.get("index_key") == item["index_key"] and
                    row.get("content_kind") in {"opening", "credits", "black"} and
                    min(clip["source_end_frame"], row["end_frame"]) > max(clip["source_start_frame"], row["start_frame"])):
                raise ValueError(f"镜头{clip['clip_id']}碰到当前已登记禁用范围。")


def submit(project_id, task_id, filename):
    with project_lock(project_id), runner.library_operation(project_id, prefix="narration:"):
        directory, root, key, manifest, snapshot, catalogue = _load(project_id, task_id)
        path = _owned(root, filename)
        if not path.resolve().is_relative_to(_owned(root, "proposals").resolve()) or path.suffix != ".json":
            raise ValueError("提交文件必须位于本任务proposals内。")
        incoming = read_json(path)
        if not isinstance(incoming, dict):
            raise ValueError("提案应为JSON对象。")
        for field in ("protocol", "project_id", "task_id", "input_hash"):
            if incoming.get(field) != manifest[field]:
                raise ValueError("提案的" + field + "与任务不符。")
        if type(incoming.get("complete")) is not bool or type(incoming.get("expected_revision")) is not int:
            raise ValueError("需填写complete布尔值与expected_revision整数。")
        previous, head = _revision(root, key)
        submission_hash = stable_hash(incoming)
        submitted_receipt = _owned(root, "receipts/submission-" + submission_hash + ".json")
        if submitted_receipt.is_file():
            saved = _signed(submitted_receipt, key)
            if saved["revision"] <= head["revision"]:
                return {**saved, "current_revision": head["revision"], "duplicate": True}
        if head.get("submission_hash") == submission_hash:
            return head
        workset_id = incoming.get("workset_id")
        if workset_id is not None and (not isinstance(workset_id, str) or not workset_id):
            raise ValueError("workset_id应为非空分工编号，主方案应省略此字段。")
        expected = head["workset_revisions"].get(workset_id, 0) if workset_id else head["revision"]
        if incoming["expected_revision"] != expected:
            raise ValueError(f"修订冲突：当前应使用expected_revision={expected}，请先describe再合并；已保存结果保留。")
        value = {k: incoming[k] for k in ("sections", "clips", "complete")}
        revisions = dict(head["workset_revisions"])
        if workset_id:
            if head["complete"]:
                raise ValueError("主Agent已提交完成版，分工不能覆盖；需要主Agent先开启新的草稿修订。")
            worksets = _signed(_owned(root, "worksets-record.json"), key)
            workset = next((w for w in worksets if w["id"] == workset_id), None)
            if not workset or incoming["complete"]:
                raise ValueError("分工不存在或试图直接完成整片；请由主Agent最终汇总。")
            for section in value["sections"]:
                a, b = _span(section, "target", snapshot["total_frames"])
                if not workset["start_frame"] <= a < b <= workset["end_frame"]:
                    raise ValueError("分工草稿超出自己的时间范围。")
            value = {"complete": False, **{name: [r for r in previous[name] if r.get("workset_id") != workset_id] +
                [{**r, "workset_id": workset_id} for r in incoming[name]] for name in ("sections", "clips")}}
            revisions[workset_id] = expected + 1
        else:
            # A coordinator replacement invalidates outstanding workset revisions.
            workset_path = _owned(root, "worksets-record.json")
            ids = [row["id"] for row in _signed(workset_path, key)] if workset_path.is_file() else list(revisions)
            revisions = {identifier: revisions.get(identifier, 0)+1 for identifier in ids}
        _compile(snapshot, catalogue, value, complete=value["complete"])
        value.update(protocol=PROTOCOL, project_id=directory.name, task_id=root.name, input_hash=manifest["input_hash"],
                     revision=head["revision"]+1, previous_revision=head["revision"], submitted=time.time())
        number = value["revision"]
        receipt = {"revision": number, "result_hash": stable_hash(value), "input_hash": manifest["input_hash"]}
        _save(_owned(root, f"results/revision-{number}.json"), value, key)
        _save(_owned(root, f"receipts/revision-{number}.json"), receipt, key)
        head = {"revision": number, "complete": value["complete"], "clips": len(value["clips"]),
                "covered_frames": sum(c["target_end_frame"]-c["target_start_frame"] for c in value["clips"]),
                "levels": {level: sum(c["match_level"] == level for c in value["clips"]) for level in sorted(LEVELS)},
                "workset_revisions": revisions, "submission_hash": submission_hash, "updated": value["submitted"]}
        _save(_owned(root, "head.json"), head, key)
        _save(submitted_receipt, head, key)
        return head


def occupancy(project_id, task_id):
    _, root, key, manifest, _, _ = _load(project_id, task_id)
    value, head = _revision(root, key)
    return {"revision": head["revision"], "complete": head["complete"], "original_intervals": manifest["original_intervals"],
            "sections": value["sections"], "clips": value["clips"]}


def worksets(project_id, task_id, workers=4):
    if type(workers) is not int or not 1 <= workers <= 64:
        raise ValueError("分工数需为1至64。")
    with project_lock(project_id), runner.library_operation(project_id, prefix="narration:"):
        _, root, key, manifest, snapshot, _ = _load(project_id, task_id)
        if _signed(_owned(root, "head.json"), key)["revision"]:
            raise ValueError("已提交草稿，不能重新划分分工；复用现有worksets或由主Agent汇总。")
        boundaries = sorted({0, snapshot["total_frames"], *[round(s["audio_end"]*FPS) for s in snapshot["segments"]]})
        cuts = sorted({0, snapshot["total_frames"], *[min(boundaries, key=lambda n: abs(n-snapshot["total_frames"]*i/workers))
                                                   for i in range(1, workers)]})
        result = [{"id": f"W{i+1:02d}", "start_frame": a, "end_frame": b,
                   "segments": [s for s in snapshot["segments"] if round(s["audio_end"]*FPS) > a and round(s["audio_start"]*FPS) < b],
                   "original_intervals": [[max(a,l), min(b,r)] for l,r in manifest["original_intervals"] if min(b,r)>max(a,l)]}
                  for i,(a,b) in enumerate(zip(cuts,cuts[1:]))]
        for group in result:
            ids = {s["id"] for s in group["segments"]}
            positions = [i for i,s in enumerate(snapshot["segments"]) if s["id"] in ids]
            group["context_segments"] = (snapshot["segments"][max(0,positions[0]-1):positions[0]] +
                                         snapshot["segments"][positions[-1]+1:positions[-1]+2]) if positions else []
        _save(_owned(root, "worksets-record.json"), result, key)
        write_json(_owned(root, "worksets.json"), result)
        return result


def check_task(project_id, task_id, *, commit=False):
    with project_lock(project_id), runner.library_operation(project_id, prefix="narration:"):
        directory, root, key, manifest, snapshot, catalogue = _load(project_id, task_id)
        with runner.library_operation(snapshot["library_ids"]):
            value, head = _revision(root, key)
            if not head["revision"]:
                raise ValueError("尚无已提交的画面草稿。")
            segments, shots, assets = _compile(snapshot, catalogue, value, complete=True)
            _current(directory, snapshot, catalogue, value)
            report = {"valid": True, "complete": head["complete"], "revision": head["revision"],
                      "clips": head["clips"], "levels": head["levels"], "imported": False}
            if commit:
                if not head["complete"]:
                    raise ValueError("草稿已完整覆盖，但尚未由主Agent提交complete=true完成版。")
                from app.services.narration_task import _preserve_visual_result
                from app.services.timeline_planner import VISUAL_POLICY_VERSION
                timeline = timeline_dict(segments, shots)
                timeline["visual_policy"] = {"version": VISUAL_POLICY_VERSION, "producer": "agent", "agent": {
                    "project_id": directory.name, "task_id": root.name, "revision": value["revision"],
                    "input_hash": manifest["input_hash"], "result_hash": stable_hash(value)},
                    "sources": {a["id"]: a for a in assets if not a["id"].startswith("motion-")}}
                plan = snapshot["motion"]["plan"]
                if plan:
                    timeline["motion"] = {"version": 1, "plan_key": plan["key"], "timing_method": plan["timing_method"],
                        "active_ids": list(dict.fromkeys(s.asset_id for s in shots if s.asset_id.startswith("motion-"))),
                        **{k: plan[k] for k in ("manual_config_key", "placement") if k in plan}}
                state = read_json(_owned(directory, "state.json"))
                _preserve_visual_result(directory)
                report["imported"] = True
                _save(_owned(root, f"receipts/import-{value['revision']}.json"),
                      {**report, "project_id": directory.name, "task_id": root.name, "input_hash": manifest["input_hash"],
                       "result_hash": stable_hash(value), "time": time.time()}, key)
                write_json(_owned(directory, "timeline.json"), timeline)
                state.update(state="paused", phase="画面方案已导入", progress=74,
                    message="外部Agent完整画面方案已导入，可从4. 从配乐继续或5. 仅合成视频推进。", agent_visual_task=root.name)
                write_json(_owned(directory, "state.json"), state)
            return report


def validate_agent_policy(segments, shots, assets, policy):
    """An Agent producer is accepted only through its signed task revision, never a flag alone."""
    reference = policy.get("agent", {})
    directory, root, key, manifest, snapshot, catalogue = _load(reference.get("project_id"), reference.get("task_id"))
    value, _ = _revision(root, key, reference.get("revision"))
    if (not value.get("complete") or reference.get("result_hash") != stable_hash(value) or
            reference.get("input_hash") != manifest["input_hash"]):
        raise ValueError("Agent时间轴缺少有效完成版或导入来源。")
    imported = _signed(_owned(root, f"receipts/import-{value['revision']}.json"), key)
    if (imported["revision"] != value["revision"] or imported["result_hash"] != stable_hash(value) or not imported["imported"] or
            any(imported.get(field) != manifest[field] for field in ("project_id", "task_id", "input_hash"))):
        raise ValueError("此Agent修订尚未经过正式导入。")
    expected_segments, expected_shots, expected_assets = _compile(snapshot, catalogue, value, complete=True)
    from app.services.timeline_planner import coalesce_motion_shots
    if stable_hash(timeline_dict(segments, coalesce_motion_shots(shots))) != stable_hash(
            timeline_dict(expected_segments, coalesce_motion_shots(expected_shots))):
        raise ValueError("正式时间轴与Agent受控完成版不一致，请重新检查并导入。")
    provided = {a["id"]: a for a in assets}
    for asset in expected_assets:
        if asset["id"] not in provided or any(provided[asset["id"]].get(k) != asset.get(k) for k in ("path", "fingerprint")):
            raise ValueError("成片来源与Agent完成版不一致。")
    _current(directory, snapshot, catalogue, value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True)
    parser.add_argument("--task", required=True)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("describe").add_argument("--full", action="store_true")
    commands.add_parser("occupancy")
    commands.add_parser("check")
    p = commands.add_parser("search")
    p.add_argument("--query", default="")
    p.add_argument("--kind", choices=("ranges", "contents"), default="ranges")
    p.add_argument("--episode", type=int)
    p.add_argument("--library")
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--limit", type=int, default=20)
    for name in ("view", "fit"):
        p = commands.add_parser(name)
        p.add_argument("--library", required=True)
        p.add_argument("--range", dest="range_id", required=True)
        if name == "view":
            p.add_argument("--capture", action="store_true")
            p.add_argument("--frames", help="Actual presentation frame ordinals separated by commas")
        else:
            p.add_argument("--start-frame", type=int, required=True)
            p.add_argument("--target-frames", type=int, required=True)
    p = commands.add_parser("submit")
    p.add_argument("--file", required=True)
    p = commands.add_parser("worksets")
    p.add_argument("--workers", type=int, default=4)
    args = vars(parser.parse_args())
    project_id, task_id, command = (args.pop(k) for k in ("project", "task", "command"))
    try:
        if command == "submit":
            args["filename"] = args.pop("file")
        if command == "view" and args.get("frames"):
            args["frames"] = [int(n) for n in args["frames"].split(",")]
            args["capture"] = True
        result = {"describe": describe, "search": search, "view": view, "fit": fit, "submit": submit,
                  "occupancy": occupancy, "worksets": worksets, "check": check_task}[command](project_id, task_id, **args)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
