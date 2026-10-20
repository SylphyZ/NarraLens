"""Project creation and progress with shared, lazily rendered settings."""

import json
import io
import wave
from copy import deepcopy
from pathlib import Path
from uuid import uuid4

import streamlit as st

from app.config import config
from app.services.audio_library import AudioLibraryStore
from app.services.library import runner
from app.services.library.gateway import ModelGateway
from app.services.library.media_server import video_url
from app.services.library.store import LibraryStore, library_ids, write_json, read_json, stable_hash
from app.services.narration_task import (
    audio_mode, create_task, delete_task, fork_task, list_tasks, read_task, restore_voice, restore_models,
    project_edit_status, update_project_settings,
    RESUME_STAGES, resume_stage_status, run_task, task_deletion_info, task_directory,
)
from app.services.narration_alignment import LEGACY_MODE, LONG_MODE
from app.services.narration_bgm import bgm_libraries
from app.services.narration_motion import (
    TEMPLATES, PLACEMENTS, MIN_SECONDS, text_anchors, manual_motion_entries, inspect_manual_motion_ranges, motion_entry_from_animation,
)
from app.models.timeline import NarrationSegment
from webui.components.library_settings import choose_library
from webui.components.narration_audio_settings import (
    audio_mode_label, initial_audio_options, render_narration_audio_panel, validate_voice, voice_values, audio_values,
)
from webui.components.audio_library_settings import render_bgm_plan, render_narration_bgm, bgm_values
from webui.utils.file_utils import remembered_widget_value, remembered_expander, remember_widget_values
from webui.components.model_settings import seed_model_choices, selected_models as workspace_models, render_model_choices, selected_bindings, check_model_choices
from app.services.model_profiles import resolve_models


def _start_task(task_id, models, voice, use_current_models=False, *, start_stage="auto"):
    def work(stop):
        try:
            run_task(task_id, models, voice, stop, use_current_models, start_stage=start_stage)
        except Exception as exc:
            state = read_task(task_id) or {"id": task_id}
            state.update(state="failed", message=str(exc))
            write_json(task_directory(task_id) / "state.json", state)
    saved = read_json(task_directory(task_id) / "settings.json")
    resources = tuple(["narration:" + task_id, *["library:" + identifier for identifier in library_ids(saved)]]
                      + ["audio-library:" + identifier for identifier in bgm_libraries(saved["options"])])
    if not runner.start("narration:" + task_id, work, resources=resources):
        raise ValueError("视频或音频素材库、关联任务正在运行，请等待任务结束。")


def _render_request_input(task_id, latest, tr):
    filename = latest.get("request_file")
    if not filename:
        return
    directory = task_directory(task_id)
    request_path = (directory / filename).resolve()
    if request_path.parent != (directory / "model-requests").resolve() or request_path.suffix != ".json":
        return
    try:
        request = read_json(request_path)
        if not isinstance(request, dict):
            raise ValueError("请求输入记录格式无效。")
    except (OSError, ValueError) as exc:
        st.warning(tr("读取请求输入记录失败：") + str(exc))
        return
    st.caption(tr("请求输入记录"))
    st.code(str(request_path), language=None)
    st.download_button(tr("下载请求输入记录"), json.dumps(request, ensure_ascii=False, indent=2),
                       file_name=request_path.name, mime="application/json", key="request_input_" + task_id + request_path.stem)
    st.json(request.get("metadata", {}))
    for message in request.get("messages", []):
        st.code(f"{message.get('role', '')}:\n{message.get('content', '')}", language=None)
    frame_root = (Path(config.root_dir) / "storage" / "libraries").resolve()
    frames, captions = [], []
    for reference in request.get("images", [])[:9]:
        frame = Path(reference.get("path", "")).resolve()
        if frame.is_relative_to(frame_root) and frame.suffix.lower() in {".jpg", ".jpeg", ".png"} and frame.is_file():
            frames.append(str(frame))
            captions.append(f"#{reference.get('index', '?')} · {frame.name}")
    if frames:
        st.caption(tr("候选帧（按发送顺序）"))
        st.image(frames, caption=captions, width=240)


def _render_process_diagnostics(task_id, state, tr, folder, title, phases):
    directory = task_directory(task_id) / folder
    if not directory.is_dir():
        return
    key = "narration_diag_open_" + task_id + "_" + folder.replace("/", "_")
    with remembered_expander(tr(title), key=key, default=state["state"] == "failed" and state.get("phase") in phases) as panel:
        if not panel.open:
            return
        paths = list(directory.glob("*.json"))
        if not paths:
            st.caption(tr("暂无执行记录。"))
            return
        path = max(paths, key=lambda item: item.stat().st_mtime)
        try:
            record = read_json(path)
        except (OSError, ValueError) as exc:
            st.warning(tr("读取执行记录失败：") + str(exc))
            return
        st.json({key: value for key, value in record.items() if key != "output_tail"})
        if record.get("output_tail"):
            st.code(record["output_tail"][-6000:], language=None)
        st.code(str(path), language=None)
        st.download_button(tr("下载执行记录"), json.dumps(record, ensure_ascii=False, indent=2),
                           file_name=path.name, mime="application/json", key=folder + "_record_" + task_id + path.stem)
        log_path = path.with_suffix(".log")
        if log_path.is_file():
            st.download_button(tr("下载输出日志"), log_path.read_bytes(), file_name=log_path.name,
                               mime="text/plain", key=folder + "_log_" + task_id + path.stem)


@st.fragment(run_every="2s")
def _render_task_status(task_id, tr):
    remember_widget_values()
    state = read_task(task_id)
    if not state:
        return
    observed_key = "narration_observed_state_" + task_id
    previous_state = st.session_state.get(observed_key)
    st.session_state[observed_key] = state["state"]
    if previous_state == "running" and state["state"] != "running":
        st.rerun(scope="app")
    st.progress(max(0, min(1, state.get("progress", 0) / 100)))
    message = state.get("phase", "") + " · " + state.get("message", "")
    if state["state"] == "failed":
        st.error(message)
    elif state["state"] == "complete":
        st.success(message)
    else:
        st.info(message)
    usage = state.get("token_usage", {})
    if isinstance(usage, dict) and usage.get("version") == 1:
        current = usage.get("current", {})
        cumulative = usage.get("cumulative", {})
        total = lambda rows, field: sum(row.get(field, 0) for row in rows.values())
        st.caption(tr("本次已报告Token：") + f"{total(current, 'total_tokens'):,}" + tr(" · 项目累计：") +
                   f"{total(cumulative, 'total_tokens'):,}" + tr(" · 本次用量不完整请求：") + str(total(current, "unknown")))
        with remembered_expander(tr("Token用量"), key="narration_token_usage_"+task_id) as panel:
            if panel.open:
                labels = {"text_preselect": "文字预选", "visual_review": "内容确认", "visual_safety": "画面可用性", "other": "其他阶段"}
                records = [{tr("范围"): tr(scope), tr("阶段"): tr(labels.get(kind, kind)), tr("请求"): row.get("requests", 0),
                    tr("已报告输入"): row.get("prompt_tokens", 0), tr("已报告输出"): row.get("completion_tokens", 0),
                    tr("总Token"): row.get("total_tokens", 0), tr("未知／不完整"): row.get("unknown", 0),
                    tr("已报告推理Token"): row.get("reasoning_tokens", 0), tr("已报告缓存输入"): row.get("cached_tokens", 0),
                    tr("缓存信息缺失请求"): row.get("cached_unknown", 0),
                    tr("图片提交"): row.get("images", 0), tr("重试／后备Token"): row.get("retry_tokens", 0),
                    tr("后续补查Token"): row.get("followup_tokens", 0)}
                    for scope, rows in (("本次继续", current), ("项目累计", cumulative)) for kind, row in rows.items()]
                st.dataframe(records, hide_index=True, width="stretch")
                st.caption(tr("只累计接口已报告用量；未知不能按0计费。重试和补查是总量的子集，不重复相加。历史图片统计可能不完整。"))
                st.caption(tr("统计本项目网关中的文案／视觉请求；配音及素材库独立分析的用量另计。推理属于输出明细，缓存输入属于输入明细。"))
    if runner.running("narration:" + task_id):
        if st.button(tr("停止当前任务"), key="cancel_narration_" + task_id):
            runner.cancel("narration:" + task_id)
            st.info(tr("正在停止，已完成的音频和分析会保留。"))
    if state["state"] == "complete" and Path(state.get("video", "")).is_file():
        url = video_url(task_id)
        st.video(url)
        st.markdown(f"[{tr('下载成片')}]({url}?download=1)")
        st.code(state["video"], language=None)
    elif state["state"] == "running" and not runner.running("narration:" + task_id):
        st.warning(tr("上次运行已中断，可以重新点击继续。"))
    layout_path = task_directory(task_id) / "subtitle_layout.json"
    if layout_path.is_file():
        _render_layout_report(task_id, state, layout_path, tr)
    diagnostic_path = task_directory(task_id) / "model-diagnostics.jsonl"
    if diagnostic_path.is_file():
        with remembered_expander(tr("模型调用详情"), key="narration_model_log_open_" + task_id) as panel:
            if panel.open:
                latest = state.get("last_model_request", {})
                if latest:
                    st.json({key: value for key, value in latest.items() if key != "response_excerpt"})
                    if latest.get("response_excerpt"):
                        st.code(latest["response_excerpt"], language=None)
                    _render_request_input(task_id, latest, tr)
                st.download_button(tr("下载模型调用日志"), diagnostic_path.read_bytes(), file_name="model-diagnostics.jsonl",
                                   mime="application/jsonl", key="model_diagnostics_" + task_id)
    _render_process_diagnostics(task_id, state, tr, "ffmpeg-diagnostics", "FFmpeg执行详情", {"合成视频", "生成配音", "生成长段配音", "制作配乐"})
    _render_process_diagnostics(task_id, state, tr, "alignment-diagnostics", "本地对齐详情", {"字幕对齐"})
    _render_process_diagnostics(task_id, state, tr, "motion/diagnostics", "动画执行详情", {"渲染解释动画"})
    remember_widget_values()


def _render_layout_report(task_id, state, layout_path, tr):
    with remembered_expander(tr("字幕排版结果"), key="narration_layout_open_" + task_id,
            default=state.get("phase") == "字幕排版" and state["state"] == "paused") as panel:
        if panel.open:
            report = read_json(layout_path)
            st.caption(f"{report['font_size']} " + tr("号字，行宽") + f" {report['width_ratio']:.0%} · {report['cue_count']} " + tr("条字幕"))
            if report["short_count"]:
                st.warning(tr("当前字号、行宽和原音时间无法全部满足阅读目标。请调整样式副本，或明确允许短字幕。"))
                st.dataframe([{"字幕": cue["text"], "秒数": round(cue["end"] - cue["start"], 2), "段落": cue["segment_id"]}
                              for cue in report["cues"] if cue.get("short")], hide_index=True, width="stretch")
            st.download_button(tr("下载字幕排版结果"), layout_path.read_bytes(), file_name="subtitle_layout.json",
                               mime="application/json", key="narration_layout_download_" + task_id)


def _motion_entries(prefix):
    rows = deepcopy(remembered_widget_value(prefix + "_motion_manual_entries", []))
    if not isinstance(rows, list):
        raise ValueError("手动动画节点记录无效。")
    digest = remembered_widget_value(prefix + "_motion_manual_text_hash", "")
    for row in rows:
        row.setdefault("text_hash", digest)
    st.session_state[prefix + "_motion_manual_entries"] = rows
    return rows


def _motion_values(prefix):
    value = lambda name, default=None: remembered_widget_value(prefix + "_motion_" + name, default)
    result = {"motion_enabled": bool(value("enabled", False))}
    if result["motion_enabled"]:
        result.update(motion_frequency=value("frequency", "normal"),
            motion_templates=value("templates", list(TEMPLATES)), motion_accent=value("accent", "#66D5B8"),
            motion_concurrency=int(value("concurrency", 2)))
    entries = _motion_entries(prefix)
    placement = value("placement", "auto")
    if placement != "auto" or entries:
        result.update(motion_placement=placement, motion_manual_entries=entries,
                      motion_manual_text_hash=value("manual_text_hash", ""))
    return result


@st.cache_data(show_spinner=False, max_entries=16)
def _motion_anchors(text, clauses):
    return [{**row, "ordinal": index} for index, row in enumerate(text_anchors(text, clauses=clauses), 1)]


def _motion_position(label, anchors, field, key, default):
    choices = {row[field]: row for row in anchors}
    if key in st.session_state and st.session_state[key] not in choices:
        st.session_state[key] = None
    value = remembered_widget_value(key, default)
    values = list(choices)
    return st.selectbox(label, values, index=None if key in st.session_state else values.index(value) if value in choices else None,
        format_func=lambda position: f"{choices[position]['ordinal']}. {choices[position]['text'][:72]}", key=key,
        disabled=not values, placeholder="选择对应原文")


def _motion_segments(directory):
    plan = read_json(directory / "document_plan.json")
    audio = read_json(directory / "audio_manifest.json")
    timings = {row["id"]: row for row in audio["segments"]}
    segments = [NarrationSegment(**row) for row in plan["segments"]]
    for segment in segments:
        row = timings[segment.id]
        segment.audio_start, segment.audio_end = row["start"], row["end"]
    return segments


def _motion_listen(tr, directory, start, end, key):
    if not st.button(tr("试听对应配音"), key=key):
        return
    try:
        data = io.BytesIO()
        with wave.open(str(directory / "narration.wav"), "rb") as source:
            rate = source.getframerate()
            first, last = round(start * rate), round(end * rate)
            if not 0 <= first < last <= source.getnframes():
                raise ValueError("试听范围超出完整配音。")
            source.setpos(first)
            with wave.open(data, "wb") as target:
                target.setnchannels(source.getnchannels())
                target.setsampwidth(source.getsampwidth())
                target.setframerate(rate)
                target.writeframes(source.readframes(last-first))
        st.audio(data.getvalue(), format="audio/wav")
    except (OSError, ValueError, wave.Error) as exc:
        st.warning(tr("无法试听对应配音：") + str(exc))


@st.fragment
def _render_manual_motion(tr, text, prefix):
    remember_widget_values(prefix)
    with remembered_expander(tr("手动插入节点"), key=prefix + "_motion_manual_open") as panel:
        if not panel.open:
            return
        if not text.strip():
            st.info(tr("先输入完整文案，再设置动画节点。"))
            return
        rows = _motion_entries(prefix)
        digest = stable_hash(text)
        clauses = st.checkbox(tr("细分到逗号、冒号"),
            value=remembered_widget_value(prefix + "_motion_clauses", False), key=prefix + "_motion_clauses")
        anchors = _motion_anchors(text, clauses)
        if not anchors:
            st.info(tr("文案没有可绑定的发声内容。"))
            return
        ordered = sorted(rows, key=lambda row: row.get("text_start") if type(row.get("text_start")) is int else len(text))
        page_key, select_key = prefix + "_motion_page", prefix + "_motion_selected"
        pending = st.session_state.pop(prefix + "_motion_pending_select", None)
        identifiers = [row["id"] for row in ordered]
        if pending in identifiers:
            st.session_state[page_key] = identifiers.index(pending) // 50 + 1
            st.session_state[select_key] = pending
        pages = max(1, (len(rows) + 49) // 50)
        page = min(max(1, int(remembered_widget_value(page_key, 1))), pages)
        st.session_state[page_key] = page
        page = st.number_input(tr("节点列表页码"), 1, pages, step=1, key=page_key)
        visible = ordered[(page-1)*50:page*50]
        choices = [row["id"] for row in visible]
        labels = {row["id"]: f"{identifiers.index(row['id'])+1}. {row['id'][7:15]} · {TEMPLATES.get(row.get('template'), '动画')} · " +
            (text[row['text_start']:row['text_end']].strip()[:48] if type(row.get('text_start')) is int and type(row.get('text_end')) is int else '未绑定') +
            ("（已禁用）" if not row.get("enabled", True) else "") + ("（需重新绑定）" if row.get("text_hash") != digest else "") for row in visible}

        def start_node():
            st.session_state[select_key] = None
            st.session_state.pop(prefix + "_motion_pending_select", None)
            remember_widget_values(prefix)

        add_col, select_col = st.columns([1, 5], vertical_alignment="bottom")
        add_col.button(tr("新增节点"), type="primary", key=prefix + "_motion_add_node", on_click=start_node)
        if select_key in st.session_state and st.session_state[select_key] not in choices:
            st.session_state[select_key] = None
        selected = remembered_widget_value(select_key)
        selected = selected if selected in choices else None
        if select_key not in st.session_state:
            st.session_state[select_key] = selected
        selected = select_col.selectbox(tr("选择编辑节点"), choices, index=None,
            format_func=lambda identifier: labels[identifier], key=select_key, disabled=not choices,
            placeholder=tr("选择已有节点进行编辑"))
        st.caption(tr("按原文位置显示，每页最多50条；保存节点草稿后，通过保存项目设置或动画副本应用。"))
        if selected is None:
            st.caption(tr("正在填写新节点草稿；保存后加入列表。"))
        new_key = prefix + "_motion_new_id_" + digest[:16]
        row = deepcopy(next((row for row in rows if row["id"] == selected), None) or
            {"id": st.session_state.setdefault(new_key, "manual-" + uuid4().hex), "text_hash": digest,
             "text_start": anchors[0]["text_start"], "text_end": anchors[0]["text_end"], "template": "chapter"})
        stem = prefix + "_motion_node_" + row["id"] + "_" + digest[:16] + ("_clauses" if clauses else "_sentences")
        stale = row.get("text_hash") != digest
        st.caption(tr("节点编号：") + row["id"])
        if stale:
            st.warning(tr("原文已改变，请重新选择起止并保存此节点。其他旧节点会继续标记失效。"))
        enabled = st.checkbox(tr("启用此节点"), value=remembered_widget_value(stem + "_enabled", row.get("enabled", True)), key=stem + "_enabled")
        first = _motion_position(tr("起句：开始朗读时显示动画"), anchors, "text_start", stem + "_start", None if stale else row.get("text_start"))
        endings = [anchor for anchor in anchors if first is not None and anchor["text_end"] > first]
        last = _motion_position(tr("终句：读完此句结束动画"), endings, "text_end", stem + "_end", None if stale else row.get("text_end"))
        bound = first is not None and last is not None
        if bound:
            st.text(text[first:last].strip())
        template = st.selectbox(tr("节点动画类型"), list(TEMPLATES),
            index=list(TEMPLATES).index(remembered_widget_value(stem + "_template", row.get("template", "chapter"))),
            format_func=lambda value: tr(TEMPLATES[value]), key=stem + "_template")
        st.caption(tr("此模板最短时长：") + f"{MIN_SECONDS[template]} " + tr("秒；实际时间在配音完成后检查。"))
        mode = st.radio(tr("动画内容来源"), ("model", "manual"),
            index=("model", "manual").index(remembered_widget_value(stem + "_content_mode", row.get("content_mode", "model"))),
            format_func=lambda value: tr("模型概括固定范围") if value == "model" else tr("手工填写"), key=stem + "_content_mode")
        draft = {**row, "enabled": enabled, "template": template, "content_mode": mode,
                 "text_start": first if bound else row.get("text_start"), "text_end": last if bound else row.get("text_end"),
                 "text_hash": digest if bound else row.get("text_hash")}
        for name, label in (("include_leading_pause", "包含起句前停顿"), ("include_trailing_pause", "包含终句后停顿")):
            draft[name] = st.checkbox(tr(label), value=remembered_widget_value(stem + "_" + name, row.get(name, False)),
                key=stem + "_" + name, help=tr("按前一字词结束／下一字词开始延伸，首尾使用整条配音边界。相邻节点应避免同时覆盖同一停顿。"))
        if mode == "manual":
            st.caption(tr("可保存未完成的内容草稿；生成前需要补齐标题、条目与原文绑定。"))
            draft["title"] = st.text_input(tr("动画标题"), value=remembered_widget_value(stem + "_title", row.get("title", "")),
                key=stem + "_title", help=tr("章节最多32字，其他类型最多20字。"))
            if template == "compare":
                for side, label in (("left", "左侧标题"), ("right", "右侧标题")):
                    draft[side + "_title"] = st.text_input(tr(label), value=remembered_widget_value(stem + "_" + side, row.get(side + "_title", "")),
                        key=stem + "_" + side, help=tr("最多8字。"))
            lower, upper = (0, 2) if template == "chapter" else (2, 4) if template == "flow" else (2, 6)
            count_key = stem + "_count_" + template
            st.session_state[count_key] = min(upper, max(lower, int(remembered_widget_value(count_key, len(row.get("items", []))))))
            count = st.number_input(tr("条目数"), lower, upper, step=1, key=count_key)
            item_anchors = [anchor for anchor in _motion_anchors(text, True) if bound and anchor["text_start"] >= first and anchor["text_end"] <= last]
            draft["items"] = []
            for index in range(count):
                saved_item = row.get("items", [])[index] if index < len(row.get("items", [])) else {}
                key = stem + "_item_" + str(index)
                label = st.text_input(tr("条目文字") + f" {index+1}", value=remembered_widget_value(key + "_text", saved_item.get("text", "")),
                    key=key + "_text", help=tr("章节副标题最多32字，其他条目最多10字。"))
                column = st.selectbox(tr("所在侧") + f" {index+1}", ("left", "right"),
                    index=("left", "right").index(remembered_widget_value(key + "_column", saved_item.get("column", "left"))),
                    format_func=lambda value: tr("左侧") if value == "left" else tr("右侧"), key=key + "_column", disabled=template != "compare")
                default = item_anchors[min(index, len(item_anchors)-1)] if item_anchors else {}
                a = _motion_position(tr("条目起句") + f" {index+1}", item_anchors, "text_start", key + "_start",
                    saved_item.get("text_start", default.get("text_start")))
                b = _motion_position(tr("条目终句") + f" {index+1}", [anchor for anchor in item_anchors if a is not None and anchor["text_end"] > a],
                    "text_end", key + "_end", saved_item.get("text_end", default.get("text_end")))
                draft["items"].append({"text": label, "column": column, "text_start": a, "text_end": b})
        else:
            st.caption(tr("第3阶段由现有文案模型概括标题和条目，位置与模板固定；打开编辑区不会调用API。"))
        save_col, remove_col = st.columns(2)
        if save_col.button(tr("保存节点草稿"), disabled=enabled and not bound, key=stem + "_save"):
            updated = [other for other in rows if other["id"] != row["id"]] + [draft]
            try:
                manual_motion_entries(text, {"motion_enabled": True, "motion_placement": "manual", "motion_manual_entries": [draft]})
                st.session_state[prefix + "_motion_manual_entries"] = updated
                st.session_state[prefix + "_motion_manual_text_hash"] = digest
                st.session_state[prefix + "_motion_pending_select"] = row["id"]
                if selected is None:
                    st.session_state.pop(new_key, None)
                remember_widget_values(prefix)
                st.rerun()
            except ValueError as exc:
                st.error(str(exc))
        if remove_col.button(tr("移除此节点"), disabled=selected is None, key=stem + "_remove"):
            st.session_state[prefix + "_motion_manual_entries"] = [other for other in rows if other["id"] != selected]
            remember_widget_values(prefix)
            st.rerun()
        if prefix.startswith("narration_edit_"):
            directory = task_directory(prefix[len("narration_edit_"):])
            manifest = directory / "audio_manifest.json"
            try:
                revision = manifest.stat().st_mtime_ns
            except OSError:
                revision = None
            signature = stable_hash((digest, draft, rows, revision))
            current_col, all_col = st.columns(2)
            check_current = current_col.button(tr("查看当前范围的配音时间"), disabled=not bound, key=stem + "_timing")
            check_all = all_col.button(tr("检查全部节点"), disabled=not rows, key=prefix + "_motion_check_all")
            st.caption(tr("全部检查包含已保存的节点草稿及当前所选节点的修改；新节点请先保存。"))
            if check_current or check_all:
                try:
                    if not manifest.is_file():
                        raise ValueError(tr("配音完成后可检查实际时长；当前可以先保存节点草稿。"))
                    current = draft if check_all else {**draft, "enabled": True}
                    checking = [current if other["id"] == selected else other for other in rows]
                    if check_current and selected is None:
                        checking.append(current)
                    options = {"motion_enabled": True, "motion_placement": "manual", "motion_manual_entries": checking}
                    report = inspect_manual_motion_ranges(_motion_segments(directory), text, directory, options)
                    if check_current:
                        report["nodes"] = [node for node in report["nodes"] if node["id"] == draft["id"]]
                    report.pop("ranges")
                    st.session_state[prefix + "_motion_time_preview"] = {"signature": signature, "report": report, "all": check_all}
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    st.session_state.pop(prefix + "_motion_time_preview", None)
                    st.warning(tr("配音时间尚不可用：") + str(exc))
            preview = st.session_state.get(prefix + "_motion_time_preview", {})
            if preview.get("signature") == signature:
                report = preview["report"]
                st.caption(tr({"aligned_words": "本地字词对齐", "text_estimate": "文字估时"}.get(report["method"], "字幕插值")))
                if report["nodes"]:
                    st.dataframe([{
                        tr("序号"): node["number"], tr("节点"): str(node["id"])[7:15],
                        tr("类型"): node["template_name"],
                        tr("原文"): node["text"],
                        tr("开始（秒）"): round(node["start"], 3) if node["time_valid"] else None,
                        tr("结束（秒）"): round(node["end"], 3) if node["time_valid"] else None,
                        tr("时长（秒）"): round(node["end"]-node["start"], 3) if node["time_valid"] else None,
                        tr("最短（秒）"): node["minimum_seconds"],
                        tr("检查结果"): "；".join(node["issues"]) if node["issues"] else tr("通过"),
                    } for node in report["nodes"]], hide_index=True, width="stretch")
                    for node in report["nodes"]:
                        if node["issues"]:
                            st.warning(node["label"] + "：" + "；".join(node["issues"]))
                    if not any(node["issues"] for node in report["nodes"]):
                        st.success(tr("原文范围与配音时间检查通过。"))
                    st.caption(tr("本次只检查原文与时间；动画内容仍在生成阶段校验。"))
                    if not preview["all"] and report["nodes"][0]["time_valid"]:
                        node = report["nodes"][0]
                        _motion_listen(tr, directory, node["start"], node["end"], stem + "_range_preview")
                else:
                    st.info(tr("没有启用的手动节点。"))
        else:
            st.caption(tr("时间将在配音对齐后确定；模型概括内容可在方案预览中载入手工编辑。"))
    remember_widget_values(prefix)


@st.fragment
def _render_motion_options(tr, text, prefix="narration"):
    remember_widget_values(prefix)
    with remembered_expander(tr("解释动画"), key=prefix + "_motion_settings") as panel:
        if not panel.open:
            return _motion_values(prefix)
        enabled = st.checkbox(tr("加入解释动画"), value=remembered_widget_value(prefix + "_motion_enabled", False), key=prefix + "_motion_enabled")
        if not enabled:
            return _motion_values(prefix)
        st.caption(tr("根据论述内容加入章节、概念和观点动画，与旁白时间同步。"))
        placement = st.selectbox(tr("动画位置方式"), list(PLACEMENTS),
            index=list(PLACEMENTS).index(remembered_widget_value(prefix + "_motion_placement", "auto")),
            format_func=lambda value: tr(PLACEMENTS[value]), key=prefix + "_motion_placement")
        if placement != "manual":
            st.selectbox(tr("动画频率"), ["low", "normal", "rich"],
                index=["low", "normal", "rich"].index(remembered_widget_value(prefix + "_motion_frequency", "normal")),
                format_func=lambda value: tr({"low": "少量", "normal": "适中", "rich": "丰富"}[value]), key=prefix + "_motion_frequency")
            st.multiselect(tr("自动动画类型"), list(TEMPLATES), default=remembered_widget_value(prefix + "_motion_templates", list(TEMPLATES)),
                format_func=lambda value: tr(TEMPLATES[value]), key=prefix + "_motion_templates")
            st.caption(tr("自动动画时长目标约为全片的6%、12%、22%；混合模式先保留手动范围，再自动补充。"))
        if placement != "auto" or _motion_entries(prefix):
            _render_manual_motion(tr, text, prefix)
            st.caption(tr("启用的手动节点：") + str(sum(row.get("enabled", True) for row in _motion_entries(prefix))))
        st.color_picker(tr("动画强调色"), value=remembered_widget_value(prefix + "_motion_accent", "#66D5B8"), key=prefix + "_motion_accent")
        st.number_input(tr("动画渲染并发"), min_value=1, value=int(remembered_widget_value(prefix + "_motion_concurrency", 2)), step=1,
            key=prefix + "_motion_concurrency", help=tr("同时绘制的帧数，独立于素材库API并发；增加会使用更多CPU和内存。"))
        st.caption(tr("动画底部留白会跟随字幕字号和画幅变化。"))
        if Path(_output_values(prefix)["font"]).suffix.lower() == ".ttc":
            st.caption(tr("字幕使用TTC集合字体时，动画使用项目内置黑体。"))
    remember_widget_values(prefix)
    return _motion_values(prefix)


@st.fragment
def _render_motion_plan(task_id, tr):
    remember_widget_values()
    directory = task_directory(task_id)
    path = directory / "motion" / "plan.json"
    if not path.is_file():
        return
    with remembered_expander(tr("解释动画方案与预览"), key="narration_motion_plan_" + task_id) as panel:
        if not panel.open:
            return
        try:
            plan = read_json(path)
            animations = plan["animations"]
            edits = read_json(directory / "motion" / "edits.json", default={})
            manifest = read_json(directory / "motion" / "manifest.json", default={})
            saved = read_json(directory / "settings.json")
        except (OSError, ValueError, KeyError, TypeError) as exc:
            st.warning(tr("无法读取解释动画结果：") + str(exc))
            return
        editable = not (runner.running("narration:" + task_id) or runner.library_busy(library_ids(saved))
            or runner.library_busy(bgm_libraries(saved["options"]), prefix="audio-library:"))
        if plan.get("timing_method") == "subtitle_interpolation":
            st.caption(tr("动画按已有字幕时间定位，句内节点时间使用插值。"))
        elif plan.get("timing_method") == "text_estimate":
            st.caption(tr("本地对齐已关闭，动画使用原任务的文字估时。"))
        else:
            st.caption(tr("动画按原文的本地字词对齐时间定位。"))
        st.dataframe([{"来源": tr("手动") if row.get("source") == "manual" else tr("自动"),
                       "段落": "、".join(span["segment_id"] for span in row.get("spans", [])) or row["segment_id"], "类型": tr(TEMPLATES[row["template"]]), "标题": row["title"],
                       "开始（秒）": round(row["start"], 2), "结束（秒）": round(row["end"], 2), "理由": row["reason"]}
                      for row in animations], hide_index=True, width="stretch")
        st.download_button(tr("下载动画方案"), path.read_bytes(), file_name="motion-plan.json", mime="application/json",
                           key="narration_motion_plan_download_" + task_id)
        if not animations:
            st.caption(tr("本次没有符合文案与频率的动画区间，使用影视画面。"))
            return
        identifiers = [row["id"] for row in animations]
        labels = {row["id"]: f"{row['segment_id']} · {TEMPLATES[row['template']]} · {row['title']}" for row in animations}
        automatic_ids = [row["id"] for row in animations if row.get("source") != "manual"]
        st.multiselect(tr("禁用这些自动动画（创建副本时应用）"), automatic_ids,
            default=[identifier for identifier in remembered_widget_value("narration_motion_disabled_" + task_id + plan["key"][:12], edits.get("disabled_ids", [])) if identifier in automatic_ids],
            format_func=labels.get, disabled=not editable, key="narration_motion_disabled_" + task_id + plan["key"][:12])
        st.caption(tr("手动节点的启用、范围与内容在项目设置中的手动编辑区修改。"))
        selected = st.selectbox(tr("预览动画片段"), identifiers, format_func=labels.get,
            key="narration_motion_preview_" + task_id + plan["key"][:12])
        row = next(row for row in animations if row["id"] == selected)
        st.caption(tr("对应原文："))
        st.write(row["source_text"])
        _motion_listen(tr, directory, row["start"], row["end"], "narration_motion_audio_preview_" + task_id + selected)
        if st.button(tr("以此方案载入手动节点草稿"), disabled=not editable, key="narration_motion_copy_" + task_id + selected):
            try:
                with (directory / "input.txt").open("r", encoding="utf-8", newline="") as source:
                    text = source.read()
                if stable_hash(text) != saved["text_hash"] or plan["text_hash"] != saved["text_hash"]:
                    raise ValueError("动画方案与原文已不一致，无法载入草稿。")
                entry = motion_entry_from_animation(row, _motion_segments(directory), text)
                prefix = "narration_edit_" + task_id
                entries = [item for item in _motion_entries(prefix) if item["id"] != entry["id"]] + [entry]
                placement = remembered_widget_value(prefix + "_motion_placement", "auto")
                st.session_state[prefix + "_motion_pending"] = {
                    "motion_enabled": True, "motion_placement": "mixed" if placement == "auto" else placement,
                    "motion_manual_entries": entries, "motion_manual_text_hash": stable_hash(text),
                    "motion_settings": True, "motion_manual_open": True, "settings_open": True,
                    "_reset_motion_entry": entry["id"]}
                st.session_state[prefix + "_motion_pending_select"] = entry["id"]
                st.rerun()
            except (OSError, ValueError, KeyError, TypeError) as exc:
                st.warning(tr("无法载入手动草稿：") + str(exc))
        asset = next((asset for asset in manifest.get("assets", []) if asset["id"] == selected), None)
        if asset:
            clip = Path(asset["path"])
            if clip.resolve().parent == (directory / "motion" / "clips").resolve() and clip.is_file():
                st.video(str(clip))
        else:
            st.caption(tr("该片段尚未完成渲染，或已在此任务中禁用。"))
    remember_widget_values()


def remember_project_widgets():
    """Keep inactive-page defaults in ordinary session data, excluding uploads/buttons."""
    remember_widget_values()


def _seed_project_widgets(prefix, saved=None):
    memory = st.session_state.setdefault("narration_widget_memory", {})
    st.session_state["_narration_settings_prefix"] = prefix
    seed_model_choices(prefix, models=(saved or {}).get("models"), voice=(saved or {}).get("voice"), options=(saved or {}).get("options"))
    fields = ("aspect", "subtitles", "font", "shot_seconds", "verify_visual", "visual_review_concurrency", "visual_no_gain_rounds", "subtitle_min_seconds",
              "subtitle_allow_short", "tts_block_chars", "tts_chunking", "tts_break_after", "tts_break_text_hash",
              "alignment_enabled", "alignment_python", "alignment_model", "alignment_language",
              "bgm_min_seconds", "bgm_crossfade", "bgm_ducking", "bgm_volume",
              "motion_enabled", "motion_frequency", "motion_templates", "motion_accent", "motion_concurrency",
              "motion_placement", "motion_manual_text_hash", "motion_manual_entries")
    aliases = {"font_size": "font_size_custom_v2", "mask_source_subtitles": "mask", "tts_budget": "tts_budget_v2",
               "bgm_mode": "bgm_source", "bgm_path": "bgm", "bgm_auto_strategy": "bgm_strategy",
               "bgm_only_instrumental": "auto_bgm_instrumental", "bgm_library_ids": "auto_bgm_libraries"}
    values = {}
    if saved:
        options, voice = saved["options"], saved["voice"]
        memory.setdefault(prefix + "_saved_tts_profile", saved.get("models", {}).get("_bindings", {}).get("tts")
                          or voice.get("profile_id") or remembered_widget_value(prefix + "_profile_tts", ""))
        values.update({field: options[field] for field in fields if field in options})
        values.update({name: options[field] for field, name in aliases.items() if field in options})
        values.update(audio_mode=audio_mode(options), tts_chunking=options.get("tts_chunking", "auto"),
                      font_size_custom_v2=options.get("font_size", 40),
                      subtitle_width=round(options.get("subtitle_width_ratio", .9) * 100),
                      bgm_source=options.get("bgm_mode", "path" if options.get("bgm_path") else "none"))
        voice_fields = {"engine": "tts_engine", "base_url": "mimo_base", "model": "mimo_model",
                        "rate": "edge_rate", "reference_audio": "reference_path", "instruction": "voice_instruction",
                        "voice": "edge_voice" if voice.get("engine") == "edge_tts" else "mimo_voice"}
        values.update({name: voice[field] for field, name in voice_fields.items() if field in voice})
        memory.setdefault(prefix + "_library_choice_multi", library_ids(saved))
    for name, value in values.items():
        memory.setdefault(prefix + "_" + name, deepcopy(value))
    pending = st.session_state.pop(prefix + "_motion_pending", None)
    if pending:
        reset_entry = pending.pop("_reset_motion_entry", None)
        if reset_entry:
            for mapping in (memory, st.session_state):
                for key in list(mapping):
                    if key.startswith(prefix + "_motion_node_" + reset_entry + "_"):
                        mapping.pop(key, None)
        for name, value in pending.items():
            key = prefix + "_" + name
            memory[key] = deepcopy(value)
            if name in {"motion_enabled", "motion_placement"}:
                st.session_state.pop(key, None)
            else:
                st.session_state[key] = deepcopy(value)


def _navigate(page, task_id=None):
    remember_project_widgets()
    if task_id:
        st.session_state["active_narration_task"] = task_id
    st.session_state["narration_pending_workspace"] = page
    st.rerun()


@st.fragment
def _render_task_manager(tr, selected, state):
    remember_widget_values()
    key = "narration_task_manager_open"
    with remembered_expander(tr("管理项目"), key=key) as panel:
        if not panel.open:
            return
        info_key = "narration_delete_info_" + selected
        st.caption(tr("先检查项目文件与删除范围，再确认删除。检查结果仅用于显示，删除时重新核验。"))
        if st.button(tr("检查文件与删除范围"), key="check_narration_delete_" + selected):
            try:
                st.session_state[info_key] = task_deletion_info(selected)
            except (OSError, ValueError) as exc:
                st.session_state.pop(info_key, None)
                st.error(tr("无法读取项目删除信息：") + str(exc))
        info = st.session_state.get(info_key)
        if not info:
            return
        st.code(info["directory"], language=None)
        st.caption(f"{info['files']} " + tr("个文件，约") + f" {info['bytes']/1024**2:.1f} MB")
        st.caption(tr("删除当前项目的输入副本、配音、缓存、日志和成片；原始素材及外部文案文件保留，关联副本可复用音频会先保存。"))
        state = read_task(selected) or state
        running = runner.running("narration:" + selected)
        busy = running or runner.library_busy(library_ids(state))
        confirmed = st.checkbox(tr("确认删除以上项目及其产物"), value=False, disabled=busy, key="delete_task_confirm_" + selected)
        interrupted = False
        if state["state"] == "running" and not running:
            interrupted = st.checkbox(tr("确认旧进程及其他窗口中的关联项目已经停止"), value=False, key="delete_task_interrupted_" + selected)
        if st.button(tr("永久删除当前项目"), disabled=busy or not confirmed or (state["state"] == "running" and not interrupted),
                     key="delete_narration_task_" + selected):
            try:
                message = delete_task(selected, confirmed=True, interrupted_confirmed=interrupted)
                st.session_state.pop("active_narration_task", None)
                st.session_state.pop("_narration_settings_prefix", None)
                st.session_state.pop(info_key, None)
                memory = st.session_state.get("narration_widget_memory", {})
                for name in list(memory):
                    if name.startswith("narration_edit_" + selected + "_"):
                        memory.pop(name, None)
                st.session_state["narration_task_delete_message"] = message
                st.rerun()
            except (OSError, ValueError) as exc:
                st.error(tr("删除项目失败：") + str(exc))


@st.cache_data(ttl=15, show_spinner=False, max_entries=64)
def _library_counts(identifiers):
    """UI totals only; generation still reads uncached scene records."""
    store = LibraryStore()
    assets = ready = partial = scenes = 0
    for identifier in identifiers:
        with store.database(identifier) as db:
            counts = db.execute("SELECT COUNT(*), COALESCE(SUM(status IN ('ready','partial')),0), COALESCE(SUM(status='partial'),0) FROM assets").fetchone()
            assets += counts[0]
            ready += counts[1]
            partial += counts[2]
            if db.execute("PRAGMA user_version").fetchone()[0] == 2:
                scenes += db.execute("SELECT COUNT(*) FROM content_segments").fetchone()[0]
    return assets, ready, partial, scenes


def _default_font():
    root = Path(config.root_dir) / "resource" / "fonts"
    fonts = sorted([*root.glob("*.ttf"), *root.glob("*.otf")])
    return str(fonts[0]) if fonts else r"C:\Windows\Fonts\msyh.ttc"


def _output_values(prefix):
    value = lambda name, default=None: remembered_widget_value(prefix + "_" + name, default)
    font = value("font")
    if font is None:
        font = _default_font()
        st.session_state.setdefault("narration_widget_memory", {}).setdefault(prefix + "_font", font)
    return {"aspect": value("aspect", "landscape"), "subtitles": bool(value("subtitles", True)), "font": font.strip(),
            "font_size": int(value("font_size_custom_v2", st.session_state.get("narration_font_size_mobile_v1", 72))),
            "mask_source_subtitles": bool(value("mask", False)), "shot_seconds": int(value("shot_seconds", 7)),
            "verify_visual": bool(value("verify_visual", True)), "subtitle_single_line": True,
            "visual_review_concurrency": int(value("visual_review_concurrency", 4)),
            "visual_no_gain_rounds": int(value("visual_no_gain_rounds", 2)),
            "subtitle_width_ratio": int(value("subtitle_width", 70)) / 100,
            "subtitle_min_seconds": float(value("subtitle_min_seconds", 1.0)), "subtitle_allow_short": bool(value("subtitle_allow_short", False))}


@st.fragment
def _render_models(tr, prefix, task_id, saved):
    remember_widget_values(prefix)
    with remembered_expander(tr("模型选择"), key=prefix + "_models_open") as panel:
        if panel.open:
            render_model_choices(tr, prefix)
            models = saved.get("models", {})
            st.caption(tr("项目保存模型：") + " · ".join(
                f"{label}: {models.get(kind, {}).get('model', '未保存')}" for kind, label in (("text", "文案"), ("vision", "视觉"))))
            st.checkbox(tr("本次继续或创建副本使用所选模型"),
                value=remembered_widget_value("resume_current_models_" + task_id, False), key="resume_current_models_" + task_id,
                help=tr("勾选采用本项目选择区的模型、思考等级和后备配置；取消沿用保存快照。接口凭据在模型配置工作区维护。"))
    remember_widget_values(prefix)


@st.fragment
def _render_output_options(tr, prefix):
    remember_widget_values(prefix)
    options = _output_values(prefix)
    with remembered_expander(tr("输出设置"), key=prefix + "_output_open") as panel:
        if not panel.open:
            return options
        aspects = ("landscape", "portrait")
        st.selectbox(tr("画面比例"), aspects, index=aspects.index(options["aspect"]),
            format_func=lambda value: "16:9" if value == "landscape" else "9:16", key=prefix + "_aspect")
        st.checkbox(tr("显示解说字幕"), value=options["subtitles"], key=prefix + "_subtitles")
        st.text_input(tr("字幕字体文件"), value=options["font"], key=prefix + "_font")
        st.number_input(tr("字幕字号"), min_value=1,
            value=options["font_size"],
            step=1, key=prefix + "_font_size_custom_v2")
        st.caption(tr("每次只显示一行字幕，长句按真实时间切换为连续字幕。"))
        st.slider(tr("字幕单行宽度（画面百分比）"), 50, 95,
            round(options["subtitle_width_ratio"] * 100), 5, key=prefix + "_subtitle_width")
        st.number_input(tr("字幕最短阅读目标（秒）"), min_value=.2,
            value=options["subtitle_min_seconds"], step=.1,
            disabled=remembered_widget_value(prefix + "_audio_mode", LONG_MODE) == LEGACY_MODE, key=prefix + "_subtitle_min_seconds")
        st.checkbox(tr("允许未达阅读目标的短字幕继续生成"),
            value=options["subtitle_allow_short"],
            disabled=remembered_widget_value(prefix + "_audio_mode", LONG_MODE) == LEGACY_MODE, key=prefix + "_subtitle_allow_short",
            help=tr("保留字号、单行宽度和原字词时间。目标无法满足时默认暂停并列出短条；勾选表示接受这些短条。"))
        st.caption(tr("大字号下长句会细分为连续字幕，镜头不必随字幕切换；不可拆长词仅该条适配字号。"))
        if remembered_widget_value(prefix + "_audio_mode", LONG_MODE) == LEGACY_MODE:
            st.caption(tr("原模式细分后对应更多配音单元，可能新增TTS调用；长段模式可复用配音和字词时间。"))
        st.checkbox(tr("遮盖原画面字幕"), value=options["mask_source_subtitles"], key=prefix + "_mask")
        st.slider(tr("目标镜头长度（秒）"), 3, 12, options["shot_seconds"], key=prefix + "_shot_seconds")
        st.caption(tr("2.8 使用素材库已保存的镜头观察做文案匹配；有疑问的素材到视频素材库补拉片。"))
        st.number_input(tr("画面审核并发数"), min_value=1, value=options["visual_review_concurrency"], step=1,
                        key=prefix + "_visual_review_concurrency",
                        help=tr("默认4，设为1使用同一流程串行审核。建议4–8；不同文案单元独立请求，每波后统一编排，不改变已付费缓存。"))
        st.number_input(tr("单元连续无改善暂停轮数"), min_value=1, value=options["visual_no_gain_rounds"], step=1,
                        key=prefix+"_visual_no_gain_rounds")
        st.caption(tr("按事件检索镜头，再用高效率模型确认已保存的观察；连续无新增匹配时暂停，继续沿游标查询。"))
        st.caption(tr("旧窗口库请先创建2.8拉片副本；旧项目可创建复用副本选择新库，保留配音和字幕。"))
        st.caption(tr("文案与视觉模型调用不设任务总次数上限。"))
        st.caption(tr("字幕分段由文案生成设置中的高效率模型完成，结果会校验和缓存。"))
    remember_widget_values(prefix)
    return _output_values(prefix)


def _render_project_libraries(tr, prefix):
    store = LibraryStore()
    library_key = "narration_library_choice" if prefix == "narration_create" else prefix + "_library_choice"
    previous = remembered_widget_value(library_key + "_multi")
    if library_key + "_multi" not in st.session_state and previous is not None:
        st.session_state[library_key + "_multi"] = list(previous)
    selected = choose_library(store, tr, key=library_key, multiple=True)
    if st.button(tr("刷新素材统计"), key="refresh_narration_library_counts_" + prefix):
        _library_counts.clear()
    assets, ready, partial, scenes = _library_counts(tuple(selected))
    st.caption(tr("可用素材与场景：") + f"{ready}/{assets} · {scenes}")
    st.caption(tr("多个视频素材库联合检索，使用各库已有分析与原视频路径。"))
    if ready < assets or partial:
        st.info(tr("当前生成使用已保存的成功场景；部分完成素材也可用，失败窗口不会参与检索。"))
    remember_widget_values(prefix)
    return selected, bool(scenes)


def _project_values(text, prefix, saved):
    """Actions read current settings even when a child fragment ran separately."""
    options = {**saved["options"], **_output_values(prefix), **bgm_values(prefix),
               **audio_values(text, prefix), **_motion_values(prefix)}
    return options, voice_values(prefix)


def _render_project_options(tr, text, prefix, task_id, saved):
    render_narration_audio_panel(tr, text, key_prefix=prefix)
    render_narration_bgm(tr, key_prefix=prefix)
    _render_models(tr, prefix, task_id, saved)
    _render_output_options(tr, prefix)
    _render_motion_options(tr, text, prefix)
    remember_widget_values(prefix)


def _initial_project_settings(text, inherited=None):
    """Creation uses explicit defaults or a named source project's public snapshot."""
    if inherited:
        return deepcopy(inherited)
    return {"options": {
                "aspect": "landscape", "subtitles": True, "font": _default_font(), "font_size": 72,
                "mask_source_subtitles": False, "shot_seconds": 7, "verify_visual": True,
                "subtitle_single_line": True, "subtitle_width_ratio": .7, "subtitle_min_seconds": 1.0,
                "subtitle_allow_short": False, "bgm_mode": "none", "bgm_path": "", "bgm_volume": .15,
                "motion_enabled": False, **initial_audio_options(text)},
            "voice": voice_values(prefix=None), "models": resolve_models(historical=True, allow_missing=True)}


def _manuscript_upload_changed(prefix):
    if not st.session_state.get(prefix + "_manuscript_upload"):
        for suffix in ("source_document", "uploaded_hash"):
            st.session_state.pop(prefix + "_" + suffix, None)
            st.session_state.get("narration_widget_memory", {}).pop(prefix + "_" + suffix, None)


def render_project_create(tr):
    st.subheader(tr("新建项目"))
    st.caption(tr("登记完整文案和视频素材库，保存后到制作工作台设置配音、配乐、模型选择、输出和动画。"))
    prefix = "narration_create"
    st.session_state["_narration_settings_prefix"] = prefix
    seed = st.session_state.pop("narration_create_seed", None)
    if seed:
        memory = st.session_state.setdefault("narration_widget_memory", {})
        for mapping in (memory, st.session_state):
            for key in list(mapping):
                if key.startswith(prefix + "_") or key == "narration_library_choice_multi":
                    del mapping[key]
        st.session_state[prefix + "_inherited_settings"] = deepcopy(seed["settings"])
        memory[prefix + "_manuscript"] = seed["text"]
        memory["narration_library_choice_multi"] = library_ids(seed["settings"])
        if seed.get("source"):
            st.session_state[prefix + "_source_document"] = seed["source"]
    upload = st.file_uploader(tr("上传完整文案（TXT／Markdown）"), type=["txt", "md", "markdown"],
        key=prefix + "_manuscript_upload", on_change=_manuscript_upload_changed, args=(prefix,))
    if upload:
        fingerprint = stable_hash(upload.getvalue().hex())
        if st.session_state.get(prefix + "_uploaded_hash") != fingerprint:
            try:
                raw_text = upload.getvalue().decode("utf-8-sig")
                extension = Path(upload.name).suffix.lower()
                if extension in {".md", ".markdown"}:
                    from app.services.manuscript_text import markdown_to_narration
                    text = markdown_to_narration(raw_text)
                else:
                    text = raw_text
                st.session_state[prefix + "_manuscript"] = text
                st.session_state[prefix + "_source_document"] = {"name": upload.name, "extension": extension, "text": raw_text}
                st.session_state[prefix + "_uploaded_hash"] = fingerprint
            except UnicodeDecodeError:
                st.session_state.pop(prefix + "_source_document", None)
                st.session_state.get("narration_widget_memory", {}).pop(prefix + "_source_document", None)
                st.error(tr("请将文案保存为 UTF-8 TXT，或直接粘贴文本。"))
    text_key = prefix + "_manuscript"
    if text_key not in st.session_state:
        st.session_state[text_key] = remembered_widget_value(text_key, "")
    text = st.text_area(tr("完整文案"), value=None, height=350, key=text_key)
    source = remembered_widget_value(prefix + "_source_document")
    if source:
        st.caption(tr("来源文档：") + source["name"])
        if source["extension"] in {".md", ".markdown"}:
            st.caption(tr("Markdown 格式标记已转换为正文。原始文件会随项目保存。"))
    st.caption(tr("文案字数：") + str(len(text)))
    inherited = remembered_widget_value(prefix + "_inherited_settings")
    initial = _initial_project_settings(text, inherited)
    if inherited:
        options = initial["options"]
        st.caption(tr("继承来源项目设置：") + tr(audio_mode_label(audio_mode(options))) +
            f" · {tr('字幕字号')} {options.get('font_size', 40)} · BGM {options.get('bgm_mode', 'none')} · " +
            tr("动画开启" if options.get("motion_enabled") else "动画关闭"))
        if inherited.get("text_hash") != stable_hash(text):
            st.info(tr("原文已改变；进入制作工作台后重新确认手动配音断点和动画绑定，再开始生成。"))
        if st.button(tr("使用新项目默认设置"), key=prefix + "_reset_inherited"):
            st.session_state.pop(prefix + "_inherited_settings", None)
            st.session_state.get("narration_widget_memory", {}).pop(prefix + "_inherited_settings", None)
            st.rerun()
    else:
        st.caption(tr("初始配置：长段配音、本地对齐开启、72号单行字幕；BGM与解释动画关闭。可在制作工作台调整。"))
    selected_libraries, _ = _render_project_libraries(tr, prefix)
    busy = runner.library_busy(selected_libraries)
    if st.button(tr("创建项目"), type="primary", disabled=busy or not selected_libraries or not text.strip(), key="create_narration_project"):
        try:
            task_id = create_task(selected_libraries, text, initial["options"], initial["voice"], source=source,
                                  model_profile=initial.get("models"), draft=True)
            st.session_state.setdefault("narration_widget_memory", {})["resume_current_models_" + task_id] = not bool(inherited)
            _navigate("narration_progress", task_id)
        except (OSError, ValueError) as exc:
            st.error(str(exc))


def _project_label(task):
    directory = task_directory(task["id"])
    title = ""
    try:
        title = read_json(directory / "source.json", default={}).get("name", "")
        if not title:
            with (directory / "input.txt").open("r", encoding="utf-8", newline="") as source:
                title = next((line.strip() for line in source if line.strip()), "")
    except (OSError, ValueError):
        pass
    status = {"pending": "待开始", "running": "运行中", "paused": "已暂停", "failed": "失败", "complete": "已完成"}.get(task.get("state"), "")
    return task["id"][:8] + " · " + title[:48] + " · " + status + " · " + task.get("phase", "")


def _select_project():
    st.session_state["active_narration_task"] = st.session_state["narration_project_choice"]


def _project_busy(task_id, saved):
    return (runner.running("narration:" + task_id) or runner.library_busy(library_ids(saved))
            or runner.library_busy(bgm_libraries(saved["options"]), prefix="audio-library:"))


@st.fragment
def _render_project_original(tr, task_id, text, saved_settings):
    remember_widget_values()
    original_key = "narration_original_open_" + task_id
    with remembered_expander(tr("项目原文"), key=original_key) as panel:
        if not panel.open:
            return
        st.text_area(tr("已保存的完整文案"), value=text, height=220, disabled=True, key="narration_original_" + task_id)
        if st.button(tr("载入原文和设置到项目管理"), key="load_narration_text_" + task_id):
            source_info = read_json(task_directory(task_id) / "source.json", default={})
            extension = source_info.get("extension")
            source = None
            if extension in {".txt", ".md", ".markdown"}:
                path = task_directory(task_id) / ("original" + extension)
                if path.is_file():
                    with path.open("r", encoding="utf-8", newline="") as original:
                        source = {**source_info, "text": original.read()}
            st.session_state["narration_create_seed"] = {"settings": saved_settings, "text": text, "source": source}
            _navigate("narration_create")


@st.fragment
def _render_project_forks(tr, task_id, text, saved_settings, missing_libraries, missing_music):
    prefix = "narration_edit_" + task_id
    remember_widget_values(prefix)
    task_busy = _project_busy(task_id, saved_settings)
    use_current_models = bool(remembered_widget_value("resume_current_models_" + task_id, False))
    settings_key = prefix + "_settings_open"
    with remembered_expander(tr("复用副本"), key=settings_key) as panel:
        if not panel.open:
            return
        st.caption(tr("副本动作读取上方五项设置和视频素材库草稿。普通推进继续使用原项目保存设置。"))
        selected_libraries = list(remembered_widget_value(prefix + "_library_choice_multi", library_ids(saved_settings)))
        options, voice_profile = _project_values(text, prefix, saved_settings)
        has_scenes = bool(_library_counts(tuple(selected_libraries))[3])
        models = workspace_models(prefix)
        music_options = {key: value for key, value in options.items() if key.startswith("bgm_")}
        motion_settings = {key: value for key, value in options.items() if key.startswith("motion_")}
        music_busy = runner.library_busy(bgm_libraries(options), prefix="audio-library:")
        busy = runner.library_busy(selected_libraries) or music_busy
        if not task_busy:
            st.caption(tr("仅更新字幕样式会沿用原任务视频素材库、音色、分块、配音方式和输出比例；长段模式可复用音轨及字词时间。"))
            if st.button(tr("仅更新字幕样式，创建副本合成"), disabled=task_busy or bool(missing_libraries or missing_music), key="style_fork_narration_" + task_id):
                try:
                    recorded_voice = restore_voice(saved_settings["voice"], voice_profile, profile_id=saved_settings.get("models", {}).get("_bindings", {}).get("tts"))
                    validate_voice(recorded_voice)
                    style_options = {**saved_settings["options"], "font": options["font"], "font_size": options["font_size"],
                                     "subtitle_single_line": True, "subtitle_width_ratio": options["subtitle_width_ratio"],
                                     "subtitle_min_seconds": options["subtitle_min_seconds"], "subtitle_allow_short": options["subtitle_allow_short"]}
                    child_id = fork_task(task_id, style_options, recorded_voice, models, use_current_models)
                    st.session_state["active_narration_task"] = child_id
                    _start_task(child_id, models, recorded_voice)
                    _navigate("narration_progress", child_id)
                except ValueError as exc:
                    st.error(str(exc))
            st.caption(tr("仅更新解释动画使用上方动画设置与禁用选择，沿用历史任务完整配音、字幕和视频素材库，从3. 寻找画面继续。"))
            if st.button(tr("仅更新解释动画，复用配音并创建副本"), disabled=task_busy or bool(missing_libraries or missing_music),
                         key="motion_fork_narration_" + task_id):
                try:
                    animation_options = {key: value for key, value in saved_settings["options"].items() if not key.startswith("motion_")}
                    animation_options.update(motion_settings)
                    if animation_options["motion_enabled"]:
                        plan = read_json(task_directory(task_id) / "motion" / "plan.json", default={})
                        edits = read_json(task_directory(task_id) / "motion" / "edits.json", default={})
                        disabled_ids = remembered_widget_value(
                            "narration_motion_disabled_" + task_id + plan.get("key", "")[:12], edits.get("disabled_ids", []))
                        valid_ids = {row["id"] for row in plan.get("animations", [])}
                        animation_options["motion_disabled_ids"] = [identifier for identifier in disabled_ids if identifier in valid_ids]
                    manual_motion_entries(text, animation_options, complete=True)
                    child_id = fork_task(task_id, animation_options, dict(saved_settings["voice"]), models,
                                         use_current_models, reuse_audio=True)
                    st.session_state["active_narration_task"] = child_id
                    _start_task(child_id, models, dict(saved_settings["voice"]), use_current_models, start_stage="visual")
                    _navigate("narration_progress", child_id)
                except (OSError, ValueError) as exc:
                    st.error(str(exc))
            st.caption(tr("使用该历史任务原文、上方多选的视频素材库和输出设置创建新任务；有效的规划、配音及对齐缓存会复用。"))
            if st.button(tr("仅更新 BGM，复用配音并创建副本"), disabled=task_busy or music_busy or bool(missing_libraries), key="bgm_fork_narration_" + task_id):
                try:
                    recorded_voice = restore_voice(saved_settings["voice"], voice_profile, profile_id=saved_settings.get("models", {}).get("_bindings", {}).get("tts"))
                    validate_voice(recorded_voice)
                    bgm_fork_options = {key: value for key, value in saved_settings["options"].items() if not key.startswith("bgm_")}
                    bgm_fork_options.update(music_options)
                    child_id = fork_task(task_id, bgm_fork_options, recorded_voice, models, use_current_models)
                    st.session_state["active_narration_task"] = child_id
                    _start_task(child_id, models, recorded_voice)
                    _navigate("narration_progress", child_id)
                except ValueError as exc:
                    st.error(str(exc))
            st.caption(tr("更换视频素材库会重新匹配画面，原任务及成片保留；原视频素材库的分析无需重做。"))
            keep_audio = st.checkbox(tr("沿用历史任务配音设置"), value=remembered_widget_value("fork_keep_audio_" + task_id, True), key="fork_keep_audio_" + task_id,
                help=tr("沿用原音色、配音方式、分块和对齐设置，减少重新配音；取消后使用上方当前配音配置。"))
            st.caption(tr("模型配置由“模型选择”中的“本次继续或创建副本使用所选模型”决定。"))
            st.caption(tr("配音文字、音色、分块或原模式字幕单元改变时，需要的新音频会产生配音调用。"))
            if st.button(tr("创建复用副本并生成"), disabled=task_busy or busy or not has_scenes, key="fork_narration_" + task_id):
                try:
                    fork_voice = restore_voice(saved_settings["voice"], voice_profile, profile_id=saved_settings.get("models", {}).get("_bindings", {}).get("tts")) if keep_audio else voice_profile
                    if not keep_audio:
                        check_model_choices(prefix, ["tts"])
                    validate_voice(fork_voice)
                    fork_options = dict(options)
                    if keep_audio:
                        for key in ("tts_block_chars", "tts_chunking", "tts_break_after", "tts_break_text_hash",
                                    "alignment_enabled", "alignment_python", "alignment_model", "alignment_language"):
                            if key in saved_settings["options"]:
                                fork_options[key] = saved_settings["options"][key]
                            else:
                                fork_options.pop(key, None)
                        fork_options["audio_mode"] = audio_mode(saved_settings["options"])
                    manual_motion_entries(text, fork_options, complete=True)
                    child_id = fork_task(task_id, fork_options, fork_voice, models, use_current_models, selected_libraries=selected_libraries)
                    st.session_state["active_narration_task"] = child_id
                    _start_task(child_id, models, fork_voice)
                    _navigate("narration_progress", child_id)
                except ValueError as exc:
                    st.error(str(exc))



@st.fragment
def _render_project_continue(tr, task_id, saved_settings, missing_libraries, missing_music):
    remember_widget_values()
    state = read_task(task_id)
    if not state:
        return
    saved_settings = read_json(task_directory(task_id) / "settings.json")
    edit_status = project_edit_status(task_id)
    prefix = "narration_edit_" + task_id
    if edit_status["unstarted"]:
        selected_libraries = list(remembered_widget_value(prefix + "_library_choice_multi", library_ids(saved_settings)))
        with (task_directory(task_id) / "input.txt").open("r", encoding="utf-8", newline="") as source:
            text = source.read()
        options, voice_profile = _project_values(text, prefix, saved_settings)
        models = workspace_models(prefix)
        use_current = bool(remembered_widget_value("resume_current_models_" + task_id, False))
        busy = (not edit_status["editable"] or runner.library_busy(selected_libraries)
                or runner.library_busy(bgm_libraries(options), prefix="audio-library:"))
        has_scenes = bool(selected_libraries and _library_counts(tuple(selected_libraries))[3])
        if not has_scenes:
            st.caption(tr("可以先保存项目设置；开始生成前，请在视频素材库完成素材分析。"))
        save_col, start_col = st.columns(2)
        save = save_col.button(tr("保存项目设置"), disabled=busy or not selected_libraries, key="save_narration_settings_" + task_id)
        start = start_col.button(tr("开始生成"), type="primary", disabled=busy or not has_scenes, key="resume_narration_" + task_id)
        if save or start:
            try:
                required = {"text", "text_fast", *(["vision"] if options["verify_visual"] else [])} if start else set()
                selected_models = models if use_current or not saved_settings.get("models") else restore_models(
                    saved_settings["models"], models, required_kinds=required)
                selected_models = deepcopy(selected_models)
                bindings = dict(selected_models.get("_bindings", {}))
                current_bindings = selected_bindings(prefix)
                for role in ("tts", "alignment"):
                    bindings[role] = current_bindings.get(role, "")
                selected_models["_bindings"] = bindings
                if start:
                    check_model_choices(prefix, ["tts", *(required if use_current else []),
                        *(["alignment"] if options.get("alignment_enabled", False) else []),
                        *(["text_backup", "vision_backup"] if use_current else [])])
                    validate_voice(voice_profile)
                    if options["subtitles"] and not Path(options["font"]).is_file():
                        raise ValueError(tr("请选择有效字幕字体文件。"))
                    gateway = ModelGateway(selected_models)
                    for kind in required:
                        gateway.require(kind)
                update_project_settings(task_id, selected_libraries, options, voice_profile, selected_models, for_start=start)
                if start:
                    _start_task(task_id, selected_models, voice_profile)
                else:
                    st.session_state["narration_project_notice"] = tr("项目设置已保存。")
                remember_widget_values(prefix)
                st.rerun()
            except (OSError, ValueError) as exc:
                st.error(str(exc))
        return
    task_libraries = library_ids(saved_settings)
    task_busy = _project_busy(task_id, saved_settings)
    voice_profile = voice_values("narration_edit_" + task_id)
    try:
        models = workspace_models(prefix)
    except ValueError:
        models = {"proxy": dict(config.proxy)}
    use_current_models = bool(remembered_widget_value("resume_current_models_" + task_id, False))
    if not task_busy:
        st.caption(tr("推进使用项目保存的视频素材库、原文、音色和输出设置。"))
        original_has_scenes = not missing_libraries and bool(_library_counts(tuple(task_libraries))[3])
        if not original_has_scenes and not missing_libraries:
            st.caption(tr("项目素材库尚无可用分析场景；请先在视频素材库完成分析。已有完整时间轴可使用第5步仅合成。"))
        def continue_from(stage="auto"):
            try:
                if stage in {"auto", "tts"}:
                    restored_voice = restore_voice(saved_settings["voice"], voice_profile, profile_id=saved_settings.get("models", {}).get("_bindings", {}).get("tts"))
                    validate_voice(restored_voice)
                else:
                    restored_voice = dict(saved_settings["voice"])
                _start_task(task_id, models, restored_voice, use_current_models, start_stage=stage)
                st.rerun()
            except ValueError as exc:
                st.error(str(exc))
        action = "开始生成" if state["state"] == "pending" else "复用缓存重新合成" if state["state"] == "complete" else "继续生成"
        if st.button(tr(action), type="primary", disabled=bool(missing_libraries or missing_music) or not original_has_scenes, key="resume_narration_" + task_id):
            continue_from()
        stage_key = "narration_resume_stages_" + task_id
        with remembered_expander(tr("按阶段继续"), key=stage_key) as panel:
            if not panel.open:
                return
            st.caption(tr("从所选阶段执行到成片，复用已保存结果；有效缓存仍会复用。"))
            st.caption(tr("第5步使用已有配音、字幕、镜头、动画缓存和完成的配乐，无需TTS、模型或本地对齐凭据。"))
            if audio_mode(saved_settings["options"]) == LEGACY_MODE:
                st.caption(tr("原模式的字幕处理复用配音时记录的字幕时间，不重新进行本地语音对齐。"))
            try:
                stages = resume_stage_status(task_id)
            except (OSError, ValueError) as exc:
                st.warning(tr("无法读取前序结果：") + str(exc))
                stages = {}
            entries = list(RESUME_STAGES.items())
            for offset in range(0, len(entries), 2):
                columns = st.columns(2)
                for column, (stage, label) in zip(columns, entries[offset:offset + 2]):
                    status = stages.get(stage, {"ready": False, "reason": "前序结果不可读。"})
                    disabled = bool(missing_libraries) or (bool(missing_music) and stage != "render") or not status["ready"]
                    with column:
                        if st.button(tr(label), disabled=disabled, key="resume_narration_" + stage + "_" + task_id,
                                     help=tr(status["reason"]), width="stretch"):
                            continue_from(stage)
                        if not status["ready"]:
                            st.caption(tr(status["reason"]))


@st.fragment
def _render_project_results(tr, task_id):
    remember_widget_values()
    results_key = "narration_results_open_" + task_id
    with remembered_expander(tr("配音与字幕成果"), key=results_key) as panel:
        if panel.open:
            directory = task_directory(task_id)
            audio = directory / "narration.wav"
            if audio.is_file() and st.button(tr("试听完整配音"), key="project_audio_preview_" + task_id):
                st.audio(str(audio), format="audio/wav")
            subtitles = directory / "subtitles.srt"
            if subtitles.is_file():
                st.download_button(tr("下载字幕"), subtitles.read_bytes(), file_name="subtitles.srt", mime="text/plain", key="project_subtitle_download_" + task_id)
    directory = task_directory(task_id)
    timeline_file, matches_file = directory / "timeline.json", directory / "visual_matches.json"
    if timeline_file.is_file() or matches_file.is_file():
        matches_key = "narration_matches_open_" + task_id
        with remembered_expander(tr("画面匹配详情"), key=matches_key) as panel:
            if not panel.open:
                return
            timeline = read_json(timeline_file, default={})
            report = read_json(matches_file, default={})
            plan = read_json(directory / "visual_plan.json", default={})
            document = timeline if timeline.get("segments") else read_json(directory / "document_plan.json", default={})
            segment_map = {item["id"]: item for item in document.get("segments", [])}
            matching = timeline.get("visual_policy", {}).get("matching", {})

            def original(unit):
                segment = segment_map.get(unit.get("segment_id"), {})
                offset = segment.get("text_start", 0)
                return segment.get("text", "")[unit.get("text_start", offset)-offset:unit.get("text_end", offset)-offset]

            def paged(rows, label, suffix):
                pages = max(1, (len(rows)+29)//30)
                key = "narration_matches_" + suffix + "_" + task_id
                st.session_state[key] = min(max(1, int(remembered_widget_value(key, 1))), pages)
                page = st.number_input(tr(label), 1, pages, step=1, key=key)
                st.caption(f"{len(rows)} · {page}/{pages}")
                return rows[(page-1)*30:page*30]

            if report:
                timing_labels = {"aligned_words": "字词对齐", "text_estimate": "按文字估算", "subtitle_interpolation": "按已有字幕插值"}
                st.caption(tr("画面时间来源：") + tr(timing_labels.get(report.get("timing_method"), "时间为估算")))
                st.caption(tr("已排除名单／不明画面：") + str(report.get("excluded_windows", 0)) + " / " + str(report.get("excluded_ranges", 0)))
                if report.get("state") != "complete":
                    st.info(tr("本次画面匹配尚未完成；已有配音、字幕和旧成片保留。缺口会随补选更新。"))
                gap_units = {unit["id"]: unit for unit in plan.get("units", [])}
                gaps = report.get("gaps", [])
                if gaps:
                    st.warning(tr("尚缺内容匹配且未用的画面：") + str(len(gaps)))
                    for gap in paged(gaps, "缺口页码", "gaps"):
                        identifier = gap["visual_unit_id"]
                        unit = gap_units.get(identifier, {})
                        st.markdown(f"**{identifier} · {gap['start']:.1f}–{gap['end']:.1f}s · {gap['missing_seconds']:.2f}s**")
                        st.write(original(unit))
                        st.caption(gap.get("detail",gap.get("reason", "")))
                        facts = unit.get("requirement", {}).get("required_facts", [])
                        if facts:
                            st.caption(tr("需要对应：") + "；".join(facts))
                        stats = report.get("units", {}).get(identifier, {})
                        st.caption(tr("已检查镜头范围：") + str(stats.get("examined",stats.get("examined_windows", 0))) + " · " +
                                   "；".join(f"{name}: {count}" for name, count in stats.get("reasons", {}).items()))
                st.download_button(tr("下载画面匹配记录"), matches_file.read_bytes(), file_name="visual_matches.json",
                                   mime="application/json", key="visual_matches_download_" + task_id)
            if not timeline:
                return
            if report and matching.get("plan_key") != report.get("plan_key"):
                st.caption(tr("以下显示的是保留的旧时间线，本次新匹配尚未形成完整时间线。"))
            units = {unit["id"]: unit for unit in matching.get("units", [])}
            approved = timeline.get("visual_policy", {}).get("approved", [])
            choices = {"all": "全部镜头", "supporting": "辅助匹配", "unchecked": "未视觉复核", "weak": "匹配偏弱（含旧版）"}
            mode = st.selectbox(tr("匹配筛选"), list(choices), format_func=lambda item: tr(choices[item]), key="match_filter_" + task_id)
            rows = []
            for shot in timeline.get("shots", []):
                review = next((item for item in approved if item.get("visual_unit_id") == shot.get("visual_unit_id")
                    and item.get("asset_id") == shot["asset_id"] and item.get("scene_id") == shot["scene_id"]
                    and item["source_start"] <= shot["source_start"]+.000001 and item["source_end"] >= shot["source_end"]-.000001), {})
                if mode == "all" or (mode == "supporting" and review.get("relation") == "supporting") or \
                        (mode == "unchecked" and review.get("visual_checked") is False) or (mode == "weak" and shot.get("weak_match")):
                    rows.append((shot, review))
            for shot, review in paged(rows, "镜头详情页码", "shots"):
                unit = units.get(shot.get("visual_unit_id"), {})
                st.markdown(f"**{shot.get('visual_unit_id') or shot['segment_id']} · {shot['timeline_start']:.1f}–{shot['timeline_end']:.1f}s**")
                st.write(original(unit) if unit else segment_map.get(shot["segment_id"], {}).get("text", ""))
                st.caption(f"{Path(shot['video_path']).name} · {shot['source_start']:.1f}–{shot['source_end']:.1f}s")
                if review:
                    st.caption(tr({"direct": "直接对应", "supporting": "辅助匹配"}.get(review.get("relation"), "旧版匹配")) +
                               (" · " + tr("未进行本次视觉复核") if review.get("visual_checked") is False else ""))
                    st.write("；".join(review.get("evidence", [])))
                    if review.get("subtitle_quotes"):
                        st.caption(tr("范围字幕依据：") + "；".join(review["subtitle_quotes"]))
                st.write(shot["reason"])
            st.download_button(tr("下载剪辑时间线"), timeline_file.read_bytes(), file_name="timeline.json", mime="application/json", key="timeline_download_" + task_id)


def _project_choice(tr, tasks):
    identifiers = [task["id"] for task in tasks]
    labels = {task["id"]: _project_label(task) for task in tasks}
    current = st.session_state.get("active_narration_task")
    if current not in identifiers:
        current = identifiers[0]
        st.session_state["active_narration_task"] = current
    if st.session_state.get("narration_project_choice") != current:
        st.session_state["narration_project_choice"] = current
    task_id = st.selectbox(tr("选择项目"), identifiers, index=None, format_func=labels.get,
        key="narration_project_choice", on_change=_select_project)
    if not task_id:
        return
    return task_id


def render_project_management(tr):
    st.header(tr("项目管理"))
    tasks = list_tasks()
    for key in ("narration_task_delete_message", "narration_project_notice"):
        notice = st.session_state.pop(key, None)
        if notice:
            st.info(notice)
    if tasks:
        task_id = _project_choice(tr, tasks)
        if task_id:
            state = read_task(task_id)
            try:
                saved = read_json(task_directory(task_id) / "settings.json")
                with (task_directory(task_id) / "input.txt").open("r", encoding="utf-8", newline="") as source:
                    text = source.read()
                _seed_project_widgets("narration_edit_" + task_id, saved)
                st.caption(task_id + " · " + state.get("phase", "") + " · " + state.get("message", ""))
                if st.button(tr("进入制作工作台"), type="primary", key="manage_enter_" + task_id):
                    _navigate("narration_progress", task_id)
                _render_project_original(tr, task_id, text, saved)
                _render_task_manager(tr, task_id, state)
            except (OSError, ValueError) as exc:
                st.error(str(exc))
                _render_task_manager(tr, task_id, state)
    else:
        st.info(tr("还没有项目，请登记完整文案和视频素材库。"))
    key = "narration_manage_new_open"
    if st.session_state.get("narration_create_seed"):
        st.session_state[key] = True
    with remembered_expander(tr("新建项目"), key=key, default=not tasks) as panel:
        if panel.open:
            render_project_create(tr)


def render_project_progress(tr):
    st.header(tr("制作工作台"))
    if st.button(tr("返回项目管理"), key="workbench_return_management"):
        _navigate("narration_create")
    for key in ("narration_task_delete_message", "narration_project_notice"):
        notice = st.session_state.pop(key, None)
        if notice:
            st.info(notice)
    tasks = list_tasks()
    if not tasks:
        st.info(tr("还没有项目，先在项目管理页面登记完整文案和视频素材库。"))
        if st.button(tr("前往项目管理"), key="empty_project_create"):
            _navigate("narration_create")
        return
    task_id = _project_choice(tr, tasks)
    if not task_id:
        return
    state = read_task(task_id)
    try:
        saved_settings = read_json(task_directory(task_id) / "settings.json")
        with (task_directory(task_id) / "input.txt").open("r", encoding="utf-8", newline="") as manuscript:
            text = manuscript.read()
        task_libraries = library_ids(saved_settings)
    except (OSError, ValueError) as exc:
        st.error(tr("无法读取项目设置或原文：") + str(exc))
        _render_task_manager(tr, task_id, state)
        return
    store = LibraryStore()
    library_names = {item["id"]: item["name"] for item in store.list_libraries()}
    music_names = {item["id"]: item["name"] for item in AudioLibraryStore().list_libraries()}
    missing_libraries = [identifier for identifier in task_libraries if identifier not in library_names]
    missing_music = [identifier for identifier in bgm_libraries(saved_settings["options"]) if identifier not in music_names]
    task_busy = _project_busy(task_id, saved_settings)
    prefix = "narration_edit_" + task_id
    _seed_project_widgets(prefix, saved_settings)
    st.caption(tr("项目视频素材库：") + "、".join(library_names.get(identifier, identifier[:8] + "（已删除）") for identifier in task_libraries))
    st.caption(tr("配音方式：") + tr(audio_mode_label(audio_mode(saved_settings["options"]))))
    if saved_settings.get("parent_task_id"):
        st.caption(tr("来源项目：") + saved_settings["parent_task_id"][:8])
    if missing_libraries:
        st.warning(tr("项目的视频素材库已删除；可选择现存素材库创建复用副本。"))
    if missing_music:
        st.warning(tr("项目的音频素材库已删除；已完成BGM仍可从第5步复用，可修改配乐创建副本。"))
    if task_busy and not runner.running("narration:" + task_id):
        st.info(tr("关联素材库正在被其他任务使用，请等待结束后继续制作。"))
    _render_task_status(task_id, tr)
    if state.get("subtitle_timing") == "text_estimate":
        st.caption(tr("此项目已关闭本地对齐，字幕时间为文字估算。"))
    render_bgm_plan(task_directory(task_id), task_id, tr)
    _render_motion_plan(task_id, tr)
    _render_project_original(tr, task_id, text, saved_settings)
    st.caption(tr(project_edit_status(task_id)["reason"]))
    _render_project_libraries(tr, prefix)
    _render_project_options(tr, text, prefix, task_id, saved_settings)
    _render_project_continue(tr, task_id, saved_settings, missing_libraries, missing_music)
    _render_project_forks(tr, task_id, text, saved_settings, missing_libraries, missing_music)
    _render_project_results(tr, task_id)
