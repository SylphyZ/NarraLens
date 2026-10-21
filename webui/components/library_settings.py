"""Local directory import and reusable material-library management."""

from pathlib import Path
import copy
import json
import time

import pandas as pd
import streamlit as st

from app.config import config
from app.models.library import ImportMapping
from app.services.ass_subtitle import parse_ass_events, plain_ass_lines, select_chinese_ass
from app.services.library import runner
from app.services.library.frame_extractor import inspect_video
from app.services.library.gateway import ModelGateway
from webui.components.model_settings import seed_model_choices, render_model_choices, selected_models, selected_bindings, check_model_choices
from app.services.library.indexer import VIDEO_EXTENSIONS, SUBTITLE_EXTENSIONS, scan_directory, suggest_mappings, import_materials, build_library, audit_subtitles, restore_subtitle_text
from app.services.library.store import LibraryStore, _json_access, library_ids, read_json, write_json, stable_hash
from app.services.subtitle_text import decode_subtitle_bytes
from webui.utils.file_utils import folder_input, remembered_expander, remember_widget_values, probe_file


def _remember_library_selection(store, widget_key, multiple, tr):
    selected = st.session_state.get(widget_key)
    if selected is None:
        return
    selected = list(selected) if multiple else selected
    if not multiple and st.session_state.get("active_library_id") != selected:
        st.session_state["active_library_id"] = selected
    path = store.root / "ui-selection.json"
    try:
        with _json_access(path):
            try:
                saved = read_json(path, default={})
            except ValueError:
                saved = {}
            if not isinstance(saved, dict):
                saved = {}
            if saved.get(widget_key) != selected:
                saved[widget_key] = selected
                write_json(path, saved)
    except OSError as exc:
        st.warning(tr("无法保存视频素材库选择：") + str(exc))


def choose_library(store, tr, key="library_choice", multiple=False):
    libraries = store.list_libraries()
    if not libraries:
        st.info(tr("请先在视频素材库页面创建一个视频素材库。"))
        return [] if multiple else None
    identifiers = [item["id"] for item in libraries]
    labels = {item["id"]: item["name"] + (" · 旧窗口库" if store.schema(item["id"]) != 2 else " · 内容拉片") for item in libraries}
    try:
        saved = read_json(store.root / "ui-selection.json", default={})
    except (OSError, ValueError) as exc:
        st.warning(tr("无法读取上次视频素材库选择：") + str(exc))
        saved = {}
    if not isinstance(saved, dict):
        saved = {}
    previous = st.session_state.get("active_library_id")
    if multiple:
        widget_key = key + "_multi"
        fallback = previous if previous in identifiers else saved.get("library_choice")
        raw = st.session_state.get(widget_key, saved.get(widget_key, [fallback if fallback in identifiers else identifiers[0]]))
        valid = list(dict.fromkeys(identifier for identifier in raw if isinstance(identifier, str) and identifier in labels)) if isinstance(raw, (list, tuple)) else []
        if widget_key not in st.session_state or st.session_state[widget_key] != valid:
            st.session_state[widget_key] = valid
        selected = st.multiselect(tr("选择视频素材库（可多选）"), identifiers,
            default=None, format_func=lambda identifier: labels[identifier], key=widget_key,
            on_change=_remember_library_selection, args=(store, widget_key, True, tr))
        _remember_library_selection(store, widget_key, True, tr)
        return selected
    current = next((value for value in (previous, st.session_state.get(key), saved.get(key))
                    if isinstance(value, str) and value in labels), identifiers[0])
    if st.session_state.get(key) != current:
        st.session_state[key] = current
    selected = st.selectbox(tr("选择视频素材库"), identifiers, format_func=lambda identifier: labels[identifier], key=key,
                            on_change=_remember_library_selection, args=(store, key, False, tr))
    _remember_library_selection(store, key, False, tr)
    return selected


def _launch_library_job(store, library_id, kind, callback):
    key = "library:" + library_id
    def finish(state, error):
        previous = store.job(library_id, kind) or {}
        message = str(error)
        if kind == "build" and previous.get("message"):
            message += "\n" + previous["message"] + "\n已保存的镜头观察和缓存仍保留，可继续处理缺项。"
        store.job(library_id, kind, state=state, progress=previous.get("progress", 0), message=message)
    def work(stop):
        try:
            store.job(library_id, kind, state="running", progress=0, message="任务准备开始")
            callback(stop)
        except runner.JobCancelled as exc:
            finish("paused", exc)
        except Exception as exc:
            finish("failed", exc)
    if not runner.start(key, work):
        raise ValueError("视频素材库或关联生成任务正在运行，请等待任务结束。")
    st.session_state["library_observed_state_" + library_id] = "running"


@st.fragment(run_every="2s")
def _job_status(library_id, tr):
    store = LibraryStore()
    key = "library:" + library_id
    jobs = [job for job in (store.job(library_id, kind) for kind in ("import", "build", "subtitle", "files", "agent")) if job]
    if not jobs:
        return
    job = max(jobs, key=lambda item: item["updated"])
    observed_key = "library_observed_state_" + library_id
    previous_state = st.session_state.get(observed_key)
    observed_state = "running" if runner.running(key) else job["state"]
    st.session_state[observed_key] = observed_state
    if previous_state == "running" and observed_state != "running":
        st.rerun(scope="app")
    st.progress(max(0, min(1, job["progress"] / 100)))
    st.caption(tr("最近状态更新：") + time.strftime("%H:%M:%S", time.localtime(job["updated"])))
    if job["state"] == "failed":
        st.error(job["message"])
    elif job["state"] == "complete":
        st.success(job["message"])
    else:
        st.info(job["message"])
    if runner.running(key):
        if st.button(tr("停止当前任务"), key="cancel_library_" + library_id):
            runner.cancel(key)
            st.info(tr("当前请求结束后停止，已完成结果会保留。"))
    elif job["state"] == "running":
        st.warning(tr("上次运行已中断，可以重新点击继续。"))
    errors = store.analysis_errors(library_id)
    log_path = store.directory(library_id) / "errors.jsonl"
    if errors or log_path.is_file():
        with remembered_expander(tr("分析错误详情"), key="library_errors_" + library_id) as panel:
            if not panel.open:
                return
            if not errors:
                st.caption(tr("当前没有未解决错误，历史错误日志仍保留。"))
            for error in errors[:20]:
                st.error(error["message"])
                details = error["details"]
                if details.get("response_excerpt"):
                    st.code(details["response_excerpt"], language=None)
            if log_path.is_file():
                st.download_button(tr("下载分析错误日志"), log_path.read_bytes(), file_name="analysis-errors.jsonl", mime="application/jsonl", key="library_error_log_" + library_id)


def _subtitle_settings(subtitles, defaults, tr, prefix):
    settings = {"chinese_only": bool(defaults.get("chinese_only", True)), "styles": defaults.get("styles", []), "line_mode": defaults.get("line_mode", "auto")}
    ass_files = [path for path in subtitles if Path(path).suffix.lower() in {".ass", ".ssa"}]
    if not ass_files:
        return settings
    settings = st.session_state.setdefault(prefix + "_subtitle_draft", settings)
    with remembered_expander(tr("中文字幕筛选与预览"), key=prefix + "_subtitle_preview") as panel:
        if not panel.open:
            return settings
        sample_path = st.selectbox(tr("预览字幕文件"), ass_files, format_func=lambda value: Path(value).name, key=prefix + "sample")
        decoded = decode_subtitle_bytes(Path(sample_path).read_bytes())
        events = parse_ass_events(decoded.text)
        styles = sorted({event.style for event in events})
        settings["chinese_only"] = st.checkbox(tr("Keep Only Chinese Subtitles"), value=settings["chinese_only"], key=prefix + "chinese")
        if settings["chinese_only"]:
            settings["styles"] = st.multiselect(tr("Chinese ASS Styles"), styles, default=[value for value in settings["styles"] if value in styles], key=prefix + "styles")
            modes = ("auto", "first", "last")
            labels = {"auto": "Chinese Line Auto", "first": "Chinese Line First", "last": "Chinese Line Last"}
            settings["line_mode"] = st.selectbox(tr("Chinese Subtitle Line"), modes,
                index=modes.index(settings["line_mode"]) if settings["line_mode"] in modes else 0,
                format_func=lambda value: tr(labels[value]), key=prefix + "lines")
            result = select_chinese_ass(decoded.text, settings["styles"], settings["line_mode"])
            left, right = st.columns(2)
            for column, label, rows in ((left, "Subtitles Before Selection", events), (right, "Subtitles After Selection", result.selected)):
                with column:
                    st.markdown(f"**{tr(label)}**")
                    st.code("\n\n".join(f"[{event.style}] {event.start} --> {event.end}\n" + "\n".join(plain_ass_lines(event.text)) for event in rows[:8]), language=None)
            if result.uncertain_events:
                st.warning(tr("Chinese Subtitle Selection Uncertain").format(count=result.uncertain_events))
            st.caption(tr("筛选设置会用于本次导入的全部 ASS 字幕，请确认样式和行顺序一致。"))
    st.session_state[prefix + "_subtitle_draft"] = copy.deepcopy(settings)
    return settings


_KINDS = {"story": "剧情画面", "credits": "片尾名单", "opening": "片头", "black": "黑屏", "unknown": "不确定"}
_EVIDENCE = {"visual": "可见画面", "subtitle": "字幕说明", "context": "上下文推断"}
_FILE_STATES = {"available": "可读取", "missing": "文件不存在", "denied": "无权限", "unreachable": "网络／共享路径不可访问",
                "timeout": "检查超时", "unconfirmed": "结果未确认", "unchecked": "未检查"}


def _range_label(item):
    return f"第{item['episode']}集 · {item['start']:.1f}–{item['end']:.1f}s · {item.get('summary', '')[:36]}"


def _rows_editor(rows, columns, key, busy):
    return st.data_editor(pd.DataFrame(rows, columns=list(columns)), column_config=columns,
                          hide_index=True, num_rows="dynamic", disabled=busy, width="stretch", key=key).to_dict("records")


def _set_draft(prefix, value):
    st.session_state[prefix + "_draft"] = copy.deepcopy(value)
    st.session_state[prefix + "_epoch"] = st.session_state.get(prefix + "_epoch", 0) + 1


def _advanced_json(prefix, draft, validate, tr, busy):
    with remembered_expander(tr("高级 JSON（导入替换表单草稿）"), key=prefix + "_advanced") as panel:
        if not panel.open:
            return
        payload = json.dumps(draft, ensure_ascii=False, indent=2)
        st.caption(tr("导入只替换草稿，核对表单后点击保存。请先保存表单中尚未提交的修改。编号和时间不可手工修改。"))
        raw = st.text_area(tr("可编辑字段 JSON"), value=payload, height=180,
                           key=prefix + "_json_" + str(st.session_state.get(prefix + "_epoch", 0)), disabled=busy)
        st.download_button(tr("导出已载入草稿"), payload, file_name="library-draft.json",
                           mime="application/json", key=prefix + "_download")
        if st.button(tr("校验并载入表单"), key=prefix + "_load_json", disabled=busy):
            try:
                value = json.loads(raw)
                if not isinstance(value, dict) or set(value) != set(draft):
                    raise ValueError("请保留草稿中的可编辑字段，不增删字段。")
                validate(value)
                _set_draft(prefix, value)
                st.rerun(scope="fragment")
            except (ValueError, OSError) as exc:
                st.error(str(exc))


def _range_images(store, library_id, item):
    if item.get("analysis_baseline"):
        root = store.directory(library_id)
        frames = [root / f["image"] for f in item["frames"] if (root / f["image"]).resolve().is_relative_to(root.resolve()) and (root / f["image"]).is_file()]
    else:
        frames = sorted((store.directory(library_id)/"frames"/item.get("observed_fingerprint", item["fingerprint"])/item["id"]).glob("*.jpg"))
    if frames:
        st.image([str(f) for f in frames], width=160,
                 caption=[f"帧 {i+1} · {frame['time']:.2f}s" for i, frame in enumerate(item.get("frames", []))][:len(frames)]
                 if len(frames) == len(item.get("frames", [])) else None)


@st.fragment
def _work_panel(store, library_id, tr):
    from app.services.library.knowledge import save_work, fetch_bangumi, validate_work
    prefix = "library_work_" + library_id
    with remembered_expander(tr("作品与资料"), key=prefix + "_open") as panel:
        if not panel.open:
            return
        busy = runner.library_busy(library_id)
        work = store.cache_get(library_id, "work") or {}
        fields = {"title": "", "adaptation": "动画", "season": "", "notes": "", "characters": [], "episode_map": {}}
        draft = st.session_state.setdefault(prefix + "_draft", {k: work.get(k, v) for k, v in fields.items()})
        epoch = str(st.session_state.get(prefix + "_epoch", 0))
        ranges = {r["id"]: r for r in store.records(library_id, include_stale=True)}
        with st.form(prefix + "_form_" + epoch):
            title = st.text_input(tr("作品名称"), value=draft["title"], disabled=busy)
            adaptation = st.text_input(tr("改编版本"), value=draft["adaptation"], disabled=busy,
                                       help=tr("例如电视动画第一版。小说、漫画和动画不要混用为画面事实。"))
            season = st.text_input(tr("季／篇章"), value=draft["season"], disabled=busy)
            notes = st.text_area(tr("人物、时期与剧情说明"), value=draft["notes"], height=160, disabled=busy)
            upload = st.file_uploader(tr("保存时用 MD／TXT 替换上方说明（可选）"), type=["md", "txt"], disabled=busy)
            st.caption(tr("用“## 第 4 集”区分单集资料；未分集的说明用于全部素材。外部资料不作为本地画面证据。"))
            characters = _rows_editor(draft["characters"], {
                "name": st.column_config.TextColumn(tr("人物名称"), required=True),
                "aliases": st.column_config.MultiselectColumn(tr("别名"), accept_new_options=True),
                "period": st.column_config.TextColumn(tr("人物时期")),
                "identity_basis": st.column_config.TextColumn(tr("身份说明")),
                "reference_ranges": st.column_config.MultiselectColumn(tr("参考镜头"), options=list(ranges),
                    format_func=lambda rid: _range_label(ranges[rid]) if rid in ranges else tr("引用已失效：") + rid,
                    accept_new_options=False),
            }, prefix + "_people_" + epoch, busy)
            st.caption(tr("可先填人物说明，拉片后再选择参考镜头；多值单元格可逐项添加别名。"))
            mapping = _rows_editor([{"local": int(k), "external": v} for k, v in draft["episode_map"].items()], {
                "local": st.column_config.NumberColumn(tr("本库集号"), min_value=1, step=1, required=True),
                "external": st.column_config.NumberColumn(tr("资料集号"), min_value=0.01, required=True),
            }, prefix + "_mapping_" + epoch, busy)
            if st.form_submit_button(tr("保存作品资料"), disabled=busy):
                try:
                    episode_map = {}
                    for row in mapping:
                        local = row["local"]
                        if local is None or not float(local).is_integer() or str(int(local)) in episode_map:
                            raise ValueError("本库集号需为不重复的正整数。")
                        episode_map[str(int(local))] = row["external"]
                    people = [{**p, "aliases": p.get("aliases") or [], "reference_ranges": p.get("reference_ranges") or [],
                               "period": p.get("period") or "", "identity_basis": p.get("identity_basis") or ""} for p in characters]
                    value = {"title": title, "adaptation": adaptation, "season": season, "characters": people,
                             "notes": upload.getvalue().decode("utf-8-sig") if upload else notes, "episode_map": episode_map}
                    save_work(store, library_id, value)
                    _set_draft(prefix, value)
                    st.session_state[prefix + "_notice"] = tr("作品资料已保存；继续拉片会按当前资料检查观察缓存。")
                    st.rerun(scope="fragment")
                except (ValueError, TypeError, OSError) as exc:
                    st.error(str(exc))
        if notice := st.session_state.pop(prefix + "_notice", None):
            st.success(notice)
        _advanced_json(prefix, draft, lambda value: validate_work(value, work, ranges), tr, busy)
        with remembered_expander(tr("参考镜头预览"), key=prefix + "_references") as preview:
            if preview.open and ranges:
                rid = st.selectbox(tr("选择本库镜头"), list(ranges), format_func=lambda rid: _range_label(ranges[rid]), key=prefix + "_reference")
                _range_images(store, library_id, ranges[rid])
            elif preview.open:
                st.info(tr("完成镜头观察后可在这里核对参考画面。"))
        with remembered_expander(tr("可选外部资料"), key=prefix + "_external") as external_panel:
            if external_panel.open:
                external = work.get("external", {})
                source = st.text_input(tr("Bangumi 作品链接或编号"), value=external.get("source", ""), key=prefix + "_source", disabled=busy)
                if st.button(tr("获取并保存公开资料"), key=prefix + "_fetch", disabled=busy or not source.strip()):
                    try:
                        with runner.library_operation(library_id), st.spinner(tr("获取公开作品资料")):
                            external = fetch_bangumi(source, config.proxy)
                            save_work(store, library_id, {"external": external})
                        st.success(tr("公开来源已保存；上方未提交的表单草稿保留。请核对动画版本和集号。"))
                    except (OSError, ValueError) as exc:
                        st.warning(tr("公开资料获取失败，本地资料保留：") + str(exc))
                if external:
                    st.caption(tr("资料来源：") + external.get("source", "") + " · " +
                               time.strftime("%Y-%m-%d %H:%M", time.localtime(external.get("fetched_at", 0))))
                    st.write(external.get("summary", ""))


def _start_build(store, library_id, settings, *, models=None, persist=False):
    from app.services.model_profiles import resolve_models
    settings = copy.deepcopy(settings)
    gateway = ModelGateway(copy.deepcopy(models if models is not None else resolve_models(settings.get("model_bindings"))))
    gateway.require("text")
    gateway.require("vision")
    def work(stop):
        if persist:
            saved = next(r["settings"] for r in store.list_libraries() if r["id"] == library_id)
            store.save_settings(library_id, {**saved, "build_settings": {**settings, "reanalyze": False}})
        build_library(store, library_id, gateway, settings, stop)
    _launch_library_job(store, library_id, "build", work)


def _range_detail(store, library_id, item, ranges, tr):
    from app.services.library.validation import correction_result
    busy = runner.library_busy(library_id)
    rid = item["id"]
    prefix = "library_range_" + library_id + "_" + rid
    if item.get("stale_reasons"):
        st.warning("；".join(item["stale_reasons"]) + tr("。请在拉片任务中选择本集继续，旧结果仅供查看。"))
    _range_images(store, library_id, item)
    st.write(item.get("summary", ""))
    st.caption(f"{tr('画面属性')}：{tr(_KINDS.get(item.get('content_kind'), '不确定'))} · {tr('时期')}：{item.get('period') or tr('未知')}")
    if item.get("uncertainties"):
        st.info("；".join(item["uncertainties"]))
    st.dataframe([{tr("依据"): tr(_EVIDENCE.get(o["kind"], o["kind"])), tr("观察"): o["text"]}
                  for o in item.get("observations", [])], width="stretch", hide_index=True)
    signatures = set(item.get("frame_signatures", []))
    similar = [r for r in ranges.values() if r["id"] != rid and signatures and
               len(signatures & set(r.get("frame_signatures", []))) >= min(2, len(signatures))]
    if similar:
        st.info(tr("相似采样画面（仅提示，须人工确认是否重复）：") + "；".join(_range_label(r) for r in similar[:10]))
    if st.button(tr("预览原片此处"), key=prefix + "_preview"):
        try:
            st.video(item["path"], start_time=int(item["start"]), end_time=max(int(item["start"])+1, int(item["end"])))
        except (OSError, ValueError) as exc:
            st.warning(tr("原片暂不可读取，可在已登记素材中检查路径：") + str(exc))
    if st.button(tr("补分析当前镜头"), key=prefix + "_enrich", disabled=busy or bool(item.get("stale_reasons"))):
        try:
            saved = next(r["settings"] for r in store.list_libraries() if r["id"] == library_id)
            _start_build(store, library_id, {**saved.get("build_settings", {}), "asset_ids": [item["asset_id"]],
                                           "range_ids": [rid], "reanalyze": True})
            st.rerun(scope="app")
        except (ValueError, OSError) as exc:
            st.error(str(exc))
    with remembered_expander(tr("更正当前观察"), key=prefix + "_edit") as editing:
        if not editing.open or item.get("stale_reasons"):
            return
        prefix += "_" + item["revision"]
        fields = ("summary", "period", "location", "content_kind", "duplicate_group", "tags", "uncertainties")
        initial = {k: item.get(k, [] if k in {"tags", "uncertainties"} else "") for k in fields}
        initial["visible_people"] = [{k: p.get(k, [] if k == "frame_ids" else "") for k in
                                     ("name", "period", "identity_basis", "frame_ids")} for p in item.get("visible_people", [])]
        initial["observations"] = [{k: o.get(k, [] if k.endswith("_ids") else "") for k in
                                   ("kind", "text", "frame_ids", "cue_ids")} for o in item.get("observations", [])]
        draft = st.session_state.setdefault(prefix + "_draft", initial)
        epoch = str(st.session_state.get(prefix + "_epoch", 0))
        frames, cues = {f["id"]: f for f in item["frames"]}, {c["id"]: c for c in item["cues"]}
        frame_col = st.column_config.MultiselectColumn(tr("可见帧"), options=list(frames), accept_new_options=False,
                       format_func=lambda fid: f"帧 {list(frames).index(fid)+1} · {frames[fid]['time']:.2f}s")
        cue_col = st.column_config.MultiselectColumn(tr("字幕引用"), options=list(cues), accept_new_options=False,
                     format_func=lambda cid: f"{cues[cid]['start']:.1f}s · {cues[cid]['text']}" + ("（跨界，仅作上下文）" if cues[cid].get("partial") else ""))
        with st.form(prefix + "_form_" + epoch):
            summary = st.text_area(tr("当前内容"), value=draft["summary"], disabled=busy)
            period = st.text_input(tr("时期"), value=draft["period"], disabled=busy)
            location = st.text_input(tr("地点"), value=draft["location"], disabled=busy)
            kind = st.selectbox(tr("画面属性"), list(_KINDS), index=list(_KINDS).index(draft["content_kind"]),
                                format_func=lambda k: tr(_KINDS[k]), disabled=busy)
            duplicate = st.text_input(tr("人工确认的重复画面组（可选）"), value=draft["duplicate_group"], disabled=busy)
            people = _rows_editor(draft["visible_people"], {
                "name": st.column_config.TextColumn(tr("出镜人物"), required=True), "period": tr("人物时期"),
                "identity_basis": st.column_config.TextColumn(tr("身份依据"), required=True), "frame_ids": frame_col,
            }, prefix + "_people_" + epoch, busy)
            observations = _rows_editor(draft["observations"], {
                "kind": st.column_config.SelectboxColumn(tr("依据种类"), options=list(_EVIDENCE), required=True,
                                                         format_func=lambda k: tr(_EVIDENCE[k])),
                "text": st.column_config.TextColumn(tr("观察说明"), required=True), "frame_ids": frame_col, "cue_ids": cue_col,
            }, prefix + "_observations_" + epoch, busy)
            st.caption(tr("可见画面必须引用当前帧，字幕说明必须引用完整台词；跨界台词仅可作为上下文。抽帧证据不代表整段画面。"))
            tags = st.text_area(tr("表现方向（每行一项）"), value="\n".join(draft["tags"]), disabled=busy)
            uncertainties = st.text_area(tr("尚未确认（每行一项）"), value="\n".join(draft["uncertainties"]), disabled=busy)
            reason = st.text_input(tr("更正依据"), disabled=busy)
            if st.form_submit_button(tr("保存本镜头更正"), disabled=busy):
                try:
                    people = [{**p, "period": p.get("period") or "", "frame_ids": p.get("frame_ids") or []} for p in people]
                    observations = [{**o, "frame_ids": o.get("frame_ids") or [], "cue_ids": o.get("cue_ids") or []} for o in observations]
                    store.correct_range(library_id, rid, {"summary": summary, "period": period, "location": location,
                        "content_kind": kind, "duplicate_group": duplicate, "visible_people": people, "observations": observations,
                        "tags": [v.strip() for v in tags.splitlines() if v.strip()],
                        "uncertainties": [v.strip() for v in uncertainties.splitlines() if v.strip()]},
                        reason, expected_revision=item["revision"])
                    st.session_state["library_notice_" + library_id] = tr("更正已保存；请继续本集内容分段以更新关联目录。其他镜头观察保留。")
                    st.rerun(scope="app")
                except (ValueError, OSError) as exc:
                    st.error(str(exc))
        _advanced_json(prefix, draft, lambda value: correction_result(item, value), tr, busy)


@st.fragment
def _content_panel(store, library_id, tr):
    prefix = "library_content_" + library_id
    with remembered_expander(tr("内容与镜头浏览"), key=prefix + "_open") as panel:
        if not panel.open:
            return
        ranges = {r["id"]: r for r in store.records(library_id, include_stale=True)}
        events = store.records(library_id, "contents", include_stale=True)
        if not ranges and not events:
            st.info(tr("目前只有素材登记，请打开“拉片任务”建立观察和内容目录。"))
            return
        view = st.radio(tr("查看方式"), ["ranges", "contents"], format_func=lambda v: tr("镜头观察" if v == "ranges" else "内容段"),
                        horizontal=True, key=prefix + "_view")
        if not events:
            st.info(tr("观察已保存，内容目录尚未完成；可先查看或更正观察，再在拉片任务中继续。"))
        episodes = sorted({r["episode"] for r in [*ranges.values(), *events]})
        episode = st.selectbox(tr("按集筛选"), [None, *episodes], format_func=lambda v: tr("全部") if v is None else f"第{v}集", key=prefix + "_episode")
        status = st.selectbox(tr("按状态筛选"), ["全部", "待更新", "待确认", *_KINDS.values()], key=prefix + "_status")
        query = st.text_input(tr("搜索事件、人物、时期或地点"), key=prefix + "_search").strip().casefold()
        visible = []
        for row in list(ranges.values()) if view == "ranges" else events:
            related = [row] if view == "ranges" else [ranges[rid] for rid in row["range_ids"] if rid in ranges]
            if episode is not None and row["episode"] != episode:
                continue
            if status == "待更新" and not row.get("stale_reasons"):
                continue
            if status == "待确认" and not any(r.get("analysis_state") != "ready" and not r.get("stale_reasons") for r in related):
                continue
            if status in _KINDS.values() and not any(_KINDS.get(r.get("content_kind")) == status for r in related):
                continue
            if query and query not in json.dumps([row, related], ensure_ascii=False).casefold():
                continue
            visible.append(row)
        st.caption(f"{len(events)} 个已保存内容段 · {len(ranges)} 个已保存观察 · {len(visible)} 个筛选结果")
        if not visible:
            st.info(tr("当前筛选下没有记录。"))
            return
        pages = max(1, (len(visible)+9)//10)
        page_key = prefix + "_page"
        st.session_state[page_key] = min(max(1, int(st.session_state.get(page_key, 1))), pages)
        page = st.number_input(tr("列表页码"), min_value=1, max_value=pages, step=1, key=page_key)
        rows = {r["id"]: r for r in visible[(page-1)*10:page*10]}
        if st.session_state.get(prefix + "_selected") not in rows:
            st.session_state.pop(prefix + "_selected", None)
        selected = st.selectbox(tr("选择查看"), list(rows), key=prefix + "_selected",
                               format_func=lambda rid: _range_label(rows[rid]) if view == "ranges" else
                               f"第{rows[rid]['episode']}集 · {rows[rid]['start']:.1f}–{rows[rid]['end']:.1f}s · {rows[rid].get('title', '')}")
        with remembered_expander(tr("当前条目详情"), key=prefix + "_detail", default=True) as detail:
            if not detail.open:
                return
            row = rows[selected]
            if view == "contents":
                st.write(row["summary"])
                if row.get("stale_reasons"):
                    st.warning("；".join(row["stale_reasons"]))
                    st.caption(tr("缺失或已更新的关联范围：") + "、".join(row.get("missing_range_ids", [])))
                available = [rid for rid in row["range_ids"] if rid in ranges]
                if not available:
                    st.info(tr("当前内容段没有可查看的关联观察，请继续本集拉片。"))
                    return
                if st.session_state.get(prefix + "_range") not in available:
                    st.session_state.pop(prefix + "_range", None)
                rid = st.selectbox(tr("关联镜头"), available, format_func=lambda rid: _range_label(ranges[rid]), key=prefix + "_range")
                row = ranges[rid]
            _range_detail(store, library_id, row, ranges, tr)


@st.fragment
def _import_panel(store, library_id, tr):
    saved = next(row["settings"] for row in store.list_libraries() if row["id"] == library_id)
    busy = runner.library_busy(library_id)
    with remembered_expander(tr("导入本地目录"), key="library_import_open_" + library_id, default=not store.assets(library_id)) as panel:
        if not panel.open:
            return
        st.caption(tr("请输入运行 NarraLens 的电脑上的目录，大视频无需浏览器上传。"))
        video_dir = folder_input(tr("视频目录"), value=saved.get("video_dir", str(Path(config.root_dir) / "resource" / "videos")), key="lib_video_dir_" + library_id, tr=tr, disabled=busy)
        subtitle_dir = folder_input(tr("字幕目录（可选）"), value=saved.get("subtitle_dir", str(Path(config.root_dir) / "resource" / "srt")), key="lib_subtitle_dir_" + library_id, tr=tr, disabled=busy)
        recursive = st.checkbox(tr("扫描子目录"), value=saved.get("recursive", True), key="lib_recursive_" + library_id, disabled=busy)
        draft_key = "library_scan_" + library_id
        if st.button(tr("扫描并匹配"), disabled=busy, key="lib_scan_" + library_id):
            try:
                videos = scan_directory(video_dir, VIDEO_EXTENSIONS, recursive)
                subtitles = scan_directory(subtitle_dir, SUBTITLE_EXTENSIONS, recursive) if subtitle_dir.strip() else []
                if not videos:
                    raise ValueError(tr("目录中没有支持的视频文件。"))
                st.session_state[draft_key] = {"mappings": suggest_mappings(videos, subtitles), "subtitles": subtitles}
                store.save_settings(library_id, {**saved, "video_dir": video_dir, "subtitle_dir": subtitle_dir, "recursive": recursive})
            except (OSError, ValueError) as exc:
                st.error(str(exc))
        draft = st.session_state.get(draft_key)
        if draft and not busy:
            st.caption(tr("多字幕选择说明"))
            no_subtitle = tr("不导入字幕")
            choose_subtitle = tr("请选择字幕")
            subtitle_options = [no_subtitle, choose_subtitle] + draft["subtitles"]
            rows = [{"import": True, "episode": row["episode"], "video": row["video"],
                     "subtitle": row["subtitle"] or (choose_subtitle if row["ambiguous"] else no_subtitle)} for row in draft["mappings"]]
            edited = st.data_editor(pd.DataFrame(rows), hide_index=True, width="stretch",
                disabled=True if busy else ["video"], key="library_mapping_" + library_id + "_" + str(hash(str(rows))),
                column_config={"import": st.column_config.CheckboxColumn(tr("导入")),
                               "episode": st.column_config.NumberColumn(tr("集数"), min_value=1, step=1, required=True),
                               "video": st.column_config.TextColumn(tr("视频路径"), width="large"),
                               "subtitle": st.column_config.SelectboxColumn(tr("对应字幕"), options=subtitle_options, width="large", required=True)})
            subtitle_settings = None
            try:
                subtitle_settings = _subtitle_settings(draft["subtitles"], saved.get("subtitle_settings", {}), tr, library_id)
            except (OSError, ValueError) as exc:
                st.error(str(exc))
            if st.button(tr("登记所选素材"), disabled=busy or subtitle_settings is None, key="library_import_" + library_id):
                try:
                    mappings = []
                    for row in edited.to_dict("records"):
                        if not row["import"]:
                            continue
                        if row["subtitle"] == choose_subtitle:
                            raise ValueError(tr("请确认有歧义的视频与字幕对应关系。"))
                        subtitle_path = "" if row["subtitle"] == no_subtitle else row["subtitle"]
                        if subtitle_path and subtitle_path not in draft["subtitles"]:
                            raise ValueError(tr("请选择扫描到的字幕文件。"))
                        mappings.append(ImportMapping(row["video"], subtitle_path, int(row["episode"])))
                    if not mappings:
                        raise ValueError(tr("请至少选择一个视频。"))
                    mappings, subtitle_settings = tuple(mappings), copy.deepcopy(subtitle_settings)
                    def import_snapshot(stop):
                        current = next(row["settings"] for row in store.list_libraries() if row["id"] == library_id)
                        store.save_settings(library_id, {**current, "video_dir": video_dir, "subtitle_dir": subtitle_dir,
                                                        "recursive": recursive, "subtitle_settings": subtitle_settings})
                        import_materials(store, library_id, mappings, subtitle_settings, stop)
                    _launch_library_job(store, library_id, "import", import_snapshot)
                    st.rerun()
                except (OSError, ValueError, TypeError) as exc:
                    st.error(str(exc))

def _asset_actions(store, library_id, assets, selected, tr, busy):
    if selected:
        from app.services.narration_task import list_tasks
        related = [task for task in list_tasks() if library_id in library_ids(task)]
        active = runner.running("library:" + library_id) or any(runner.running("narration:" + task["id"]) for task in related)
        stale = store.deletion_info(library_id)["running_records"] or any(task.get("state") == "running" for task in related)
        st.caption(tr("移除登记素材说明"))
        st.write(tr("已勾选：") + "、".join(Path(item["path"]).name for item in assets if item["id"] in selected))
        confirmed = st.checkbox(tr("确认移除所选登记和派生数据"), key="remove_assets_ack_" + library_id)
        interrupted_confirmed = st.checkbox(tr("确认旧程序及其他窗口没有在运行关联任务"), key="remove_assets_interrupted_" + library_id) if stale and not active else False
        if active:
            st.info(tr("请先停止视频素材库或关联的生成任务，再删除。"))
        if st.button(tr("移除勾选素材"), disabled=active or not confirmed or bool(stale and not interrupted_confirmed), key="remove_assets_button_" + library_id):
            try:
                remaining = store.remove_assets(library_id, selected, confirmed, interrupted_confirmed)
                st.session_state["library_delete_notice"] = tr("所选素材已移除，原媒体与成片保留。") + ("\n" + "\n".join(remaining) if remaining else "")
                st.rerun()
            except (OSError, ValueError) as exc:
                st.error(str(exc))
    with remembered_expander(tr("重新绑定移动后的原视频"), key="library_rebind_open_" + library_id) as panel:
        if not panel.open:
            return
        asset_id = st.selectbox(tr("选择素材"), [item["id"] for item in assets], format_func=lambda value: Path(next(item["path"] for item in assets if item["id"] == value)).name, key="rebind_asset_" + library_id)
        new_path = st.text_input(tr("新视频完整路径"), key="rebind_path_" + library_id)
        st.caption(tr("重绑定会核对文件指纹、大小和时长，确认是同一个原视频后保留分析结果。"))
        if st.button(tr("确认重绑定"), disabled=busy, key="rebind_button_" + library_id):
            try:
                store.rebind(library_id, asset_id, inspect_video(new_path.strip().strip('"')))
                st.session_state["library_notice_" + library_id] = tr("路径已更新，素材ID和分析结果保留。请重新检查文件可用性。")
                st.rerun(scope="app")
            except (OSError, ValueError) as exc:
                st.error(str(exc))

def _delete_panel(store, library_id, tr):
    library = next(row for row in store.list_libraries() if row["id"] == library_id)
    with remembered_expander(tr("删除视频素材库"), key="library_delete_open_" + library_id) as panel:
        if not panel.open:
            return
        from app.services.narration_task import list_tasks
        related_tasks = [task for task in list_tasks() if library_id in library_ids(task)]
        active = runner.running("library:" + library_id) or any(runner.running("narration:" + task["id"]) for task in related_tasks)
        info = store.deletion_info(library_id)
        st.write(tr("将删除视频素材库：") + library["name"])
        st.caption(tr("删除范围说明"))
        st.caption(tr("删除数据统计").format(assets=info["assets"], scenes=info["scenes"], cache=info["analysis_cache"]))
        st.code(info["directory"], language=None)
        stale = info["running_records"] or any(task.get("state") == "running" for task in related_tasks)
        interrupted_confirmed = False
        if active:
            st.info(tr("请先停止视频素材库或关联的生成任务，再删除。"))
        elif stale:
            interrupted_confirmed = st.checkbox(tr("确认旧程序及其他窗口没有在运行关联任务"), key="library_delete_interrupted_" + library_id)
        confirmation = st.text_input(tr("输入视频素材库名称确认删除"), key="library_delete_name_" + library_id)
        if st.button(tr("永久删除此视频素材库"), disabled=active or confirmation.strip() != library["name"] or bool(stale and not interrupted_confirmed), key="library_delete_confirm_" + library_id):
            try:
                cleanup_message = store.delete_library(library_id, confirmation, interrupted_confirmed=interrupted_confirmed)
                if st.session_state.get("active_library_id") == library_id:
                    st.session_state.pop("active_library_id", None)
                st.session_state.pop("library_scan_" + library_id, None)
                st.session_state["library_delete_notice"] = cleanup_message or tr("视频素材库已删除，原媒体和已生成成片保留。")
                st.rerun()
            except (OSError, ValueError) as exc:
                st.error(str(exc))


def _check_files(store, library_id, assets, stop):
    for index, asset in enumerate(assets):
        runner.check_cancel(stop)
        store.job(library_id, "files", progress=100*index/max(1, len(assets)),
                  message=f"检查原视频可用性 {index+1}/{len(assets)} · 第{asset['episode']}集（每项最多8秒）")
        result = probe_file(asset["path"])
        store.cache_put(library_id, "file-check:" + asset["id"], {
            **result, "identity": [asset["path"], asset["fingerprint"], asset["size"], asset["mtime_ns"]],
            "checked_at": time.time(),
        })
    store.job(library_id, "files", state="complete", progress=100, message="文件检查已完成；结果见已登记素材，未修改原文件。")


@st.fragment
def _registered_panel(store, library_id, tr):
    with remembered_expander(tr("已登记素材"), key="library_assets_" + library_id, default=True) as panel:
        if not panel.open:
            return
        assets = store.assets(library_id)
        if not assets:
            st.info(tr("尚未登记素材，请打开下方导入本地目录。"))
            return
        busy = runner.library_busy(library_id)
        status = store.content_status(library_id)
        with store.database(library_id) as db:
            checks = {row[0][11:]: json.loads(row[1]) for row in db.execute(
                "SELECT key,value FROM analysis_cache WHERE key LIKE 'file-check:%'")}
        rows = []
        for asset in assets:
            check = checks.get(asset["id"], {})
            if check.get("identity") != [asset["path"], asset["fingerprint"], asset["size"], asset["mtime_ns"]]:
                check = {}
            state = status[(library_id, asset["id"])]
            rows.append({"selected": False, "id": asset["id"], "episode": asset["episode"], "video": Path(asset["path"]).name,
                "duration": round(asset["duration"], 1), "subtitle": Path(asset["subtitle_path"]).name if asset.get("subtitle_path") else "",
                "status": tr(state["label"]), "observations": state["observations"], "contents": state["contents"], "usable": state["usable"],
                "available": tr(_FILE_STATES.get(check.get("state", "unchecked"), "结果未确认")),
                "checked": time.strftime("%m-%d %H:%M:%S", time.localtime(check["checked_at"])) if check.get("checked_at") else "",
                "reason": check.get("reason", "")})
        table = st.data_editor(pd.DataFrame(rows), hide_index=True, width="stretch",
            disabled=True if busy else [key for key in rows[0] if key != "selected"],
            key="registered_assets_" + library_id + "_" + stable_hash([(a["id"], a["active_subtitle"]) for a in assets])[:12],
            column_config={"id": None, "selected": st.column_config.CheckboxColumn(tr("选择"), required=True), "episode": tr("集数"),
                "video": tr("视频"), "duration": tr("时长（秒）"), "subtitle": tr("字幕"), "status": tr("分析状态"),
                "observations": tr("当前观察"), "contents": tr("有效内容段"), "usable": tr("可用剧情范围"),
                "available": tr("文件状态"), "checked": tr("检查时间"), "reason": tr("文件检查说明")})
        selected = [row["id"] for row in table.to_dict("records") if row["selected"]]
        st.caption(tr("页面刷新不会访问原视频。文件状态来自上次显式检查，和分析状态分别显示；更换路径后需重新检查。"))
        if st.button(tr("检查所选原视频" if selected else "检查全部原视频"), disabled=busy, key="check_files_" + library_id):
            try:
                snapshot = copy.deepcopy([a for a in assets if not selected or a["id"] in selected])
                _launch_library_job(store, library_id, "files", lambda stop: _check_files(store, library_id, snapshot, stop))
                st.rerun(scope="app")
            except (OSError, ValueError) as exc:
                st.error(str(exc))
        _asset_actions(store, library_id, assets, selected, tr, busy)


@st.fragment
def _agent_panel(store, library_id, targets, settings, tr):
    from app.services.library import agent_tasks
    prefix = "library_agent_" + library_id
    with remembered_expander(tr("外部 Agent 任务"), key=prefix + "_open") as panel:
        if not panel.open:
            return
        busy = runner.library_busy(library_id)
        st.caption(tr("先创建本地任务，再将指令交给支持看图和本地命令的Agent。准备不调用模型，原视频保留原路径。"))
        if st.button(tr("创建 Agent 拉片任务"), disabled=busy or not targets, key=prefix + "_create"):
            try:
                selected = [a["id"] for a in targets]
                snapshot = copy.deepcopy(settings)
                _launch_library_job(store, library_id, "agent", lambda stop: agent_tasks.create_task(store, library_id, selected, snapshot, stop))
                st.rerun(scope="app")
            except (ValueError, OSError) as exc:
                st.error(str(exc))
        tasks = agent_tasks.list_tasks(library_id)
        if not tasks:
            st.info(tr("尚无外部Agent任务。"))
            return
        identifiers = [r["id"] for r in tasks]
        if st.session_state.get(prefix + "_task") not in identifiers:
            st.session_state.pop(prefix + "_task", None)
        identifier = st.selectbox(tr("选择 Agent 任务"), identifiers, key=prefix + "_task",
            format_func=lambda tid: next(f"{time.strftime('%m-%d %H:%M', time.localtime(r['created']))} · {tid[:8]} · {len(r['assets'])}个素材" for r in tasks if r["id"] == tid))
        if st.button(tr("刷新任务进度"), key=prefix + "_refresh"):
            st.rerun(scope="fragment")
        try:
            status = agent_tasks.describe(identifier)
            root = agent_tasks.task_directory(identifier)
        except (ValueError, OSError) as exc:
            st.error(str(exc))
            return
        state_labels = {"preparing": "准备尚未完成", "partial": "已准备部分素材", "ready": "本地任务已准备"}
        st.write(tr(state_labels.get(status["state"], status["state"])))
        st.caption(f"已收到观察草稿 {sum(r['kind']=='observation' for r in status['saved'])} 项 · 内容分组 {sum(r['kind']=='contents' for r in status['saved'])} 份")
        st.caption(tr("尚无观察草稿或可复用结果的范围：") + str(status["pending_observations"]))
        st.caption(tr("最近收到结果：") + (time.strftime("%m-%d %H:%M:%S", time.localtime(status["last_received"])) if status["last_received"] else tr("尚未收到")))
        st.caption(tr("这里显示已落盘成果；外部Agent是否正在运行，请在其应用中查看。"))
        st.caption(tr("任务指令（点击代码框右上角复制）"))
        st.code(f"请阅读 {status['instructions']}，按其中工具和要求完成拉片；先查看已有进度，逐项提交结果，最后告知我回到NarraLens检查并导入。", language=None)
        if st.button(tr("打开任务文件夹"), key=prefix + "_folder"):
            try:
                import os
                if os.name == "nt":
                    os.startfile(str(root))
                else:
                    st.info(str(root))
            except OSError as exc:
                st.warning(str(exc))
        if status["state"] in {"preparing", "partial"} and st.button(tr("继续本地准备"), disabled=busy, key=prefix + "_prepare"):
            try:
                agent_tasks.control_task(identifier, False)
                _launch_library_job(store, library_id, "agent", lambda stop: agent_tasks.prepare_task(store, identifier, stop))
                st.rerun(scope="app")
            except (ValueError, OSError) as exc:
                st.error(str(exc))
        left, right = st.columns(2)
        check = left.button(tr("检查结果"), disabled=busy or status["state"] == "preparing", key=prefix + "_check")
        commit = right.button(tr("导入合格结果"), disabled=busy or not status["saved"] or status["state"] == "preparing", key=prefix + "_import")
        if check or commit:
            try:
                def process(stop):
                    result = agent_tasks.check_task(store, identifier, stop, commit=commit)
                    store.job(library_id, "agent", state="partial" if result["pending"] else "complete", progress=100,
                        message=f"Agent结果检查完成：可导入{result['ready']}、已存在{result['existing']}、待处理{result['pending']}；本次写入{result['imported']}项。")
                _launch_library_job(store, library_id, "agent", process)
                st.rerun(scope="app")
            except (ValueError, OSError) as exc:
                st.error(str(exc))
        report = read_json(root / "reports/last-check.json", default={})
        if report:
            st.caption(tr("上次检查：") + time.strftime("%m-%d %H:%M:%S", time.localtime(report["checked_at"])))
            labels = {"ready_to_import": "合格／可导入", "already_imported": "已存在", "pending": "待补", "conflict": "输入冲突"}
            st.dataframe([{tr("素材"): r.get("asset_id", ""), tr("镜头"): r.get("range_id", r.get("key", "")),
                           tr("检查结果"): tr(labels.get(r["state"], r["state"])), tr("说明"): r["reason"]} for r in report["rows"][:200]],
                         hide_index=True, width="stretch")
            st.download_button(tr("下载完整检查／待补报告"), json.dumps(report, ensure_ascii=False, indent=2),
                               file_name="agent-analysis-report.json", mime="application/json", key=prefix + "_download_report")
        if status["issues"]:
            with remembered_expander(tr("任务待处理问题"), key=prefix + "_issues") as issues:
                if issues.open:
                    st.write(status["issues"])
        if st.button(tr("允许继续取证" if status["stop_requested"] else "请求停止后续取证"), key=prefix + "_stop"):
            agent_tasks.control_task(identifier, not status["stop_requested"])
            st.rerun(scope="fragment")
        st.caption(tr("停止标记只约束后续本地工具调用；已发出的模型请求请在Agent应用中停止。"))
        with remembered_expander(tr("移除 Agent 任务目录"), key=prefix + "_delete_open") as deletion:
            if deletion.open:
                st.code(str(root), language=None)
                confirmed = st.checkbox(tr("确认移除此任务的输入副本、提案、未导入结果和取证图；库内已导入证据与原媒体保留"), key=prefix + "_delete_ack_" + identifier)
                if st.button(tr("移除此任务目录"), disabled=busy or not confirmed, key=prefix + "_delete"):
                    try:
                        agent_tasks.delete_task(store, identifier, identifier)
                        st.rerun(scope="app")
                    except (ValueError, OSError) as exc:
                        st.error(str(exc))


@st.fragment
def _build_panel(store, library_id, tr):
    prefix = "library_build_" + library_id
    with remembered_expander(tr("拉片任务"), key=prefix + "_open", default=True) as panel:
        if not panel.open:
            return
        assets = store.assets(library_id)
        if not assets:
            st.info(tr("请先登记视频。"))
            _agent_panel(store, library_id, [], {}, tr)
            return
        busy = runner.library_busy(library_id)
        saved = next(row["settings"] for row in store.list_libraries() if row["id"] == library_id)
        draft = st.session_state.setdefault(prefix + "_draft", copy.deepcopy(saved.get("build_settings", {})))
        analysis_mode = st.radio(tr("分析方式"), ["auto", "agent"],
            index=1 if draft.get("analysis_mode") == "agent" else 0,
            format_func=lambda value: tr("自动拉片" if value == "auto" else "外部 Agent"),
            horizontal=True, key=prefix + "_analysis_mode", disabled=busy)
        model_prefix = "video_library_models_" + library_id
        if analysis_mode == "auto":
            seed_model_choices(model_prefix, bindings=draft.get("model_bindings"))
            with remembered_expander(tr("模型选择"), key=prefix + "_models") as models:
                if models.open:
                    if busy:
                        st.info(tr("运行中使用启动时的模型快照，任务结束后可修改选择。"))
                    else:
                        render_model_choices(tr, model_prefix, roles=("text", "vision"))
            remember_widget_values(model_prefix)
            st.caption(tr("图片和字幕联合拉片。普通继续只补缺项；内容目录失败时复用已保存观察。"))
        window = st.number_input(tr("长镜头分析范围目标时长（秒）"), min_value=10, max_value=120,
            value=int(draft.get("window_seconds", 15)), step=5, key=prefix + "_window", disabled=busy,
            help=tr("用于细分长镜头，可能就近采用字幕边界，不保证等长。不是抽帧间隔；每个范围按时长与字幕密度采样1–6帧。"))
        if analysis_mode == "auto":
            left, right = st.columns(2)
            with left:
                batch_size = st.number_input(tr("每次请求最多包含的镜头范围数"), min_value=1, max_value=128,
                    value=int(draft.get("batch_size", 50)), step=1, key=prefix + "_batch", disabled=busy,
                    help=tr("每批来自同一视频。范围数不是内容事件数；超出模型预算时自动缩批。"))
            with right:
                concurrency = st.number_input(tr("API 请求并发上限"), min_value=1,
                    value=int(draft.get("concurrency", runner.DEFAULT_API_CONCURRENCY)), step=1, key=prefix + "_concurrency", disabled=busy,
                    help=tr("可输入任意正整数。实际在途请求另行显示，媒体槽位和接口限流会影响实际并发。"))
        else:
            batch_size = int(draft.get("batch_size", 50))
            concurrency = int(draft.get("concurrency", runner.DEFAULT_API_CONCURRENCY))
        mode = st.radio(tr("本次分析范围"), ["全部素材", "指定素材"],
                        index=1 if draft.get("scope_mode", "指定素材" if draft.get("asset_ids") else "全部素材") == "指定素材" else 0,
                        horizontal=True, key=prefix + "_scope", disabled=busy)
        labels = {a["id"]: f"第{a['episode']}集 · {Path(a['path']).name}" for a in assets}
        selected = []
        if mode == "指定素材":
            selected = st.multiselect(tr("选择本次素材"), list(labels), format_func=labels.get,
                default=[aid for aid in draft.get("asset_ids", []) if aid in labels], key=prefix + "_assets", disabled=busy)
        targets = assets if mode == "全部素材" else [a for a in assets if a["id"] in selected]
        st.caption(f"本次处理 {len(targets)} 个素材 · 集号：" + "、".join(str(a["episode"]) for a in targets))
        reanalyze = False
        if analysis_mode == "auto":
            reanalyze = st.checkbox(tr("重新分析所选范围"), value=False, key=prefix + "_force", disabled=busy,
                                   help=tr("默认关闭。开启会重做本次范围内已成功的观察；只补目录无需开启。"))
            if reanalyze:
                st.warning(f"将重新请求上述 {len(targets)} 个素材的镜头观察。")
        settings = {"visual": True, "window_seconds": window, "batch_size": batch_size, "concurrency": concurrency,
                    "model_bindings": selected_bindings(model_prefix) if analysis_mode == "auto" else draft.get("model_bindings"),
                    "asset_ids": selected, "scope_mode": mode, "analysis_mode": analysis_mode}
        st.session_state[prefix + "_draft"] = copy.deepcopy(settings)
        if analysis_mode == "agent":
            _agent_panel(store, library_id, targets, settings, tr)
            return
        if st.button(tr("建立／继续视频素材库"), key=prefix + "_start", disabled=busy or not targets):
            try:
                check_model_choices(model_prefix, ["text", "text_backup", "vision", "vision_backup"])
                _start_build(store, library_id, {**settings, "asset_ids": [a["id"] for a in targets], "reanalyze": reanalyze},
                             models=selected_models(model_prefix), persist=True)
                st.rerun(scope="app")
            except (ValueError, OSError) as exc:
                st.error(str(exc))
        st.caption(tr("运行中锁定当前库的修改操作，停止后保留已完成观察、目录和回包。"))


@st.fragment
def _maintenance_panel(store, library_id, tr):
    with remembered_expander(tr("维护与高级选项"), key="library_maintenance_" + library_id) as panel:
        if not panel.open:
            return
        busy = runner.library_busy(library_id)
        st.subheader(tr("字幕资料校验与恢复"))
        st.caption(tr("本地校验 ASS 中文正文与来源身份；恢复后只更新当前字幕登记，受影响观察和内容段需要重新拉片。原字幕与旧结果保留，不调用模型。"))
        report_path = store.directory(library_id) / "subtitle-audit.json"
        identity = stable_hash([(a["id"], a["active_subtitle"], a["fingerprint"]) for a in store.assets(library_id)])
        if st.button(tr("校验当前字幕"), disabled=busy, key="subtitle_audit_" + library_id):
            def audit(stop):
                rows = audit_subtitles(store, library_id, stop)
                write_json(report_path, {"identity": identity, "rows": rows})
                store.job(library_id, "subtitle", state="complete", progress=100, message="字幕校验已完成，见维护与高级选项。")
            try:
                _launch_library_job(store, library_id, "subtitle", audit)
                st.rerun(scope="app")
            except (OSError, ValueError) as exc:
                st.error(str(exc))
        report = read_json(report_path, default={})
        if report.get("identity") == identity:
            rows = report["rows"]
            labels = {"ok": "正常", "recovery_needed": "正文待恢复", "lyrics_only": "只有歌词／注释", "error": "原件或路径异常"}
            st.dataframe([{tr("集数"): r["episode"], tr("视频"): r["video"], tr("校验"): tr(labels.get(r["state"], r["state"])),
                           tr("缺失正文"): r.get("missing_body_events", 0), tr("说明"): r.get("error", "")} for r in rows],
                         hide_index=True, width="stretch")
            targets = [r for r in rows if r["state"] == "recovery_needed"]
            if targets:
                st.warning(tr("需恢复素材：") + "、".join(f"第{r['episode']}集" for r in targets))
            if st.button(tr("恢复当前素材字幕"), key="restore_subtitle_" + library_id,
                         disabled=busy or not targets or any(r["state"] == "error" for r in rows)):
                try:
                    _launch_library_job(store, library_id, "subtitle", lambda stop: restore_subtitle_text(store, library_id, stop))
                    st.rerun(scope="app")
                except (ValueError, OSError) as exc:
                    st.error(str(exc))
        st.caption(tr("恢复后：打开“拉片任务”→指定受影响集号→建立／继续，普通继续即可；其余素材的结果保持。"))
        with remembered_expander(tr("最近拉片诊断"), key="library_quality_" + library_id) as quality_panel:
            if quality_panel.open:
                quality = read_json(store.directory(library_id)/"analysis-quality.json", default={})
                for issue in quality.get("issues", [])[:100]:
                    st.write(issue)
                usage = quality.get("token_usage", {}).get("current", {})
                st.caption(f"最近拉片已报告 Token：{sum(r.get('total_tokens', 0) for r in usage.values()):,} · "
                           f"未报告用量请求：{sum(r.get('unknown', 0) for r in usage.values())}")
        _delete_panel(store, library_id, tr)


def render_library_panel(tr):
    st.header(tr("视频素材库"))
    st.caption(tr("视频保留原路径，字幕、观察和内容目录保存在本项目，供不同文案复用。"))
    store = LibraryStore()
    if notice := st.session_state.pop("library_delete_notice", None):
        st.info(notice)
    with remembered_expander(tr("新建视频素材库"), key="library_create_open", default=not store.list_libraries()) as panel:
        if panel.open:
            with st.form("new_material_library"):
                name = st.text_input(tr("视频素材库名称"), placeholder=tr("例如：无职转生第一季"))
                if st.form_submit_button(tr("创建视频素材库")):
                    try:
                        st.session_state["active_library_id"] = store.create(name)
                        st.rerun()
                    except (ValueError, OSError) as exc:
                        st.error(str(exc))
    library_id = choose_library(store, tr)
    if not library_id:
        return
    library = next(item for item in store.list_libraries() if item["id"] == library_id)
    busy = runner.library_busy(library_id)
    if store.schema(library_id) != 2:
        st.warning(tr("此库为旧窗口结构，需复制登记和字幕到2.8拉片库后重新分析。"))
        if st.button(tr("创建 2.8 拉片副本"), disabled=busy, key="clone_content_" + library_id):
            try:
                st.session_state["active_library_id"] = store.clone_for_analysis(library_id, library["name"] + " · 2.8")
                st.rerun()
            except (ValueError, OSError) as exc:
                st.error(str(exc))
        _delete_panel(store, library_id, tr)
        return
    if notice := st.session_state.pop("library_notice_" + library_id, None):
        st.success(notice)
    counts = store.content_status(library_id)
    columns = st.columns(4)
    for column, label, number in zip(columns, ["已登记素材", "当前镜头观察", "有效内容段", "待处理素材"],
                                    [len(counts), sum(s["observations"] for s in counts.values()),
                                     sum(s["contents"] for s in counts.values()), sum(s["incomplete"] for s in counts.values())]):
        column.metric(tr(label), number)
    if busy:
        st.info(tr("当前库或关联生成任务正在运行，可查看已保存资料；修改操作在任务结束后恢复。"))
    _work_panel(store, library_id, tr)
    _registered_panel(store, library_id, tr)
    _import_panel(store, library_id, tr)
    _build_panel(store, library_id, tr)
    _job_status(library_id, tr)
    _content_panel(store, library_id, tr)
    _maintenance_panel(store, library_id, tr)
