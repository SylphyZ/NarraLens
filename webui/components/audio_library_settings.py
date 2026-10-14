"""Audio library management and one BGM selector shared by both video workflows."""

from pathlib import Path

import pandas as pd
import streamlit as st

from app.config import config
from app.services.audio_library import AudioLibraryStore
from app.services.library import runner
from app.services.library.gateway import ModelGateway
from app.services.library.store import read_json
from app.services.narration_bgm import eligible_tracks
from webui.utils.file_utils import folder_input, remembered_widget_value, remembered_expander, remember_widget_values


def _preview(path, key, tr):
    if path and st.button(tr("试听"), key=key) and Path(path).is_file():
        formats = {".mp3": "audio/mpeg", ".m4a": "audio/mp4", ".aac": "audio/aac", ".flac": "audio/flac", ".ogg": "audio/ogg", ".opus": "audio/ogg", ".wma": "audio/x-ms-wma", ".aiff": "audio/aiff", ".aif": "audio/aiff"}
        st.audio(path, format=formats.get(Path(path).suffix.lower(), "audio/wav"))


def _choose_audio_libraries(store, tr, *, key):
    libraries = store.list_libraries()
    if not libraries:
        st.info(tr("请先在音频素材库页面登记音乐，也可选择手动输入路径。"))
        return []
    names = {item["id"]: item["name"] for item in libraries}
    library_key = key + "_libraries"
    if library_key in st.session_state:
        st.session_state[library_key] = [identifier for identifier in st.session_state[library_key] if identifier in names]
    default = [identifier for identifier in remembered_widget_value(library_key, list(names)) if identifier in names]
    selected = st.multiselect(tr("选择音频素材库（可多选）"), list(names), default=default if library_key not in st.session_state else None,
                              format_func=lambda identifier: names[identifier], key=library_key)
    return selected


def select_library_bgm(tr, *, key, default_path=""):
    store = AudioLibraryStore()
    selected = _choose_audio_libraries(store, tr, key=key)
    search = st.text_input(tr("搜索曲名、情绪、风格或适用场景"), value=remembered_widget_value(key + "_search", ""), key=key + "_search").strip().casefold()
    instrumental = st.checkbox(tr("只看已分析的无人声音乐"), value=remembered_widget_value(key + "_instrumental", False), key=key + "_instrumental")
    tracks = [track for track in store.tracks(selected) if store.available(track)
              and (not instrumental or track["analysis"].get("vocals") == "none")
              and (not search or search in (Path(track["path"]).stem + " " + str(track["analysis"])).casefold())]
    tracks.sort(key=lambda track: (-track["analysis"].get("bgm_score", 0), track["path"].casefold()))
    choices = {track["id"]: track for track in tracks}
    track_key = key + "_track"
    if track_key not in st.session_state or st.session_state[track_key] not in choices:
        default = remembered_widget_value(track_key, next((item["id"] for item in tracks if item["path"] == default_path), None))
        st.session_state[track_key] = default if default in choices else None
    def label(identifier):
        if identifier is None:
            return tr("不使用背景音乐")
        track = choices[identifier]
        emotions = "、".join(track["analysis"].get("emotions", []))
        return f"{track['library_name']} · {Path(track['path']).stem} · {track['duration']:.0f}s" + (" · " + emotions if emotions else "")
    identifier = st.selectbox(tr("选择 BGM"), [None, *choices], index=None, format_func=label, key=track_key)
    if not identifier:
        if not tracks:
            st.caption(tr("没有符合筛选条件且源文件可用的音频。"))
        return "", ""
    track = choices[identifier]
    st.caption(track["path"])
    if track["analysis"]:
        st.caption(track["analysis"].get("summary", ""))
    _preview(track["path"], key + "_preview", tr)
    return track["path"], Path(track["path"]).stem


def bgm_values(key_prefix="narration"):
    """Read saved music choices without checking each source audio file."""
    value = lambda name, default=None: remembered_widget_value(key_prefix + "_" + name, default)
    libraries = AudioLibraryStore().list_libraries()
    source = value("bgm_source", "path" if value("bgm", "") else "auto" if libraries else "none")
    options = {"bgm_mode": source, "bgm_path": "", "bgm_volume": float(value("bgm_volume", .15))}
    if source == "auto":
        options.update(bgm_library_ids=value("auto_bgm_libraries", [item["id"] for item in libraries]),
            bgm_only_instrumental=bool(value("auto_bgm_instrumental", True)), bgm_auto_strategy=value("bgm_strategy", "sections"),
            bgm_min_seconds=int(value("bgm_min_seconds", 60)), bgm_crossfade=float(value("bgm_crossfade", 2.0)),
            bgm_ducking=bool(value("bgm_ducking", True)))
    elif source == "library":
        options["bgm_path"] = value("bgm_selected_path", value("bgm", ""))
    elif source == "path":
        options["bgm_path"] = value("bgm", "").strip().strip('"')
    return options


@st.fragment
def render_narration_bgm(tr, *, key_prefix="narration"):
    remember_widget_values(key_prefix)
    store = AudioLibraryStore()
    labels = {"auto": "从音频素材库自动配乐", "library": "手动选择一首", "path": "手动输入路径", "none": "不使用背景音乐"}
    default = bgm_values(key_prefix)["bgm_mode"]
    with remembered_expander(tr("BGM 配乐设置"), key=key_prefix + "_bgm_open") as panel:
        if not panel.open:
            return bgm_values(key_prefix)
        source = st.radio(tr("背景音乐来源"), list(labels), index=list(labels).index(default),
                          format_func=lambda value: tr(labels[value]), key=key_prefix + "_bgm_source")
        options = {"bgm_mode": source, "bgm_path": ""}
        if source == "auto":
            options["bgm_library_ids"] = _choose_audio_libraries(store, tr, key=key_prefix + "_auto_bgm")
            options["bgm_only_instrumental"] = st.checkbox(tr("只使用已分析的无人声音乐"), value=remembered_widget_value(key_prefix + "_auto_bgm_instrumental", True), key=key_prefix + "_auto_bgm_instrumental")
            options["bgm_auto_strategy"] = st.selectbox(tr("自动配乐方式"), ["sections", "single"],
                index=["sections", "single"].index(remembered_widget_value(key_prefix + "_bgm_strategy", "sections")),
                format_func=lambda value: tr({"sections": "按大段主题和情绪换曲", "single": "整篇自动选一首"}[value]), key=key_prefix + "_bgm_strategy")
            options["bgm_min_seconds"] = st.number_input(tr("最短换曲间隔（秒）"), min_value=30, max_value=600,
                value=int(remembered_widget_value(key_prefix + "_bgm_min_seconds", 60)), step=15,
                disabled=options["bgm_auto_strategy"] == "single", key=key_prefix + "_bgm_min_seconds",
                help=tr("在已有文案段落边界换曲，实际间隔可以更长；短尾段合并，相邻相同音乐连续播放。"))
            options["bgm_crossfade"] = st.slider(tr("换曲与循环淡化（秒）"), 0.0, 8.0,
                float(remembered_widget_value(key_prefix + "_bgm_crossfade", 2.0)), .5, key=key_prefix + "_bgm_crossfade")
            options["bgm_ducking"] = st.checkbox(tr("旁白发声时自动压低音乐"), value=remembered_widget_value(key_prefix + "_bgm_ducking", True), key=key_prefix + "_bgm_ducking")
            st.caption(tr("复用整首分析标签，由文案模型选择并说明理由；不重新上传音乐，不随字幕或镜头换曲。"))
            if st.button(tr("检查当前可用音乐"), key=key_prefix + "_check_bgm_tracks"):
                try:
                    st.caption(tr("当前可用于自动配乐的已分析音乐：") + str(len(eligible_tracks(options, store))))
                except ValueError as exc:
                    st.warning(str(exc))
        elif source == "library":
            options["bgm_path"] = select_library_bgm(tr, key=key_prefix + "_library_bgm",
                default_path=remembered_widget_value(key_prefix + "_bgm", ""))[0]
            st.session_state[key_prefix + "_bgm_selected_path"] = options["bgm_path"]
        elif source == "path":
            options["bgm_path"] = st.text_input(tr("背景音乐完整路径（可选）"), value=remembered_widget_value(key_prefix + "_bgm", ""), key=key_prefix + "_bgm").strip().strip('"')
            _preview(options["bgm_path"], key_prefix + "_manual_bgm_preview", tr)
        options["bgm_volume"] = st.slider(tr("背景音乐音量"), 0.0, 1.0,
            float(remembered_widget_value(key_prefix + "_bgm_volume", .15)), .05, disabled=source == "none", key=key_prefix + "_bgm_volume")
    remember_widget_values(key_prefix)
    return bgm_values(key_prefix)


@st.fragment
def render_bgm_plan(directory, task_id, tr):
    remember_widget_values()
    path = Path(directory) / "bgm_plan.json"
    if not path.is_file():
        return
    key = "narration_bgm_plan_open_" + task_id
    with remembered_expander(tr("自动配乐详情"), key=key) as panel:
        if not panel.open:
            return
        plan = read_json(path)
        st.caption(tr("不同曲目：") + str(plan["music_count"]) + tr("；音乐区间：") + str(len(plan["cues"])))
        if not plan["music_count"]:
            st.warning(tr("模型认为没有合适的音乐，当前计划全部留白，请检查理由或更换音频库。"))
        rows = []
        for cue in plan["cues"]:
            track = cue["track"]
            rows.append({tr("开始（秒）"): round(cue["start"], 2), tr("结束（秒）"): round(cue["end"], 2),
                         tr("曲目"): Path(track["path"]).stem if track else tr("留白"),
                         tr("音频素材库"): track["library_name"] if track else "",
                         tr("主题／情绪"): "；".join(dict.fromkeys(row["theme"] + " · " + row["emotion"] for row in cue["sections"])),
                         tr("选曲理由"): "；".join(dict.fromkeys(row["reason"] for row in cue["sections"]))})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        preview = plan.get("preview_path", "") if plan.get("render_ready") else ""
        _preview(preview, "narration_bgm_preview_" + task_id, tr)
        st.download_button(tr("下载配乐时间线"), path.read_bytes(), file_name="bgm_plan.json", mime="application/json", key="narration_bgm_plan_" + task_id)


def _model_settings(tr, identifier, saved):
    from webui.components.model_settings import seed_model_choices, render_model_choices, selected_models
    prefix = "audio_library_models_" + identifier
    st.session_state["_narration_settings_prefix"] = prefix
    seed_model_choices(prefix, bindings=saved.get("model_bindings"))
    with st.expander(tr("音频理解模型选择"), expanded=False):
        render_model_choices(tr, prefix, roles=("audio",), backups=False)
    return selected_models(prefix, roles=("audio",))


@st.fragment(run_every="2s")
def _job_status(identifier, tr):
    store = AudioLibraryStore()
    job = store.job(identifier)
    if not job:
        return
    observed = "audio_library_observed_" + identifier
    previous = st.session_state.get(observed)
    active = runner.running("audio-library:" + identifier)
    st.session_state[observed] = active
    if previous is True and not active:
        st.rerun(scope="app")
    st.progress(max(0, min(1, job.get("progress", 0) / 100)))
    if job.get("state") == "failed":
        st.error(job.get("message", ""))
    elif job.get("state") == "complete":
        st.success(job.get("message", ""))
    else:
        st.info(job.get("message", ""))
    st.caption(tr("本次模型调用：") + str(job.get("calls", 0)))
    if active:
        if st.button(tr("停止当前音频任务"), key="stop_audio_library_" + identifier):
            runner.cancel("audio-library:" + identifier)
    elif job.get("state") == "running":
        st.caption(tr("上次程序已中断，可重新操作；成功登记和分析会复用。"))


def render_audio_library_panel(tr):
    st.header(tr("音频素材库"))
    st.caption(tr("登记原音频路径，分析整首音乐并保存标签，在视频制作中筛选、试听和选择 BGM。"))
    store = AudioLibraryStore()
    libraries = store.list_libraries()
    notice = st.session_state.pop("audio_library_notice", None)
    if notice:
        st.info(notice)
    with st.expander(tr("新建音频素材库"), expanded=not libraries):
        with st.form("new_audio_library"):
            name = st.text_input(tr("音频素材库名称"), placeholder=tr("例如：影视解析配乐"))
            if st.form_submit_button(tr("创建音频素材库")):
                try:
                    st.session_state["audio_library_choice"] = store.create(name)
                    st.rerun()
                except (OSError, ValueError) as exc:
                    st.error(str(exc))
    if not libraries:
        return
    names = {item["id"]: item["name"] for item in libraries}
    if st.session_state.get("audio_library_choice") not in names:
        st.session_state["audio_library_choice"] = next(iter(names))
    identifier = st.selectbox(tr("选择音频素材库"), list(names), format_func=lambda value: names[value], key="audio_library_choice")
    library = store.library(identifier)
    saved = library["settings"]
    busy = runner.library_busy([identifier], prefix="audio-library:")
    stale = library["job"].get("state") == "running" and not runner.running("audio-library:" + identifier)
    if busy and not runner.running("audio-library:" + identifier):
        st.info(tr("此音频库正在用于自动配乐，任务结束后可以扫描、分析或移除登记。"))
    interrupted_confirmed = st.checkbox(tr("确认旧程序和其他窗口没有运行此音频库任务"), key="audio_interrupted_" + identifier) if stale else False
    with st.expander(tr("导入音频目录"), expanded=not store.tracks([identifier])):
        folder = folder_input(tr("音频目录"), value=saved.get("folder", str(Path(config.root_dir) / "resource" / "bgm")),
                              key="audio_folder_" + identifier, disabled=busy, tr=tr)
        recursive = st.checkbox(tr("扫描子目录"), value=saved.get("recursive", True), key="audio_recursive_" + identifier, disabled=busy)
        st.caption(tr("只登记音频原路径，不复制音乐文件；再次扫描更新原路径的登记，成功分析可复用。"))
        if st.button(tr("扫描并登记音频"), disabled=busy, key="audio_import_" + identifier):
            try:
                store.save_settings(identifier, {**store.library(identifier)["settings"], "folder": folder, "recursive": recursive})
                store.start_job(identifier, "import", lambda stop: store.register_directory(identifier, folder, recursive, stop))
                st.rerun()
            except (OSError, ValueError) as exc:
                st.error(str(exc))
    _job_status(identifier, tr)
    tracks = store.tracks([identifier])
    if tracks:
        st.subheader(tr("已登记音频"))
        search = st.text_input(tr("搜索已登记音频"), key="audio_filter_" + identifier).strip().casefold()
        visible = [track for track in tracks if not search or search in (Path(track["path"]).stem + " " + str(track["analysis"])).casefold()]
        statuses = {"pending": "待分析", "ready": "已分析", "failed": "分析失败"}
        if visible:
            rows = [{"select": False, "name": Path(track["path"]).stem, "duration": round(track["duration"], 1),
                     "status": tr(statuses.get(track["status"], track["status"])), "emotions": "、".join(track["analysis"].get("emotions", [])),
                     "score": track["analysis"].get("bgm_score"), "vocals": track["analysis"].get("vocals", ""), "available": store.available(track)} for track in visible]
            edited = st.data_editor(pd.DataFrame(rows), hide_index=True, width="stretch", disabled=[field for field in rows[0] if field != "select"],
                key="audio_tracks_" + identifier + "_" + str(hash(tuple(track["id"] for track in visible))),
                column_config={"select": st.column_config.CheckboxColumn(tr("选择")), "name": tr("曲名"), "duration": tr("时长（秒）"),
                               "status": tr("状态"), "emotions": tr("情绪"), "score": tr("旁白适配评分"), "vocals": tr("人声分类"), "available": tr("源文件可用")})
            selected = [visible[index]["id"] for index, row in enumerate(edited.to_dict("records")) if row["select"]]
            details = {track["id"]: track for track in visible}
            detail_key = "audio_detail_" + identifier
            if st.session_state.get(detail_key) not in details:
                st.session_state[detail_key] = next(iter(details))
            track = details[st.selectbox(tr("试听／查看音频"), list(details), format_func=lambda value: Path(details[value]["path"]).name, key=detail_key)]
            st.caption(track["path"])
            _preview(track["path"], "audio_preview_" + identifier, tr)
            if track["error"]:
                st.error(track["error"])
            if track["analysis"]:
                with st.expander(tr("BGM 分析结果")):
                    st.write(track["analysis"].get("summary", ""))
                    st.write(track["analysis"].get("reason", ""))
                    st.json(track["analysis"])
            with st.expander(tr("移除所选登记")):
                confirmed = st.checkbox(tr("只移除登记和分析，原音频与生成任务保留"), key="audio_remove_ack_" + identifier)
                if st.button(tr("移除所选音频"), disabled=busy or not selected or not confirmed or (stale and not interrupted_confirmed), key="audio_remove_" + identifier):
                    try:
                        store.remove_tracks(identifier, selected, interrupted_confirmed)
                        st.session_state["audio_library_notice"] = tr("所选音频登记已移除。")
                        st.rerun()
                    except (OSError, ValueError) as exc:
                        st.error(str(exc))
        else:
            selected = []
            st.info(tr("没有符合搜索条件的音频。"))
        st.subheader(tr("分析整首 BGM"))
        settings = _model_settings(tr, identifier, saved)
        st.caption(tr("每首完整音频会发送到上面配置的模型接口；必要时完整转为 MP3。超限明确报错，不截取。成功结果会缓存，失败可继续。"))
        concurrency = st.number_input(tr("音频 API 请求并发数"), min_value=1,
            value=int(saved.get("analysis_concurrency", runner.DEFAULT_API_CONCURRENCY)), step=1, disabled=busy,
            key="audio_analysis_concurrency_" + identifier,
            help=tr("默认32，可输入任意正整数，不设固定最大值。每首单独完整分析与保存；遇到429暂时降速，停止后保留成功结果。"))
        force = st.checkbox(tr("重新分析已有成功结果"), key="audio_force_" + identifier)
        left, right = st.columns(2)
        chosen = None
        if left.button(tr("分析所选音频"), disabled=busy or not selected, key="audio_analyze_selected_" + identifier):
            chosen = selected
        if right.button(tr("分析／继续全部音频"), disabled=busy, key="audio_analyze_all_" + identifier):
            chosen = [track["id"] for track in tracks]
        if chosen:
            try:
                from webui.components.model_settings import check_model_choices, selected_bindings
                prefix = "audio_library_models_" + identifier
                check_model_choices(prefix, ["audio"])
                ModelGateway(settings).require("audio")
                store.save_settings(identifier, {**store.library(identifier)["settings"], "analysis_concurrency": concurrency,
                                                 "model_bindings": selected_bindings(prefix)})
                store.start_job(identifier, "analysis", lambda stop: store.analyze(identifier, chosen, settings, force, stop, concurrency=concurrency))
                st.rerun()
            except (OSError, ValueError) as exc:
                st.error(str(exc))
        with st.expander(tr("下载音频登记与分析")):
            import json
            st.download_button(tr("下载分析 JSON"), json.dumps(tracks, ensure_ascii=False, indent=2), file_name="audio-library.json", mime="application/json", key="audio_export_" + identifier)
    with st.expander(tr("删除音频素材库")):
        st.caption(tr("删除此库的登记、分析和诊断日志；原音频及已有生成任务保留。"))
        confirmation = st.text_input(tr("输入音频库名称确认删除"), key="audio_delete_name_" + identifier)
        if st.button(tr("删除此音频素材库"), disabled=busy or confirmation.strip() != library["name"] or (stale and not interrupted_confirmed), key="audio_delete_" + identifier):
            try:
                st.session_state["audio_library_notice"] = store.delete_library(identifier, confirmation, interrupted_confirmed)
                st.session_state.pop("audio_library_choice", None)
                st.rerun()
            except (OSError, ValueError) as exc:
                st.error(str(exc))
