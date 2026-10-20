"""Local directory import and reusable material-library management."""

from pathlib import Path
import json

import pandas as pd
import streamlit as st

from app.config import config
from app.models.library import ImportMapping
from app.services.ass_subtitle import parse_ass_events, plain_ass_lines, select_chinese_ass
from app.services.library import runner
from app.services.library.frame_extractor import inspect_video
from app.services.library.gateway import ModelGateway
from webui.components.model_settings import seed_model_choices, render_model_choices, selected_models, selected_bindings, check_model_choices
from app.services.library.indexer import VIDEO_EXTENSIONS, SUBTITLE_EXTENSIONS, scan_directory, suggest_mappings, import_materials, build_library, audit_subtitles, restore_subtitle_text, rollback_subtitle_text
from app.services.library.store import LibraryStore, _json_access, library_ids, read_json, write_json
from app.services.subtitle_text import decode_subtitle_bytes
from webui.utils.file_utils import folder_input, remembered_expander


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
            message += "\n" + previous["message"] + "\n已保存的成功场景和缓存可继续使用。"
        store.job(library_id, kind, state=state, progress=previous.get("progress", 0), message=message)
    def work(stop):
        try:
            store.job(library_id, kind, state="running", message="任务准备开始")
            callback(stop)
        except runner.JobCancelled as exc:
            finish("paused", exc)
        except Exception as exc:
            finish("failed", exc)
    if not runner.start(key, work):
        raise ValueError("视频素材库或关联生成任务正在运行，请等待任务结束。")


@st.fragment(run_every="2s")
def _job_status(library_id, tr):
    store = LibraryStore()
    key = "library:" + library_id
    jobs = [job for job in (store.job(library_id, "import"), store.job(library_id, "build"), store.job(library_id, "subtitle")) if job]
    if not jobs:
        return
    job = max(jobs, key=lambda item: item["updated"])
    observed_key = "library_observed_state_" + library_id
    previous_state = st.session_state.get(observed_key)
    st.session_state[observed_key] = job["state"]
    if previous_state == "running" and job["state"] != "running":
        st.rerun(scope="app")
    st.progress(max(0, min(1, job["progress"] / 100)))
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
        with st.expander(tr("分析错误详情"), expanded=job["state"] in {"failed", "partial"}):
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
    with st.expander(tr("中文字幕筛选与预览")):
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
    return settings


@st.fragment
def _work_panel(store, library_id, tr, busy):
    from app.services.library.knowledge import save_work, fetch_bangumi
    with remembered_expander(tr("作品与资料"), key="work_panel_"+library_id) as panel:
        if not panel.open:
            return
        work = store.cache_get(library_id, "work") or {}
        title = st.text_input(tr("作品名称"), value=work.get("title",""), key="work_title_"+library_id)
        adaptation = st.text_input(tr("改编版本"), value=work.get("adaptation","动画"), key="work_adaptation_"+library_id,
                                   help=tr("例如电视动画第一版。小说、漫画和动画不要混用为画面事实。"))
        season = st.text_input(tr("季／篇章"), value=work.get("season",""), key="work_season_"+library_id)
        upload = st.file_uploader(tr("导入作品资料 MD／TXT"),type=["md","txt"],key="work_file_"+library_id)
        if upload is not None and st.button(tr("载入文档"),key="work_load_"+library_id):
            st.session_state["work_notes_"+library_id] = upload.getvalue().decode("utf-8-sig")
        if "work_notes_"+library_id not in st.session_state:
            st.session_state["work_notes_"+library_id] = work.get("notes","")
        notes = st.text_area(tr("人物别名、时期与剧情说明"),key="work_notes_"+library_id,height=180)
        st.caption(tr("可用“## 第 4 集”区分单集资料；未分集的说明用于全部素材。资料辅助理解，不代替本地证据。"))
        cards = st.text_area(tr("人物卡与参考镜头（JSON，可选）"),
            value=json.dumps(work.get("characters",[]),ensure_ascii=False,indent=2),key="work_cards_"+library_id,
            help=tr('格式：[{"name":"鲁迪","aliases":["鲁迪乌斯"],"period":"现世幼年","reference_ranges":["R..."],"identity_basis":"人工核对"}]'))
        mapping = st.text_input(tr("集号映射（JSON，可选）"),value=json.dumps(work.get("episode_map",{}),ensure_ascii=False),
                                key="work_mapping_"+library_id,help=tr('例如 {"1":13}：本库第1集对应外部第13集。'))
        if st.button(tr("保存作品资料"),disabled=busy,key="save_work_"+library_id):
            try:
                characters, episode_map = json.loads(cards), json.loads(mapping)
                if not isinstance(characters,list) or any(not isinstance(c,dict) or not isinstance(c.get("name"),str) for c in characters):
                    raise ValueError("人物卡须为带name的对象列表。")
                if not isinstance(episode_map,dict) or any(not str(k).isdigit() or not isinstance(v,(int,float)) for k,v in episode_map.items()):
                    raise ValueError("集号映射须为编号到编号的对象。")
                save_work(store,library_id,{**work,"title":title,"adaptation":adaptation,"season":season,
                                          "notes":notes,"characters":characters,"episode_map":episode_map})
                st.success(tr("资料已保存。继续拉片时仅重新分析资料身份改变的范围。"))
            except (ValueError,OSError) as exc:
                st.error(str(exc))
        source = st.text_input(tr("Bangumi 作品链接或编号"),value=work.get("external",{}).get("source",""),key="work_source_"+library_id)
        if st.button(tr("获取并保存公开资料"),disabled=busy or not source.strip(),key="fetch_work_"+library_id):
            try:
                with st.spinner(tr("获取作品、角色与章节资料")):
                    external = fetch_bangumi(source, config.proxy)
                current = store.cache_get(library_id,"work") or work
                save_work(store,library_id,{**current,"external":external})
                st.success(tr("公开资料已保存；请核对对应的动画版本和集号。"))
            except Exception as exc:
                st.warning(tr("外部资料获取失败，已有资料保留，本地拉片仍可继续：")+str(exc))
        if work.get("external"):
            st.caption(tr("资料来源：")+work["external"]["source"])


@st.fragment
def _content_panel(store, library_id, tr, busy):
    with remembered_expander(tr("内容浏览与质量修正"),key="content_panel_"+library_id) as panel:
        if not panel.open:
            return
        ranges, events = store.records(library_id), store.records(library_id,"contents")
        ready = sum(r.get("analysis_state")=="ready" for r in ranges)
        st.caption(f"{len(events)} 个内容段 · {len(ranges)} 个镜头范围 · {ready} 个观察就绪 · {len(ranges)-ready} 个待确认")
        quality = read_json(store.directory(library_id)/"analysis-quality.json",default={})
        if quality.get("issues"):
            st.warning(tr("最近拉片存在未完成项，继续会复用成功记录。"))
            st.dataframe(quality["issues"][:100],width="stretch",hide_index=True)
        usage = quality.get("token_usage",{}).get("current",{})
        if usage:
            st.caption(f"最近拉片已报告 Token：{sum(r.get('total_tokens',0) for r in usage.values()):,} · "
                       f"用量未报告请求：{sum(r.get('unknown',0) for r in usage.values())}")
        if not events:
            st.info(tr("尚无内容目录；请建立或继续拉片。"))
            return
        query = st.text_input(tr("搜索事件、人物、时期或地点"),key="content_search_"+library_id)
        by_id = {r["id"]:r for r in ranges}
        visible = [e for e in events if not query or query.casefold() in
                   (json.dumps(e,ensure_ascii=False)+json.dumps([by_id.get(i,{}) for i in e["range_ids"]],ensure_ascii=False)).casefold()]
        page_key = "content_page_"+library_id
        pages = max(1,(len(visible)+9)//10)
        st.session_state[page_key] = min(max(1,int(st.session_state.get(page_key,1))),pages)
        page = st.number_input(tr("内容目录页码"),min_value=1,max_value=pages,step=1,key=page_key)
        for event in visible[(page-1)*10:page*10]:
            with st.expander(f"第{event['episode']}集 · {event['start']:.1f}–{event['end']:.1f}s · {event.get('title','')}"):
                st.write(event["summary"])
                st.caption(event["id"])
                rid = st.selectbox(tr("镜头范围"),event["range_ids"],key="range_"+event["id"])
                item = by_id[rid]
                frames = sorted((store.directory(library_id)/"frames"/item["fingerprint"]/rid).glob("*.jpg"))
                if frames:
                    st.image([str(f) for f in frames],width=160)
                st.write(item.get("summary",""))
                st.caption(f"时期：{item.get('period','未知')} · 观察：{item.get('coverage','')} · 来源：{item.get('content_kind','unknown')}")
                st.dataframe(item.get("observations",[]),width="stretch",hide_index=True)
                st.caption(tr("抽帧观察的证据时间不会自动扩展到整个未采样区间。"))
                signatures = set(item.get("frame_signatures",[]))
                similar = [r["id"] for r in ranges if r["id"] != rid and signatures and
                           len(signatures & set(r.get("frame_signatures",[]))) >= min(2,len(signatures))]
                if similar:
                    st.info(tr("相似采样画面（请人工确认是否重复）：")+"、".join(similar[:10]))
                if st.button(tr("预览原片此处"),key="preview_"+event["id"]):
                    st.video(item["path"],start_time=int(item["start"]),end_time=max(int(item["start"])+1,int(item["end"])))
                if st.button(tr("补分析当前镜头"),disabled=busy,key="enrich_"+event["id"]):
                    from app.services.model_profiles import resolve_models
                    try:
                        saved = next(r["settings"] for r in store.list_libraries() if r["id"]==library_id)
                        settings = {**saved.get("build_settings",{}),"asset_ids":[item["asset_id"]],
                                    "range_ids":[rid],"reanalyze":True}
                        gateway = ModelGateway(resolve_models(settings.get("model_bindings")))
                        _launch_library_job(store,library_id,"build",lambda stop:build_library(store,library_id,gateway,settings,stop))
                        st.rerun(scope="app")
                    except (ValueError,OSError) as exc:
                        st.error(str(exc))
                with st.form("correct_"+event["id"]+"_"+rid):
                    summary = st.text_area(tr("更正当前内容"),value=item.get("summary",""))
                    period = st.text_input(tr("时期"),value=item.get("period",""))
                    kind = st.selectbox(tr("画面属性"),["story","credits","opening","black","unknown"],
                                        index=["story","credits","opening","black","unknown"].index(item.get("content_kind","unknown")))
                    duplicate = st.text_input(tr("已确认重复画面组（可选）"),value=item.get("duplicate_group",""),
                                              help=tr("人工确认是相同实际画面后填同一组名；相似构图不应归为重复。"))
                    people = st.text_area(tr("出镜人物与身份依据（JSON）"),value=json.dumps(item.get("visible_people",[]),ensure_ascii=False))
                    observations = st.text_area(tr("观察更正（JSON，保留当前帧／字幕引用）"),value=json.dumps(item["observations"],ensure_ascii=False))
                    reason = st.text_input(tr("更正依据"))
                    if st.form_submit_button(tr("保存本镜头更正"),disabled=busy):
                        try:
                            store.correct_range(library_id,rid,{"summary":summary,"period":period,"content_kind":kind,
                                "duplicate_group":duplicate,"visible_people":json.loads(people),"observations":json.loads(observations)},reason)
                            st.success(tr("已保存修订记录；相关匹配会重新确认，其他镜头观察保持。"))
                        except (ValueError,OSError) as exc:
                            st.error(str(exc))


def render_library_panel(tr):
    st.header(tr("视频素材库"))
    st.caption(tr("视频直接引用原路径。字幕、抽帧和分析结果保存在本项目，可供不同文案复用。"))
    store = LibraryStore()
    notice = st.session_state.pop("library_delete_notice", None)
    if notice:
        st.info(notice)
    with st.expander(tr("新建视频素材库"), expanded=not store.list_libraries()):
        with st.form("new_material_library"):
            name = st.text_input(tr("视频素材库名称"), placeholder=tr("例如：无职转生第一季"))
            if st.form_submit_button(tr("创建视频素材库")):
                try:
                    identifier = store.create(name)
                    st.session_state["active_library_id"] = identifier
                    st.rerun()
                except ValueError as exc:
                    st.error(str(exc))
    library_id = choose_library(store, tr)
    if not library_id:
        return
    library = next(item for item in store.list_libraries() if item["id"] == library_id)
    saved = library["settings"]
    busy = runner.library_busy(library_id)
    current = store.schema(library_id) == 2
    if not current:
        st.warning(tr("此库为旧窗口库。2.8 使用独立拉片库；可复制登记信息与字幕后重新分析，原库保留。"))
        if st.button(tr("创建 2.8 拉片副本"), disabled=busy, key="clone_content_"+library_id):
            try:
                new_id = store.clone_for_analysis(library_id, library["name"]+" · 2.8")
                st.session_state["active_library_id"] = new_id
                st.rerun()
            except (OSError,ValueError) as exc:
                st.error(str(exc))
    else:
        _work_panel(store,library_id,tr,busy)
    if busy and not runner.running("library:" + library_id):
        st.info(tr("关联生成任务正在使用视频素材库，请等待完成后修改素材。"))
    with st.expander(tr("导入本地目录"), expanded=not store.assets(library_id)):
        st.caption(tr("请输入运行 NarraLens 的电脑上的目录，大视频无需浏览器上传。"))
        video_dir = folder_input(tr("视频目录"), value=saved.get("video_dir", str(Path(config.root_dir) / "resource" / "videos")), key="lib_video_dir_" + library_id, tr=tr)
        subtitle_dir = folder_input(tr("字幕目录（可选）"), value=saved.get("subtitle_dir", str(Path(config.root_dir) / "resource" / "srt")), key="lib_subtitle_dir_" + library_id, tr=tr)
        recursive = st.checkbox(tr("扫描子目录"), value=saved.get("recursive", True), key="lib_recursive_" + library_id)
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
        if draft:
            st.caption(tr("多字幕选择说明"))
            no_subtitle = tr("不导入字幕")
            choose_subtitle = tr("请选择字幕")
            subtitle_options = [no_subtitle, choose_subtitle] + draft["subtitles"]
            rows = [{"import": True, "episode": row["episode"], "video": row["video"],
                     "subtitle": row["subtitle"] or (choose_subtitle if row["ambiguous"] else no_subtitle)} for row in draft["mappings"]]
            edited = st.data_editor(pd.DataFrame(rows), hide_index=True, width="stretch",
                disabled=["video"], key="library_mapping_" + library_id + "_" + str(hash(str(rows))),
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
                    store.save_settings(library_id, {**saved, "video_dir": video_dir, "subtitle_dir": subtitle_dir, "recursive": recursive, "subtitle_settings": subtitle_settings})
                    _launch_library_job(store, library_id, "import", lambda stop: import_materials(store, library_id, mappings, subtitle_settings, stop))
                    st.rerun()
                except (OSError, ValueError, TypeError) as exc:
                    st.error(str(exc))
    _job_status(library_id, tr)
    assets = store.assets(library_id)
    if assets:
        st.subheader(tr("已登记素材"))
        status_labels = {"pending": "待分析", "ready": "已完成", "partial": "部分完成（可用）", "failed": "分析失败", "changed": "原文件已变化"}
        registered = st.data_editor(pd.DataFrame([{"remove": False, "episode": item["episode"], "video": Path(item["path"]).name,
            "duration": round(item["duration"], 1), "subtitle": Path(item["subtitle_path"]).name if item.get("subtitle_path") else "",
            "status": tr(status_labels.get(item["status"], item["status"])), "available": Path(item["path"]).is_file()} for item in assets]),
            hide_index=True, width="stretch", disabled=["episode", "video", "duration", "subtitle", "status", "available"],
            key="registered_assets_" + library_id + "_" + str(hash(tuple(item["id"] for item in assets))),
            column_config={"remove": st.column_config.CheckboxColumn(tr("移除"), required=True), "episode": tr("集数"), "video": tr("视频"), "duration": tr("时长（秒）"),
                           "subtitle": tr("字幕"), "status": tr("分析状态"), "available": tr("源文件可用")})
        selected = [assets[index]["id"] for index, row in enumerate(registered.to_dict("records")) if row["remove"]]
        with st.expander(tr("字幕资料校验与恢复")):
            st.caption(tr("校验登记的ASS正文是否被误删。恢复仅在本地生成新版字幕与窗口文本，保留原ASS、旧派生文件、抽帧和视觉分析；字幕剧情语义会标为待补，不调用API。"))
            report_key = "subtitle_audit_" + library_id
            if st.button(tr("校验字幕"), disabled=busy, key=report_key + "_check"):
                try:
                    with runner.library_operation(library_id):
                        st.session_state[report_key] = audit_subtitles(store, library_id)
                except (OSError, ValueError) as exc:
                    st.error(str(exc))
            audit = st.session_state.get(report_key)
            if audit is not None:
                labels = {"ok":"正常","recovery_needed":"正文待恢复","lyrics_only":"只有歌词/注释，无剧情台词","error":"原件或路径异常"}
                st.dataframe(pd.DataFrame([{tr("集数"):r["episode"],tr("视频"):r["video"],tr("校验"):tr(labels.get(r["state"],r["state"])),
                    tr("正文条数"):r.get("body_events",0),tr("缺失正文"):r.get("missing_body_events",0),tr("当前字幕条数"):r.get("current_cues",0),
                    tr("恢复后条数"):r.get("restored_cues",0),tr("说明"):r.get("error","")} for r in audit]), hide_index=True, width="stretch")
                count = sum(row["state"] == "recovery_needed" for row in audit)
                errors = any(row["state"] == "error" for row in audit)
                if count:
                    st.warning(tr("待恢复正文的素材数：") + str(count))
                if st.button(tr("恢复正文与本地索引"), disabled=busy or not count or errors, key=report_key + "_restore"):
                    try:
                        _launch_library_job(store, library_id, "subtitle", lambda stop: restore_subtitle_text(store, library_id, stop))
                        st.session_state.pop(report_key, None)
                        st.rerun()
                    except (OSError, ValueError) as exc:
                        st.error(str(exc))
                st.caption(tr("恢复后可直接从3. 寻找画面继续；如需补充剧情语义，使用下方原有建立／继续分析入口，纯画面缓存继续复用。"))
            migrations = [m for m in store.subtitle_migrations(library_id) if m.get("state") == "committed" and m.get("asset_id") in {item["id"] for item in assets}]
            if migrations:
                st.caption(tr("需要回退时按单个素材恢复登记与窗口；若恢复后已有新分析，会停止回退以保留后续工作。新旧字幕文件均保留。"))
                selected_migration = st.selectbox(tr("本地恢复记录"),[m["id"] for m in migrations],
                    format_func=lambda value: next(tr("第")+str(item["episode"])+tr("集")+" · "+Path(item["path"]).name for m in migrations for item in assets if m["id"]==value and item["id"]==m["asset_id"]),key=report_key+"_migration")
                if st.button(tr("回退该素材的本地恢复"),disabled=busy,key=report_key+"_rollback"):
                    try:
                        rollback_subtitle_text(store,library_id,selected_migration)
                        st.session_state.pop(report_key,None)
                        st.rerun()
                    except (OSError,ValueError) as exc:
                        st.error(str(exc))
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
        with st.expander(tr("重新绑定移动后的原视频")):
            asset_id = st.selectbox(tr("选择素材"), [item["id"] for item in assets], format_func=lambda value: Path(next(item["path"] for item in assets if item["id"] == value)).name, key="rebind_asset_" + library_id)
            new_path = st.text_input(tr("新视频完整路径"), key="rebind_path_" + library_id)
            st.caption(tr("重绑定会核对文件指纹、大小和时长，确认是同一个原视频后保留分析结果。"))
            if st.button(tr("确认重绑定"), disabled=busy, key="rebind_button_" + library_id):
                try:
                    store.rebind(library_id, asset_id, inspect_video(new_path.strip().strip('"')))
                    st.success(tr("路径已更新，素材ID和分析结果保留。"))
                except (OSError, ValueError) as exc:
                    st.error(str(exc))
        st.subheader(tr("建立或继续分析视频素材库"))
        st.caption(tr("点击后调用本库选择的模型 API；已有分析会复用，停止后保留成功结果。"))
        model_prefix = "video_library_models_" + library_id
        st.session_state["_narration_settings_prefix"] = model_prefix
        seed_model_choices(model_prefix, bindings=saved.get("build_settings", {}).get("model_bindings"))
        with st.expander(tr("模型选择")):
            render_model_choices(tr, model_prefix, roles=("text", "vision"))
        build_saved = saved.get("build_settings", {})
        visual = True
        st.caption(tr("2.8 拉片联合使用图片和字幕；保存实际观察及证据后供多个文案复用。"))
        window = st.number_input(tr("长镜头采样间距（秒）"), min_value=10, max_value=120, value=int(build_saved.get("window_seconds", 15)), step=5, key="build_window_" + library_id, disabled=busy)
        left, right = st.columns(2)
        with left:
            batch_size = st.number_input(tr("每批最多分析镜头范围数"), min_value=1, max_value=128,
                value=int(build_saved.get("batch_size", 50)), step=1, key="build_batch_" + library_id, disabled=busy,
                help=tr("按同一视频打包。根据镜头时长和字幕密度采样1–6帧；长镜头采样范围不是事件分段。超预算时自动减小批量。"))
        with right:
            concurrency = st.number_input(tr("API 请求并发数"), min_value=1,
                value=int(build_saved.get("concurrency", runner.DEFAULT_API_CONCURRENCY)), step=1, key="build_concurrency_" + library_id, disabled=busy,
                help=tr("可输入任意正整数，不设固定最大值。跨视频共享请求池，联合观察与后备请求一起调度；遇到429会暂时降低实际并发。"))
        st.caption(tr("批量和并发只影响调度，成功场景仍逐条保存。停止后不再发新请求，正在返回的请求可能需要等待；成功结果会保留。"))
        selected_assets = st.multiselect(tr("本次拉片素材（留空为全部）"),[a["id"] for a in assets],
            format_func=lambda aid:next(f"第{a['episode']}集 · {Path(a['path']).name}" for a in assets if a["id"]==aid),key="build_assets_"+library_id)
        reanalyze = st.checkbox(tr("重新分析所选素材的已完成观察"),value=False,key="build_force_"+library_id,
            help=tr("正常继续只补缺失；改变模型不会清除成功观察。明确重新分析会产生新的API调用。"))
        st.caption(tr("成功观察立即保存；事件目录未完成的部分会明确列为待处理，继续不重付成功观察费用。"))
        if st.button(tr("建立／继续视频素材库"), disabled=busy or not current, key="build_library_" + library_id):
            try:
                check_model_choices(model_prefix, ["text", "text_backup", *(["vision", "vision_backup"] if visual else [])])
                gateway = ModelGateway(selected_models(model_prefix))
                gateway.require("text")
                if visual:
                    gateway.require("vision")
                if not visual and any(not item.get("processed_path") for item in assets):
                    raise ValueError(tr("无字幕素材需要启用视觉分析。"))
                build_settings = {"visual": visual, "window_seconds": window, "reanalyze": reanalyze, "asset_ids":selected_assets,
                                  "batch_size": batch_size, "concurrency": concurrency,
                                  "model_bindings": selected_bindings(model_prefix)}
                current_settings = next(item["settings"] for item in store.list_libraries() if item["id"] == library_id)
                store.save_settings(library_id, {**current_settings, "build_settings": build_settings})
                _launch_library_job(store, library_id, "build", lambda stop: build_library(store, library_id, gateway, build_settings, stop))
                st.rerun()
            except ValueError as exc:
                st.error(str(exc))
    if current:
        _content_panel(store,library_id,tr,busy)

    with st.expander(tr("删除视频素材库")):
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
