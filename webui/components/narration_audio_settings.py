"""Independent TTS settings for manuscript-driven tasks."""

from pathlib import Path

import streamlit as st

from app.config import config
from app.services.library import runner
from app.services.mimo_tts import DEFAULT_BASE_URL, MAX_TTS_HAN, MODELS, VOICES
from app.services.narration_audio import audio_break_candidates, manual_audio_ranges
from app.services.library.store import stable_hash
from app.services.narration_alignment import LEGACY_MODE, LONG_MODE, default_alignment_options
from webui.utils.file_utils import remembered_widget_value, remembered_expander, remember_widget_values
from webui.components.model_settings import render_model_choice, selected_bindings, model_config_link
from app.services.model_profiles import alignment_profile


def audio_mode_label(mode):
    return {LEGACY_MODE: "原模式：按字幕单元配音", LONG_MODE: "长段连续配音"}[mode]


@st.cache_data(show_spinner=False, max_entries=16)
def _manual_candidates(manuscript):
    return audio_break_candidates(manuscript)


def initial_audio_options(manuscript):
    """Explicit new-project defaults, without reading page widgets or loading a model."""
    rows = _manual_candidates(manuscript) if manuscript.strip() else []
    ends = {row["end"] for row in rows if row["paragraph_end"]}
    if rows:
        ends.add(len(manuscript))
    return {"audio_mode": LONG_MODE, "tts_chunking": "manual", "tts_break_after": sorted(ends),
            "tts_break_text_hash": stable_hash(manuscript), "tts_budget": runner.DEFAULT_CALL_BUDGET,
            "tts_block_chars": 800, "alignment_enabled": True, **default_alignment_options()}


def _manual_state(manuscript, prefix):
    options = {"tts_break_text_hash": stable_hash(manuscript), "tts_break_after": []}
    candidates = _manual_candidates(manuscript) if manuscript.strip() else []
    rows = [{"split": row["paragraph_end"] or index == len(candidates) - 1,
             "text": row["text"], "han": row["han"], "end": row["end"]} for index, row in enumerate(candidates)]
    document_key = options["tts_break_text_hash"][:16]
    selection_key = prefix + "_manual_selection_" + document_key
    previous = remembered_widget_value(prefix + "_manual_breaks_" + document_key, {})
    edits = previous.get("edited_rows", {}) if isinstance(previous, dict) else {}
    selected = dict(remembered_widget_value(selection_key, {}))
    ends = remembered_widget_value(prefix + "_tts_break_after", [])
    matching = remembered_widget_value(prefix + "_tts_break_text_hash") == options["tts_break_text_hash"]
    for index, row in enumerate(rows):
        default = selected.get(row["end"], row["end"] in ends if matching else
            edits.get(index, edits.get(str(index), {})).get("split", row["split"]))
        selected[row["end"]] = index == len(rows) - 1 or bool(remembered_widget_value(
            f"{prefix}_manual_boundary_{document_key}_{row['end']}", default))
    st.session_state[selection_key] = selected
    if rows:
        options["tts_break_after"] = sorted({row["end"] for row in rows if selected[row["end"]]} | {len(manuscript)})
    return options, rows, selected, document_key


@st.fragment
def _manual_blocks(tr, manuscript, prefix="narration"):
    remember_widget_values(prefix)
    options, rows, selected, document_key = _manual_state(manuscript, prefix)
    if remembered_widget_value(prefix + "_tts_break_text_hash", options["tts_break_text_hash"]) != options["tts_break_text_hash"]:
        st.info(tr("原文已改变，请展开手动配音分段，重新确认句尾断点。"))
    with remembered_expander(tr("手动配音分段"), key=prefix + "_manual_blocks_open") as panel:
        if panel.open:
            if not rows:
                st.info(tr("先输入包含可配音字词的完整文案，再设置分段。"))
            else:
                st.caption(tr("始终按原文顺序显示。勾选表示在本句句尾结束当前配音块，下一句开始新块。每页50行，翻页保留勾选；最后一行固定结束。"))
                pages = (len(rows) + 49) // 50
                page_key = prefix + "_manual_page_" + document_key
                st.session_state[page_key] = max(1, min(int(remembered_widget_value(page_key, 1)), pages))
                page = int(st.number_input(tr("分段表页码"), 1, pages, step=1, key=page_key))
                offset = (page - 1) * 50
                st.caption(f"{offset + 1}–{min(offset + 50, len(rows))} / {len(rows)}")
                with st.container(height=380, border=True):
                    columns = st.columns([1.8, 8, .8])
                    for column, label in zip(columns, ("在本句句尾结束配音块", "原文句子／段落", "汉字数")):
                        column.markdown("**" + tr(label) + "**")
                    for index in range(offset, min(offset + 50, len(rows))):
                        row = rows[index]
                        split_col, text_col, count_col = st.columns([1.8, 8, .8], vertical_alignment="center")
                        last = index == len(rows) - 1
                        with split_col:
                            checked = st.checkbox(tr("在本句句尾结束配音块"), value=bool(selected[row["end"]]),
                                key=f"{prefix}_manual_boundary_{document_key}_{row['end']}",
                                label_visibility="collapsed", disabled=last,
                                help=tr("读完本行后结束当前配音块；下一行开始新的配音块。"))
                        selected[row["end"]] = True if last else checked
                        text_col.text(f"{index + 1}. {row['text']}")
                        count_col.caption(str(row["han"]))
    options["tts_break_after"] = sorted({row["end"] for row in rows if selected[row["end"]]} | ({len(manuscript)} if rows else set()))
    if panel.open and rows:
        st.session_state[prefix + "_tts_break_text_hash"] = options["tts_break_text_hash"]
        st.session_state[prefix + "_tts_break_after"] = options["tts_break_after"]
    st.caption(tr("本次配音分段：") + str(len(options["tts_break_after"])) + tr(" 块；已有匹配配音会复用。"))
    with remembered_expander(tr("配音分段预览"), key=prefix + "_blocks_preview_open") as preview:
        if preview.open and rows:
            try:
                for index, block in enumerate(manual_audio_ranges(manuscript, options["tts_break_after"]), 1):
                    st.markdown(f"**{tr('配音块')} {index} · {block['han']} {tr('汉字')}**")
                    st.text(block["text"].strip())
            except ValueError as exc:
                st.error(str(exc))
    remember_widget_values(prefix)
    return options


def _audio_options(tr, manuscript, prefix="narration"):
    modes = (LONG_MODE, LEGACY_MODE)
    mode = st.radio(tr("配音与字幕方式"), modes,
                    index=modes.index(remembered_widget_value(prefix + "_audio_mode", LONG_MODE)),
                    format_func=lambda value: tr(audio_mode_label(value)), key=prefix + "_audio_mode")
    options = {"audio_mode": mode}
    if mode == LONG_MODE:
        options["tts_chunking"] = st.radio(tr("TTS 分段方式"), ("manual", "auto"),
            index=("manual", "auto").index(remembered_widget_value(prefix + "_tts_chunking", "manual")),
            format_func=lambda value: tr({"manual": "手动：按空行与句末选择", "auto": "自动：按目标字符组合"}[value]), key=prefix + "_tts_chunking")
        options["alignment_enabled"] = st.checkbox(tr("启用本地字幕对齐"), value=remembered_widget_value(prefix + "_alignment_enabled", True), key=prefix + "_alignment_enabled",
            help=tr("默认启用，以原文定位发声时间。关闭后保留字幕，按音频时长和文字权重估算，可能与实际发声有误差。"))
        st.caption(tr("单次配音最多12000汉字。"))
        if options["alignment_enabled"]:
            st.caption(tr("超过5分钟的配音会自动定位边界，再用原速短窗口对齐，已生成配音继续复用；全文仍须满足模型文本上下文限制。"))
        else:
            st.caption(tr("字幕时间使用文字估算，可能与配音存在时差；将跳过本地模型加载。"))
        if options["tts_chunking"] == "manual":
            options.update(_manual_blocks(tr, manuscript, prefix))
        else:
            options["tts_block_chars"] = st.number_input(tr("每块配音目标字符数"), 400, MAX_TTS_HAN,
                int(remembered_widget_value(prefix + "_tts_block_chars", 800)), 100, key=prefix + "_tts_block_chars")
        defaults = default_alignment_options()
        with remembered_expander(tr("本地对齐高级设置"), key=prefix + "_alignment_open") as panel:
            if not panel.open:
                return options
            render_model_choice(tr, prefix, "alignment")
            options.update(_alignment_values(prefix))
            st.caption(options["alignment_python"] + " · " + options["alignment_model"])
            languages = {"Chinese": "中文", "English": "英语", "Cantonese": "粤语", "French": "法语",
                         "German": "德语", "Italian": "意大利语", "Portuguese": "葡萄牙语", "Russian": "俄语", "Spanish": "西班牙语"}
            options["alignment_language"] = st.selectbox(tr("配音语言"), tuple(languages),
                index=tuple(languages).index(remembered_widget_value(prefix + "_alignment_language", "Chinese")),
                format_func=lambda value: tr(languages[value]), key=prefix + "_alignment_language", disabled=not options["alignment_enabled"])
            st.caption(tr("进入对齐阶段且存在未完成块时才加载模型；成功时间缓存可重复使用。"))
    else:
        st.caption(tr("每条字幕分别配音，按实际音频时长同步；适合继续已有任务。"))
    return options


def _clear_reference_upload(prefix):
    if not st.session_state.get(prefix + "_reference_upload"):
        key = prefix + "_reference_uploaded_path"
        st.session_state.pop(key, None)
        st.session_state.get("narration_widget_memory", {}).pop(key, None)


def _voice_profile(tr, prefix="narration"):
    render_model_choice(tr, prefix, "tts", engines={"mimo_tts", "edge_tts"})
    profile = voice_values(prefix)
    engine = profile["engine"]
    model_config_link(tr, prefix + "_tts")
    if engine not in {"mimo_tts", "edge_tts"}:
        st.warning(tr("文案项目支持MiMo或Edge配音，请选择对应配置。"))
        return profile
    if engine == "edge_tts":
        voices = ("zh-CN-XiaoxiaoNeural", "zh-CN-YunxiNeural", "zh-CN-XiaoyiNeural", "zh-CN-YunjianNeural")
        profile["voice"] = st.selectbox(tr("音色"), voices, index=voices.index(remembered_widget_value(prefix + "_edge_voice", voices[0])), key=prefix + "_edge_voice")
        profile["rate"] = st.slider(tr("语速"), .5, 2.0, float(remembered_widget_value(prefix + "_edge_rate", 1.0)), .05, key=prefix + "_edge_rate")
        return profile
    saved = config.mimo_tts
    st.caption(profile["model"] + " · " + profile["base_url"])
    profile["voice"] = remembered_widget_value(prefix + "_mimo_voice", saved.get("voice", "冰糖"))
    if profile["model"] == MODELS[0]:
        profile["voice"] = st.selectbox(tr("音色"), VOICES, index=VOICES.index(profile["voice"]) if profile["voice"] in VOICES else 0, key=prefix + "_mimo_voice")
    if profile["model"] == MODELS[2]:
        profile["reference_audio"] = st.text_input(tr("参考音频完整路径（MP3/WAV）"), value=remembered_widget_value(prefix + "_reference_path", saved.get("reference_audio", "")), key=prefix + "_reference_path").strip().strip('"')
        upload = st.file_uploader(tr("或上传参考音频"), type=["mp3", "wav"], key=prefix + "_reference_upload", on_change=_clear_reference_upload, args=(prefix,))
        if upload:
            data = upload.getvalue()
            if len(data) > 7.5 * 1024 * 1024:
                st.error(tr("MiMo Reference Audio Too Large"))
                profile["reference_audio"] = ""
            else:
                directory = Path(config.root_dir) / "storage" / "narration_references"
                directory.mkdir(parents=True, exist_ok=True)
                path = directory / (stable_hash(data.hex()) + Path(upload.name).suffix.lower())
                if not path.is_file():
                    path.write_bytes(data)
                profile["reference_audio"] = str(path)
                st.session_state[prefix + "_reference_uploaded_path"] = str(path)
        elif not profile["reference_audio"]:
            profile["reference_audio"] = remembered_widget_value(prefix + "_reference_uploaded_path", "")
        st.session_state[prefix + "_reference_selected_path"] = profile["reference_audio"]
        st.caption(tr("MiMo Reference Audio Help"))
    profile["instruction"] = st.text_area(tr("MiMo Voice Instructions"), value=remembered_widget_value(prefix + "_voice_instruction", saved.get("instruction", "")), help=tr("MiMo Voice Instructions Help"), key=prefix + "_voice_instruction")
    return profile


def validate_voice(profile, *, require_credentials=True):
    if profile.get("engine") not in {"mimo_tts", "edge_tts"}:
        raise ValueError("文案项目请选择MiMo或Edge配音配置。")
    if profile["engine"] == "mimo_tts":
        if require_credentials and not profile.get("api_key"):
            raise ValueError("请填写小米语音 API Key。")
        if profile["model"] == MODELS[1] and not profile.get("instruction", "").strip():
            raise ValueError("请填写音色设计描述。")
        if profile["model"] == MODELS[2] and not Path(profile.get("reference_audio") or "__missing__").is_file():
            raise ValueError("请选择有效参考音频。")


def voice_values(prefix="narration"):
    """Business voice options overlay a centrally selected service snapshot."""
    from app.services.model_profiles import tts_profile, legacy_catalog
    from webui.components.model_settings import selected_bindings
    value = lambda name, default=None: default if prefix is None else remembered_widget_value(prefix + "_" + name, default)
    identifier = selected_bindings(prefix)["tts"] if prefix else None
    try:
        service = tts_profile(identifier, historical=True)
        engine, saved = service["engine"], service["settings"]
    except ValueError:
        recorded = value("saved_voice", {})
        legacy = next(row for row in legacy_catalog()["profiles"] if row["id"] == "legacy_tts_mimo_tts")
        engine, saved = recorded.get("engine", "mimo_tts"), {**legacy["settings"], **recorded, "api_key": ""}
    profile = {"engine": engine, "proxy": dict(config.proxy)}
    if engine not in {"mimo_tts", "edge_tts"}:
        return profile
    if engine == "edge_tts":
        return {**profile, "voice": value("edge_voice", "zh-CN-XiaoxiaoNeural"), "rate": float(value("edge_rate", 1.0))}
    profile.update(api_key=saved.get("api_key", ""), base_url=saved.get("base_url") or DEFAULT_BASE_URL,
        model=saved.get("model") or MODELS[0], voice=value("mimo_voice", saved.get("voice", "冰糖")),
        instruction=value("voice_instruction", saved.get("instruction", "")))
    if profile["model"] == MODELS[0] and profile["voice"] not in VOICES:
        profile["voice"] = VOICES[0]
    if profile["model"] == MODELS[2]:
        profile["reference_audio"] = value("reference_selected_path", None)
        if profile["reference_audio"] is None:
            profile["reference_audio"] = value("reference_path", saved.get("reference_audio", "")).strip().strip('"') or value("reference_uploaded_path", "")
    return profile


def _alignment_values(prefix):
    try:
        return alignment_profile(selected_bindings(prefix).get("alignment"), historical=True)
    except ValueError:
        saved = remembered_widget_value(prefix + "_saved_audio_options", {})
        return {name: saved.get(name, value) for name, value in default_alignment_options().items() if name != "alignment_language"}


def audio_values(manuscript, prefix="narration"):
    """Keep hidden audio settings available to create/resume/fork actions."""
    value = lambda name, default=None: remembered_widget_value(prefix + "_" + name, default)
    options = {"audio_mode": value("audio_mode", LONG_MODE), "tts_budget": int(value("tts_budget_v2", runner.DEFAULT_CALL_BUDGET))}
    if options["audio_mode"] == LONG_MODE:
        options.update(tts_chunking=value("tts_chunking", "manual"), alignment_enabled=bool(value("alignment_enabled", True)))
        options.update(_alignment_values(prefix))
        options["alignment_language"] = value("alignment_language", "Chinese")
        for name in ("alignment_python", "alignment_model"):
            options[name] = options[name].strip().strip('"')
        if options["tts_chunking"] == "manual":
            options.update(_manual_state(manuscript, prefix)[0])
            previous_hash = value("tts_break_text_hash")
            if previous_hash and previous_hash != options["tts_break_text_hash"]:
                options.update(tts_break_text_hash=previous_hash, tts_break_after=value("tts_break_after", []))
        else:
            options["tts_block_chars"] = int(value("tts_block_chars", 800))
    return options


@st.fragment
def render_narration_audio_panel(tr, manuscript="", *, key_prefix="narration"):
    """Render only the open TTS panel; actions read live session values."""
    remember_widget_values(key_prefix)
    with remembered_expander(tr("TTS 配音设置"), key=key_prefix + "_tts_open") as panel:
        if panel.open:
            _voice_profile(tr, key_prefix)
            _audio_options(tr, manuscript, key_prefix)
            st.number_input(tr("本次最多配音调用次数"), 1, 10000,
                int(remembered_widget_value(key_prefix + "_tts_budget_v2", runner.DEFAULT_CALL_BUDGET)),
                10, key=key_prefix + "_tts_budget_v2")
            st.caption(tr("达到配音调用预算会暂停；继续生成时复用已经完成的阶段。"))
    remember_widget_values(key_prefix)
    return voice_values(key_prefix), audio_values(manuscript, key_prefix)
