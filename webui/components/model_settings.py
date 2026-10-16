"""Central model editor and shared capability-filtered selectors."""

import json
from copy import deepcopy
from hashlib import sha256
from uuid import uuid4

import streamlit as st

from app.config import config
from app.services import model_profiles as profiles
from webui.utils.file_utils import remembered_widget_value, remembered_expander, remember_widget_values, folder_input

_KIND_LABELS = {"chat": "理解模型", "tts": "配音服务", "alignment": "本地对齐", "asr": "语音识别服务"}
_CAP_LABELS = {"text": "文本", "vision": "图片", "audio": "音频理解", "tts": "配音", "alignment": "对齐", "asr": "语音识别"}


def _profile_label(summary, tr):
    label = summary["name"] + " · " + (summary["model"] or tr("未填写模型")) + " · " + tr(summary["field_status"])
    return label if summary["enabled"] else label + " · " + tr("已停用")


def _parameter_summary(summary, tr):
    params = summary["parameters"]
    if not params:
        return ""
    return (f"{tr('思考等级')} {params.get('thinking_level', 'auto')} · "
            f"{tr('输出额度')} {params.get('max_tokens', 0)} · "
            f"{tr('上下文预算')} {params.get('context_tokens', 64000)} · "
            f"T={params.get('temperature', 1)} · top_p={params.get('top_p', .95)}")


def seed_model_choices(prefix, *, models=None, voice=None, options=None, bindings=None):
    memory = st.session_state.setdefault("narration_widget_memory", {})
    defaults = profiles.default_bindings()
    recorded = (models or {}).get("_bindings", {})
    for role in profiles.ROLE_CAPABILITIES:
        if bindings is not None and role in bindings:
            value = bindings[role]
        elif role in recorded:
            value = recorded[role]
        else:
            candidates = profiles.catalog()["profiles"]
            saved = (models or {}).get(role, {})
            if role == "tts" and voice:
                engine = voice.get("engine", "mimo_tts")
                match = next((row["id"] for row in candidates if row["kind"] == "tts" and row["engine"] == engine
                              and row["model"] == voice.get("model", engine)
                              and row.get("base", "").rstrip("/") == voice.get("base_url", "").rstrip("/")), None)
            elif role == "alignment" and options:
                match = next((row["id"] for row in candidates if row["kind"] == "alignment"
                              and all(row.get("settings", {}).get(name) == options.get(name)
                                      for name in ("alignment_python", "alignment_model"))), None)
            elif saved:
                match = next((row["id"] for row in candidates if row["kind"] == "chat" and row["model"] == saved.get("model")
                              and row.get("base", "").rstrip("/") == saved.get("base", "").rstrip("/")), None)
                if role.endswith("_backup") and not saved.get("enabled", False):
                    match = ""
            else:
                match = None
            value = match if match is not None else defaults.get(role, "")
        memory.setdefault(prefix + "_profile_" + role, value)
    if voice:
        memory.setdefault(prefix + "_saved_voice", deepcopy(voice))
    if options:
        memory.setdefault(prefix + "_saved_audio_options", deepcopy(options))


def selected_bindings(prefix):
    defaults = profiles.default_bindings()
    return {role: remembered_widget_value(prefix + "_profile_" + role, defaults.get(role, ""))
            for role in profiles.ROLE_CAPABILITIES}


def selected_models(prefix, roles=None):
    return profiles.resolve_models(selected_bindings(prefix), roles=roles, historical=True, allow_missing=True)


def check_model_choices(prefix, roles):
    bindings = selected_bindings(prefix)
    for role in roles:
        if role.endswith("_backup") and not bindings.get(role):
            continue
        profiles.get_profile(bindings.get(role, ""), role)


def render_model_choice(tr, prefix, role, *, engines=None, show_pending_default=False):
    key = prefix + "_profile_" + role
    current = selected_bindings(prefix).get(role, "")
    summaries = {row["id"]: profiles.profile_summary(row) for row in profiles.catalog()["profiles"]}
    compatible = profiles.available(role, engines=engines)
    pending = [row for row in compatible if not summaries[row["id"]]["complete"]]
    show_pending = False
    if pending:
        show_key = key + "_show_pending"
        if show_key not in st.session_state:
            st.session_state[show_key] = bool(remembered_widget_value(show_key, show_pending_default))
        show_pending = st.checkbox(tr("显示此用途的待完善配置"), key=show_key,
            help=tr("仅显示字段状态，不代表服务可用；选择后仍需完善必要字段。"))
    identifiers = [row["id"] for row in compatible if show_pending or summaries[row["id"]]["complete"]]
    labels = {identifier: _profile_label(summaries[identifier], tr) for identifier in identifiers}
    optional = role.endswith("_backup")
    choices = ["", *identifiers]
    if current and current not in identifiers:
        choices.insert(1, current)
        labels[current] = (_profile_label(summaries[current], tr) if current in summaries else
                           tr("当前配置不可用，请重新选择") + " · " + current)
    if key not in st.session_state:
        st.session_state[key] = current if current in choices else ""
    selected = st.selectbox(tr(profiles.ROLE_LABELS[role]), choices, index=None,
        format_func=lambda item: labels.get(item, tr("关闭后备" if optional else "请选择模型配置")),
        key=key)
    if selected in summaries:
        summary = summaries[selected]
        st.caption(" · ".join(part for part in (summary["location"], _parameter_summary(summary, tr), tr("未验证")) if part))
        if summary["missing"]:
            st.warning(tr("所选配置待完善：") + "、".join(tr(reason) for reason in summary["missing"]))
    if selected and selected not in {row["id"] for row in compatible}:
        st.warning(tr("所选配置已停用、被移除或不支持此流程；历史项目普通继续仍使用保存快照。"))
    remember_widget_values(prefix)
    return selected


def render_model_choices(tr, prefix, *, roles=("text", "text_fast", "vision"), backups=True):
    for role in roles:
        render_model_choice(tr, prefix, role)
    if backups:
        with remembered_expander(tr("后备模型选择"), key=prefix + "_model_backups_open") as panel:
            if panel.open:
                for role in ("text_backup", "vision_backup"):
                    render_model_choice(tr, prefix, role)
    model_config_link(tr, prefix)


def model_config_link(tr, prefix):
    if st.button(tr("前往模型配置"), key=prefix + "_go_models"):
        remember_widget_values(prefix)
        st.session_state["model_config_return_page"] = st.session_state.get("workspace_page", "narration_progress")
        st.session_state["narration_pending_workspace"] = "models"
        st.rerun(scope="app")


def _render_profile_table(tr, prefix, summaries, defaults, group):
    data = []
    for summary in summaries:
        uses = [tr(profiles.ROLE_LABELS[role]) for role, identifier in defaults.items()
                if identifier == summary["id"] and role in profiles.ROLE_LABELS]
        data.append({tr("配置名称"): summary["name"], tr("实际模型名称"): summary["model"],
            tr("服务类型"): tr(_KIND_LABELS.get(summary["kind"], "未知类型")),
            tr("输入能力"): "、".join(tr(_CAP_LABELS.get(cap, cap)) for cap in summary["capabilities"]),
            tr("默认用途"): "、".join(uses), tr("接口／本地位置"): summary["location"],
            tr("参数摘要"): _parameter_summary(summary, tr),
            tr("字段状态"): "、".join(tr(reason) for reason in summary["missing"]) or tr(summary["field_status"]),
            tr("启用状态"): tr("允许新任务选择" if summary["enabled"] else "已停用"),
            tr("服务验证"): tr(summary["verification"])})
    # A changed visible table gets fresh row selection state. The selected ID is independent.
    signature = sha256(json.dumps([[row["id"] for row in summaries], data], ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:16]
    table_key = prefix + "_table_" + group + "_" + signature

    def select_row():
        selected_rows = st.session_state.get(table_key, {}).get("selection", {}).get("rows", [])
        if selected_rows and 0 <= selected_rows[0] < len(summaries):
            st.session_state[prefix + "_pending_selection"] = summaries[selected_rows[0]]["id"]
            st.session_state[prefix + "_editor_open"] = True
        remember_widget_values(prefix)

    st.dataframe(data, hide_index=True, width="stretch", height=min(420, 38 + len(data) * 35),
        key=table_key, on_select=select_row, selection_mode="single-row")


def _render_profile_list(tr, prefix, registry):
    summaries = [profiles.profile_summary(row) for row in registry["profiles"]]
    complete_count = sum(row["complete"] for row in summaries)
    st.caption(tr("共{count}项配置，{complete}项字段已填写，{pending}项待完善。").format(
        count=len(summaries), complete=complete_count, pending=len(summaries) - complete_count))
    st.caption(" · ".join(f"{tr(label)} {sum(row['kind'] == kind for row in summaries)}" for kind, label in _KIND_LABELS.items()))
    choices = {"kind": ["", *_KIND_LABELS], "cap": ["", *_CAP_LABELS],
               "enabled": ["", "enabled", "disabled"], "fields": ["", "complete", "pending"]}
    search_key, all_key = prefix + "_search", prefix + "_show_all"
    if search_key not in st.session_state:
        st.session_state[search_key] = remembered_widget_value(search_key, "")
    if all_key not in st.session_state:
        st.session_state[all_key] = bool(remembered_widget_value(all_key, False))
    for name, values in choices.items():
        key = prefix + "_filter_" + name
        value = remembered_widget_value(key, "")
        if key not in st.session_state or value not in values:
            st.session_state[key] = value if value in values else ""
    search = (st.text_input(tr("搜索配置名称或实际模型"), value=None, key=search_key) or "").strip().casefold()
    kind_col, cap_col, enabled_col, fields_col = st.columns(4)
    kind = kind_col.selectbox(tr("服务类型筛选"), choices["kind"], index=None,
        format_func=lambda item: tr(_KIND_LABELS.get(item, "全部类型")), key=prefix + "_filter_kind")
    cap = cap_col.selectbox(tr("输入能力筛选"), choices["cap"], index=None,
        format_func=lambda item: tr(_CAP_LABELS.get(item, "全部能力")), key=prefix + "_filter_cap")
    enabled = enabled_col.selectbox(tr("启用状态筛选"), choices["enabled"], index=None,
        format_func=lambda item: tr({"enabled": "已启用", "disabled": "已停用"}.get(item, "全部启用状态")), key=prefix + "_filter_enabled")
    fields = fields_col.selectbox(tr("字段状态筛选"), choices["fields"], index=None,
        format_func=lambda item: tr({"complete": "字段已填写", "pending": "待完善"}.get(item, "全部字段状态")), key=prefix + "_filter_fields")
    show_all = st.checkbox(tr("显示全部配置"), key=all_key)
    filtered = [row for row in summaries if (not search or search in (row["name"] + " " + row["model"]).casefold())
                and (not kind or row["kind"] == kind) and (not cap or cap in row["capabilities"])
                and (not enabled or row["enabled"] == (enabled == "enabled"))
                and (not fields or row["complete"] == (fields == "complete"))]
    show_pending = show_all or fields == "pending"
    main = [row for row in filtered if show_pending or row["complete"]]
    pending = [row for row in filtered if not row["complete"]]
    st.caption(tr("点选一行查看或编辑；表格排序不会改变配置ID。"))
    if main:
        _render_profile_table(tr, prefix, main, registry.get("defaults", {}), "main")
    else:
        st.info(tr("当前列表没有匹配项；可调整筛选、新建或查看待完善配置。"))
    if pending and not show_pending:
        with remembered_expander(tr("待完善配置") + " · " + str(len(pending)), key=prefix + "_pending_open") as panel:
            if panel.open:
                _render_profile_table(tr, prefix, pending, registry.get("defaults", {}), "pending")
    return {row["id"] for row in filtered}


def render_model_workspace(tr):
    st.header(tr("模型配置"))
    prefix = "model_registry"
    st.session_state["_narration_settings_prefix"] = prefix
    remember_widget_values(prefix)
    registry = profiles.catalog()
    st.caption(tr("一份配置包含实际模型、接口、凭据、参数和用途；同一模型可以保存多份调用配置。"))
    st.caption(tr("字段已填写不代表服务可用。列表不测试接口或部署，服务验证统一显示“未验证”。输入能力是配置声明。"))
    if config.model_profiles.get("schema") != 1:
        st.info(tr("当前显示旧配置的只读兼容项。导入或保存后写入集中配置，旧字段保留。"))
    if st.button(tr("导入现有模型配置"), key=prefix + "_import"):
        try:
            profiles.import_legacy()
            st.success(tr("现有配置已导入，用户已保存的配置保持。"))
            st.rerun()
        except (ValueError, OSError) as exc:
            st.error(str(exc))
    return_page = st.session_state.get("model_config_return_page")
    if return_page and st.button(tr("返回原工作区"), key=prefix + "_return"):
        st.session_state["narration_pending_workspace"] = return_page
        st.rerun()
    rows = registry["profiles"]
    ids = [row["id"] for row in rows]
    selection_key = prefix + "_selected"
    pending = st.session_state.pop(prefix + "_pending_selection", None)
    if pending is not None:
        st.session_state[selection_key] = pending if pending in ids else ""
        st.session_state[prefix + "_creating"] = False
    elif selection_key in st.session_state and st.session_state[selection_key] not in ["", *ids]:
        st.session_state[selection_key] = ""
    if selection_key not in st.session_state:
        previous = remembered_widget_value(selection_key, "")
        st.session_state[selection_key] = previous if previous in ids else ""
    visible_ids = _render_profile_list(tr, prefix, registry)
    if st.button(tr("新建模型配置"), key=prefix + "_new"):
        st.session_state[prefix + "_creating"] = True
        st.session_state.setdefault(prefix + "_new_id", "profile-" + uuid4().hex)
        st.session_state[prefix + "_editor_open"] = True
        remember_widget_values(prefix)
        st.rerun()
    selected = st.session_state[selection_key]
    creating = st.session_state.get(prefix + "_creating", False)
    if creating or selected:
        if selected and not creating and selected not in visible_ids:
            st.caption(tr("当前筛选隐藏所选项；下方仍编辑原配置。"))
        with remembered_expander(tr("配置详情与编辑"), key=prefix + "_editor_open", default=True) as panel:
            if panel.open:
                if st.button(tr("返回列表"), key=prefix + "_close_editor"):
                    st.session_state[prefix + "_pending_selection"] = ""
                    remember_widget_values(prefix)
                    st.rerun()
                _render_model_editor(tr, prefix, registry, selected, creating)
    else:
        st.info(tr("点选已有配置查看或编辑，或点击“新建模型配置”。"))
    with remembered_expander(tr("新项目和工作区的默认模型"), key=prefix + "_defaults_open") as panel:
        if panel.open:
            defaults_prefix = prefix + "_defaults"
            seed_model_choices(defaults_prefix, bindings=registry.get("defaults", {}))
            values = {role: render_model_choice(tr, defaults_prefix, role) for role in profiles.ROLE_CAPABILITIES}
            if st.button(tr("保存默认选择"), key=prefix + "_save_defaults"):
                try:
                    profiles.save_catalog({**registry, "defaults": values})
                    st.rerun()
                except (ValueError, OSError) as exc:
                    st.error(str(exc))
    remember_widget_values(prefix)


def _render_model_editor(tr, prefix, registry, selected, creating):
    rows = registry["profiles"]
    row = deepcopy({"id": st.session_state[prefix + "_new_id"], "name": "", "kind": "chat", "engine": "openai",
                    "capabilities": ["text"], "enabled": True, "model": "", "base": "", "key": "", "parameters": {}, "settings": {}}
                   if creating else next(row for row in rows if row["id"] == selected))
    stem = prefix + "_" + row["id"]
    drafts = st.session_state.setdefault(prefix + "_drafts", {})
    row = deepcopy(drafts.get(row["id"], row))

    def widget_value(suffix, default):
        return remembered_widget_value(stem + "_" + suffix, default)

    def choice_index(suffix, values, default):
        value = widget_value(suffix, default)
        return values.index(value) if value in values else 0

    st.subheader(tr("新建模型配置" if creating else "编辑模型配置"))
    st.caption("ID: " + row["id"])
    name = st.text_input(tr("配置名称"), value=widget_value("name", row["name"]), key=stem + "_name")
    kinds = ("chat", "tts", "alignment", "asr")
    kind = st.selectbox(tr("服务类型"), kinds, index=choice_index("kind", kinds, row["kind"]),
        format_func=lambda item: tr(_KIND_LABELS[item]), key=stem + "_kind")
    enabled = st.checkbox(tr("允许新任务选择"), value=widget_value("enabled", row.get("enabled", True)), key=stem + "_enabled")
    engine = "openai" if kind == "chat" else "qwen_hf"
    if kind == "tts":
        engines = list(profiles.TTS_GROUPS)
        engine = st.selectbox(tr("配音引擎"), engines, index=choice_index("engine", engines, row.get("engine")), key=stem + "_engine")
    if kind == "asr":
        engines = ("local", "firered", "bailian")
        engine = st.selectbox(tr("识别服务类型"), engines, index=choice_index("asr_engine", engines, row.get("engine")),
            format_func=lambda item: {"local": "FunASR-Pack", "firered": "FireRedASR", "bailian": "百炼Fun-ASR"}[item], key=stem + "_asr_engine")
        row["model"] = {"local": "sensevoice", "firered": "FireRedASR", "bailian": "fun-asr"}[engine]
    if kind == "tts" and engine == "mimo_tts":
        from app.services.mimo_tts import MODELS
        model = st.selectbox(tr("实际模型名称"), MODELS,
            index=choice_index("model_mimo", MODELS, row.get("model")), key=stem + "_model_mimo")
    else:
        model = st.text_input(tr("实际模型名称"), value=widget_value("model_" + kind + "_" + engine, row.get("model", "")), disabled=kind == "asr", key=stem + "_model_" + kind + "_" + engine)
    draft = {**row, "name": name, "kind": kind, "engine": engine, "model": model, "enabled": enabled}
    if kind == "chat":
        draft["capabilities"] = st.multiselect(tr("输入能力"), ["text", "vision", "audio"],
            default=[cap for cap in widget_value("caps", row.get("capabilities", [])) if cap in {"text", "vision", "audio"}],
            format_func=lambda cap: tr(_CAP_LABELS[cap]), key=stem + "_caps")
    else:
        draft["capabilities"] = [kind]
    if kind != "alignment":
        if kind == "asr" and engine == "bailian":
            row["base"] = "https://dashscope.aliyuncs.com"
        draft["base"] = st.text_input(tr("接口地址"), value=widget_value("base_" + kind + "_" + engine, row.get("base", "")), disabled=kind == "asr" and engine == "bailian", key=stem + "_base_" + kind + "_" + engine)
        draft["key"] = st.text_input(tr("API Key"), value=widget_value("key", row.get("key", "")), type="password", key=stem + "_key")
    settings = deepcopy(row.get("settings", {}))
    if kind == "chat":
        params = {**profiles.DEFAULT_LLM_GENERATION_CONFIG, "context_tokens": 64000, **row.get("parameters", {})}
        levels = ("auto", "off", "low", "medium", "high")
        thinking = st.selectbox(tr("思考等级"), levels, index=choice_index("thinking", levels, params["thinking_level"]), key=stem + "_thinking")
        temperature = st.slider(tr("采样温度"), 0.0, 2.0, float(widget_value("temperature", params["temperature"])), .05, key=stem + "_temperature")
        top_p = st.slider("top_p", 0.0, 1.0, float(widget_value("top_p", params["top_p"])), .05, key=stem + "_top_p")
        tokens = st.number_input(tr("输出额度（tokens，0为自动）"), min_value=0, value=int(widget_value("tokens", params["max_tokens"])), step=256, key=stem + "_tokens")
        context = st.number_input(tr("上下文预算（tokens）"), min_value=8192, value=int(widget_value("context", params["context_tokens"])), step=1024, key=stem + "_context")
        draft["parameters"] = {"thinking_level": thinking, "temperature": temperature, "top_p": top_p,
                               "max_tokens": tokens, "context_tokens": context}
    elif kind == "alignment":
        defaults = profiles.default_alignment_options()
        settings["alignment_python"] = st.text_input(tr("对齐环境Python路径"),
            value=widget_value("python", settings.get("alignment_python", defaults["alignment_python"])), key=stem + "_python")
        settings["alignment_model"] = folder_input(tr("对齐模型目录"),
            value=widget_value("model_folder", settings.get("alignment_model", defaults["alignment_model"])), key=stem + "_model_folder", tr=tr)
        draft["settings"] = settings
    else:
        secrets = {"azure_speech": ("speech_key",), "tencent_tts": ("secret_id", "secret_key"),
                   "doubaotts": ("ak", "sk", "token"), "soulvoice": ("api_key",)}
        for field in secrets.get(engine, ()):
            settings[field] = st.text_input(tr("服务凭据") + " · " + field, value=str(widget_value("secret_" + field, settings.get(field, ""))),
                type="password", key=stem + "_secret_" + field)
        with st.expander(tr("其他服务参数（JSON）"), expanded=False):
            public = profiles.public_values(settings)
            for field in ("model", "model_name", "base_url", "api_url", "api_key"):
                public.pop(field, None)
            raw = st.text_area(tr("服务参数"), value=remembered_widget_value(stem + "_settings_" + engine, json.dumps(public, ensure_ascii=False, indent=2)),
                height=180, key=stem + "_settings_" + engine)
        try:
            edited = json.loads(raw)
            if not isinstance(edited, dict):
                raise ValueError("服务参数应为JSON对象。")
            draft["settings"] = profiles.with_private_values(edited, settings)
        except (ValueError, TypeError) as exc:
            st.error(str(exc))
            return
    drafts[row["id"]] = deepcopy(draft)
    save_col, copy_col, test_col = st.columns(3)
    if save_col.button(tr("保存配置"), type="primary", key=stem + "_save"):
        try:
            valid = profiles.validate_profile(draft)
            updated = {**registry, "profiles": [item for item in rows if item["id"] != row["id"]] + [valid]}
            profiles.save_catalog(updated)
            drafts.pop(row["id"], None)
            st.session_state[prefix + "_pending_selection"] = valid["id"]
            if creating:
                st.session_state.pop(prefix + "_new_id", None)
            st.rerun()
        except (ValueError, OSError, TypeError) as exc:
            st.error(str(exc))
    if copy_col.button(tr("保存为副本"), key=stem + "_copy"):
        try:
            draft.update(id="profile-" + uuid4().hex, name=name.strip()[:95] + "（副本）")
            profiles.save_catalog({**registry, "profiles": rows + [profiles.validate_profile(draft)]})
            st.session_state[prefix + "_pending_selection"] = draft["id"]
            st.rerun()
        except (ValueError, OSError, TypeError) as exc:
            st.error(str(exc))
    if test_col.button(tr("检查或测试当前草稿"), key=stem + "_test"):
        try:
            valid = profiles.validate_profile(draft)
            if kind == "alignment":
                from app.services.narration_alignment import validate_alignment_options
                validate_alignment_options(settings)
                st.success(tr("环境和模型文件检查通过；未加载模型。"))
            elif kind == "chat" and "text" in valid["capabilities"]:
                from app.services.library.gateway import ModelGateway
                profile = {**valid.get("parameters", {}), "model": valid["model"], "base": valid["base"], "key": valid["key"]}
                gateway = ModelGateway({"text": profile, "proxy": deepcopy(config.proxy)})
                gateway.json('Return JSON: {"ok": true}', model_kind="text", max_output=128, attempts=1)
                st.success(tr("文本连接测试通过。"))
            else:
                st.info(tr("配置格式检查通过；实际音频、图片或配音服务请在对应工作区显式执行。"))
        except (ValueError, OSError, RuntimeError) as exc:
            st.error(str(exc))
    if selected and not creating:
        with remembered_expander(tr("停用或删除配置"), key=stem + "_delete_open") as panel:
            if panel.open:
                st.caption(tr("取消“允许新任务选择”并保存即可停用。删除前检查保存引用，已有任务快照保留。"))
                if st.button(tr("检查配置引用"), key=stem + "_references"):
                    try:
                        st.session_state[stem + "_reference_info"] = profiles.profile_references(selected)
                    except (OSError, ValueError) as exc:
                        st.error(str(exc))
                refs = st.session_state.get(stem + "_reference_info")
                if refs is not None:
                    st.write(refs or tr("没有保存引用。"))
                    confirmed = st.checkbox(tr("确认删除此模型配置"), value=False, key=stem + "_delete_confirm")
                    if st.button(tr("删除未引用配置"), disabled=bool(refs) or not confirmed, key=stem + "_delete"):
                        try:
                            profiles.delete_profile(selected)
                            drafts.pop(selected, None)
                            st.session_state[prefix + "_pending_selection"] = ""
                            st.rerun()
                        except (OSError, ValueError) as exc:
                            st.error(str(exc))
