"""Named model definitions, workspace bindings and immutable request snapshots."""

from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from math import isfinite
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit
from uuid import uuid4
import re
import threading

from app.config import config
from app.config.defaults import DEFAULT_LLM_GENERATION_CONFIG
from app.services.llm.generation_options import generation_settings
from app.services.narration_alignment import default_alignment_options

ROLE_CAPABILITIES = {"text": "text", "text_fast": "text", "vision": "vision", "audio": "audio",
                     "text_backup": "text", "vision_backup": "vision", "tts": "tts", "alignment": "alignment", "asr": "asr"}
ROLE_LABELS = {"text": "文案分析", "text_fast": "字幕分段", "vision": "视觉分析", "audio": "音频理解",
               "text_backup": "后备文案", "vision_backup": "后备视觉", "tts": "配音服务", "alignment": "本地对齐", "asr": "语音识别服务"}
TTS_GROUPS = {"mimo_tts": "mimo_tts", "edge_tts": "", "azure_speech": "azure", "tencent_tts": "tencent",
              "qwen3_tts": "tts_qwen", "soulvoice": "soulvoice", "doubaotts": "doubaotts", "indextts": "indextts",
              "indextts_macos": "indextts_macos", "indextts2": "indextts2", "omnivoice": "omnivoice",
              "voxcpm_05b": "voxcpm_05b", "voxcpm_2b": "voxcpm_2b"}
_PRIVATE = {"key", "api_key", "apikey", "secret_key", "secret_id", "token", "access_token", "speech_key", "ak", "sk",
            "password", "authorization", "headers", "cookies", "subscription_key", "client_secret", "x_api_key"}
_PRIVATE_COMPACT = {name.replace("_", "") for name in _PRIVATE}
_ID = re.compile(r"[a-zA-Z0-9_-]{1,100}\Z")
_lock = threading.RLock()
_runtime = ContextVar("narrato_model_runtime", default=None)


def _private_key(key):
    name = str(key).lower().replace("-", "_")
    return name.replace("_", "") in _PRIVATE_COMPACT or name.endswith(("_api_key", "_secret", "_token", "_password"))


def public_values(value):
    if isinstance(value, dict):
        return {k: public_values(v) for k, v in value.items() if not _private_key(k) and k != "proxy"}
    if isinstance(value, list):
        return [public_values(v) for v in value]
    return deepcopy(value)


def with_private_values(value, active):
    """Overlay credentials recursively without replacing saved public parameters."""
    if isinstance(active, list):
        if public_values(value) != public_values(active):
            raise ValueError("含凭据的参数列表结构已改变，请保留其公开结构或新建配置。")
        return [with_private_values(old, new) if isinstance(new, (dict, list)) else deepcopy(old) for old, new in zip(value, active)]
    result = deepcopy(value)
    for key, item in active.items():
        if _private_key(key):
            result[key] = deepcopy(item)
        elif isinstance(item, dict):
            nested = with_private_values(result.get(key, {}) if isinstance(result.get(key), dict) else {}, item)
            if nested:
                result[key] = nested
        elif isinstance(item, list) and public_values(item) != item:
            result[key] = with_private_values(result.get(key, []), item)
    return result


def legacy_catalog():
    """A read-only view until the user explicitly imports the original settings."""
    app = config.app
    profiles, defaults = [], {}
    for role in ("text", "text_fast", "vision", "text_backup", "vision_backup"):
        prefix = "text" if role == "text_fast" else role
        model = app.get("text_openai_fast_model_name") or app.get("text_openai_model_name", "") if role == "text_fast" else app.get(prefix + "_openai_model_name", "")
        row = {"id": "legacy_" + role, "name": ROLE_LABELS[role] + "（现有配置）", "kind": "chat", "engine": "openai",
               "model": model, "base": app.get(prefix + "_openai_base_url", ""), "key": app.get(prefix + "_openai_api_key", ""),
               "capabilities": [ROLE_CAPABILITIES[role]], "enabled": True, "parameters": generation_settings(app, prefix)}
        row["parameters"]["context_tokens"] = int(app.get("material_context_tokens", 64000))
        profiles.append(row)
        defaults[role] = row["id"] if not role.endswith("_backup") or app.get(role + "_openai_enabled", False) else ""
    supported = {"mimo-v2.6-flash", "mimo-v2.6-pro", "mimo-v2.6-pro-ultraspeed", "mimo-v2.5"}
    inherited = next((row for role in ("vision", "text_fast", "text") for row in profiles
                      if row["id"] == defaults[role] and str(row["model"]).lower().removeprefix("openai/") in supported), {})
    audio = {"id": "legacy_audio", "name": "音频理解（现有配置）", "kind": "chat", "engine": "openai", "enabled": True,
             "model": app.get("audio_openai_model_name", inherited.get("model", "mimo-v2.6-flash")),
             "base": app.get("audio_openai_base_url", inherited.get("base", "")),
             "key": app.get("audio_openai_api_key", inherited.get("key", "")), "capabilities": ["audio"],
             "parameters": {"thinking_level": app.get("audio_openai_thinking_level", "off"), "max_tokens": 8192,
                            "temperature": .2, "top_p": .95, "context_tokens": int(app.get("audio_context_tokens", 64000))}}
    profiles.append(audio)
    defaults["audio"] = audio["id"]
    for engine, group in TTS_GROUPS.items():
        settings = deepcopy(getattr(config, group, {})) if group else {}
        if engine == "mimo_tts":
            from app.services.mimo_tts import DEFAULT_BASE_URL, MODELS
            settings.setdefault("base_url", DEFAULT_BASE_URL)
            settings.setdefault("model", MODELS[0])
            if not settings.get("api_key") and "xiaomimimo.com" in app.get("text_openai_base_url", ""):
                settings["api_key"] = app.get("text_openai_api_key", "")
        model = settings.get("model", settings.get("model_name", engine))
        row = {"id": "legacy_tts_" + engine, "name": str(model) + "（现有配音）", "kind": "tts", "engine": engine,
               "model": model, "base": settings.get("base_url", settings.get("api_url", "")),
               "key": settings.get("api_key", ""), "settings": settings, "capabilities": ["tts"], "enabled": True}
        profiles.append(row)
        if engine == "mimo_tts":
            for mode in MODELS:
                if mode == model:
                    continue
                profiles.append({**deepcopy(row), "id": row["id"] + "_" + mode.rsplit("-", 1)[-1],
                                 "name": mode + "（现有配音）", "model": mode, "settings": {**settings, "model": mode}})
    defaults["tts"] = "legacy_tts_mimo_tts"
    alignment = default_alignment_options()
    profiles.append({"id": "legacy_alignment", "name": "Qwen3 ForcedAligner（本地）", "kind": "alignment", "engine": "qwen_hf",
                     "model": "Qwen3-ForcedAligner-0.6B-hf", "capabilities": ["alignment"], "enabled": True,
                     "settings": {name: value for name, value in alignment.items() if name != "alignment_language"}})
    defaults["alignment"] = "legacy_alignment"
    asr = deepcopy(config.fun_asr)
    for backend, model, base in (("local", "sensevoice", asr.get("api_url") or "http://127.0.0.1:7860"),
                                 ("firered", "FireRedASR", asr.get("firered_api_url") or "http://127.0.0.1:7867"),
                                 ("bailian", "fun-asr", "https://dashscope.aliyuncs.com")):
        profiles.append({"id": "legacy_asr_" + backend, "name": model + "（现有语音识别）", "kind": "asr", "engine": backend,
                         "model": model, "base": base, "key": asr.get("api_key", "") if backend == "bailian" else "",
                         "settings": {**asr, "backend": backend}, "capabilities": ["asr"], "enabled": True})
    backend = asr.get("backend")
    if backend not in {"local", "firered", "bailian"}:
        backend = "bailian" if asr.get("api_key") and not asr.get("api_url") else "local"
    defaults["asr"] = "legacy_asr_" + backend
    return {"schema": 1, "profiles": profiles, "defaults": defaults}


def catalog():
    with _lock:
        stored = deepcopy(config.model_profiles)
    return stored if stored.get("schema") == 1 else legacy_catalog()


def profile_summary(row):
    """Public, field-only display data; no service calls, path scans or mutations."""
    kind, engine = row.get("kind"), row.get("engine")
    settings = row.get("settings", {})
    base, key = str(row.get("base", "") or "").strip(), str(row.get("key", "") or "").strip()
    missing = []

    def require(value, message):
        if not str(value or "").strip():
            missing.append(message)

    require(row.get("name"), "缺配置名称")
    if kind in {"chat", "asr", "alignment"} or engine in {"mimo_tts", "qwen3_tts", "soulvoice"}:
        require(row.get("model"), "缺实际模型名称")
    location = base
    if kind == "alignment":
        require(settings.get("alignment_python"), "缺对齐环境Python路径")
        require(settings.get("alignment_model"), "缺对齐模型目录")
        location = str(settings.get("alignment_model", "") or "")
    elif kind == "chat":
        require(base, "缺接口地址")
        require(key, "缺API Key")
        if not row.get("capabilities"):
            missing.append("缺输入能力")
    elif kind == "asr":
        if engine == "bailian":
            require(key, "缺API Key")
            location = "https://dashscope.aliyuncs.com"
        elif engine in {"local", "firered"}:
            require(base, "缺接口地址")
        else:
            missing.append("服务引擎不支持")
    elif kind == "tts":
        if engine == "edge_tts":
            location = "Edge TTS"
        elif engine == "azure_speech":
            require(settings.get("speech_key"), "缺Speech Key")
            require(settings.get("speech_region"), "缺服务区域")
            location = "Azure Speech · " + str(settings.get("speech_region", "") or "")
        elif engine == "tencent_tts":
            require(settings.get("secret_id"), "缺Secret ID")
            require(settings.get("secret_key"), "缺Secret Key")
            location = "tts.tencentcloudapi.com"
        elif engine == "doubaotts":
            if not (key or settings.get("api_key") or settings.get("apikey") or
                    (settings.get("appid") and settings.get("token"))):
                missing.append("缺API Key或AppID与Token")
            location = "openspeech.bytedance.com"
        elif engine == "qwen3_tts":
            require(key, "缺API Key")
            location = "dashscope.aliyuncs.com"
        elif engine == "mimo_tts":
            from app.services.mimo_tts import DEFAULT_BASE_URL
            require(key, "缺API Key")
            location = base or DEFAULT_BASE_URL
        elif engine in TTS_GROUPS:
            require(base, "缺接口地址")
            if engine == "soulvoice":
                require(key or settings.get("api_key"), "缺API Key")
        else:
            missing.append("服务引擎不支持")
    else:
        missing.append("服务类型不支持")
    if kind != "alignment" and location.lower().startswith(("http:", "https:")):
        from app.utils.openai_base_url_security import validate_openai_compatible_base_url
        try:
            validate_openai_compatible_base_url(location)
        except ValueError:
            missing.append("接口地址格式需检查")
        try:
            parsed = urlsplit(location)
            host = parsed.hostname or ""
            if ":" in host:
                host = "[" + host + "]"
            authority = host + (":" + str(parsed.port) if parsed.port else "")
            # Do not expose userinfo, query parameters or fragments in a public table.
            location = urlunsplit((parsed.scheme, authority, parsed.path, "", "")) if host else "接口地址格式需检查"
        except ValueError:
            location = "接口地址格式需检查"
    elif base and kind not in {"alignment"} and location == base:
        missing.append("接口地址格式需检查")
        location = "接口地址格式需检查"
    parameters = row.get("parameters", {})
    return {"id": row["id"], "name": row.get("name", ""), "model": row.get("model", ""),
            "kind": kind, "engine": engine, "capabilities": list(row.get("capabilities", [])),
            "enabled": row.get("enabled", True), "location": location,
            "parameters": {name: parameters[name] for name in ("thinking_level", "max_tokens", "context_tokens", "temperature", "top_p") if name in parameters},
            "missing": missing, "complete": not missing,
            "field_status": "待完善" if missing else "路径已填写" if kind == "alignment" else
                            "接口已填写" if engine in {"local", "firered", "indextts", "indextts_macos", "indextts2", "omnivoice", "voxcpm_05b", "voxcpm_2b"} else "字段已填写",
            "verification": "未验证"}


def validate_profile(row):
    row = deepcopy(row)
    if not isinstance(row, dict) or not isinstance(row.get("id"), str) or not _ID.fullmatch(row["id"]):
        raise ValueError("模型配置ID无效。")
    if not isinstance(row.get("name"), str) or not row["name"].strip() or len(row["name"]) > 100:
        raise ValueError("请填写100字以内的配置名称。")
    row["name"] = row["name"].strip()
    kind = row.get("kind")
    if kind not in {"chat", "tts", "alignment", "asr"} or type(row.get("enabled", True)) is not bool:
        raise ValueError("模型配置类型或状态无效。")
    allowed = {"text", "vision", "audio"} if kind == "chat" else {kind}
    caps = row.get("capabilities", list(allowed))
    if not isinstance(caps, list) or not caps or any(cap not in allowed for cap in caps):
        raise ValueError("模型能力标记无效。")
    row["capabilities"] = sorted(set(caps))
    row["enabled"] = row.get("enabled", True)
    for name in ("model", "base", "key"):
        if not isinstance(row.get(name, ""), str):
            raise ValueError(name + "应为文字。")
        row[name] = row.get(name, "").strip()
    if kind == "tts" and row.get("engine") not in TTS_GROUPS:
        raise ValueError("配音引擎无效。")
    if kind == "asr":
        known = {"local": "sensevoice", "firered": "FireRedASR", "bailian": "fun-asr"}
        if row.get("engine") not in known or row["model"] != known[row["engine"]]:
            raise ValueError("请选择既有支持的语音识别服务。")
        if row["engine"] == "bailian":
            row["base"] = "https://dashscope.aliyuncs.com"
    if kind == "tts" and row.get("engine") == "mimo_tts":
        from app.services.mimo_tts import DEFAULT_BASE_URL, MODELS
        row["base"] = row["base"] or DEFAULT_BASE_URL
        if row["model"] not in MODELS:
            raise ValueError("请选择当前支持的MiMo配音模型名称。")
    if kind == "chat":
        row["engine"] = "openai"
        from app.utils.openai_base_url_security import validate_openai_compatible_base_url
        if row["base"]:
            validate_openai_compatible_base_url(row["base"])
        if not isinstance(row.get("parameters", {}), dict):
            raise ValueError("生成参数应为对象。")
        params = {**DEFAULT_LLM_GENERATION_CONFIG, "context_tokens": 64000, **row.get("parameters", {})}
        for name, lower, upper in (("temperature", 0, 2), ("top_p", 0, 1)):
            val = params[name]
            if isinstance(val, bool) or not isinstance(val, (int, float)) or not isfinite(val) or not lower <= val <= upper:
                raise ValueError(name + "超出有效范围。")
        for name, minimum in (("max_tokens", 0), ("context_tokens", 8192)):
            if type(params[name]) is not int or params[name] < minimum:
                raise ValueError(name + "应为有效整数。")
        if params["thinking_level"] not in {"auto", "off", "low", "medium", "high"}:
            raise ValueError("思考等级无效。")
        row["parameters"] = params
    elif not isinstance(row.get("settings", {}), dict):
        raise ValueError("服务设置应为对象。")
    return row


def save_catalog(value):
    value = deepcopy(value)
    rows = [validate_profile(row) for row in value.get("profiles", [])]
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("模型配置ID重复。")
    defaults = value.get("defaults", {})
    for role, identifier in defaults.items():
        if role not in ROLE_CAPABILITIES or (identifier and not any(row["id"] == identifier and ROLE_CAPABILITIES[role] in row["capabilities"] for row in rows)):
            raise ValueError("默认模型选择无效。")
    updated = {"schema": 1, "profiles": rows, "defaults": dict(defaults)}
    with _lock, config._config_write_lock:
        previous = deepcopy(config.model_profiles)
        config.model_profiles.clear()
        config.model_profiles.update(updated)
        try:
            config.save_config()
        except Exception:
            config.model_profiles.clear()
            config.model_profiles.update(previous)
            raise


def import_legacy():
    """Explicit and idempotent; retain user edits and named entries."""
    current = catalog()
    if config.model_profiles.get("schema") == 1:
        known = {row["id"] for row in current["profiles"]}
        legacy = legacy_catalog()
        current["profiles"].extend(row for row in legacy["profiles"] if row["id"] not in known)
        for role, identifier in legacy["defaults"].items():
            current["defaults"].setdefault(role, identifier)
    save_catalog(current)


def profile_references(identifier):
    """Only scan saved references on explicit inspection/delete, never every page render."""
    from app.services.library.store import LibraryStore, read_json
    from app.services.audio_library import AudioLibraryStore
    from app.services.narration_task import TASK_ROOT
    references = []
    for role, value in catalog().get("defaults", {}).items():
        if value == identifier:
            references.append("默认选择：" + ROLE_LABELS[role])
    if identifier in config.ui.get("legacy_model_bindings", {}).values() or config.ui.get("legacy_tts_binding") == identifier:
        references.append("视频制作工作区")
    for item in LibraryStore().list_libraries():
        if identifier in item.get("settings", {}).get("build_settings", {}).get("model_bindings", {}).values():
            references.append("视频库：" + item["name"])
    for item in AudioLibraryStore().list_libraries():
        if identifier in item.get("settings", {}).get("model_bindings", {}).values():
            references.append("音频库：" + item["name"])
    for path in TASK_ROOT.glob("*/settings.json"):
        saved = read_json(path, default={})
        if identifier in saved.get("models", {}).get("_bindings", {}).values():
            references.append("项目：" + path.parent.name[:8])
    return references


def delete_profile(identifier):
    with _lock:
        references = profile_references(identifier)
        if references:
            raise ValueError("该配置仍被引用，请停用或先调整引用：" + "、".join(references[:12]))
        current = catalog()
        current["profiles"] = [row for row in current["profiles"] if row["id"] != identifier]
        save_catalog(current)


def available(role, *, engines=None):
    cap = ROLE_CAPABILITIES[role]
    return [row for row in catalog()["profiles"] if row.get("enabled", True) and cap in row["capabilities"]
            and (engines is None or row.get("engine") in engines)]


def get_profile(identifier, role=None, *, historical=False):
    row = next((row for row in catalog()["profiles"] if row["id"] == identifier), None)
    if row is None:
        raise ValueError("模型配置已不存在，请到模型配置工作区处理。")
    if not historical and not row.get("enabled", True):
        raise ValueError("模型配置已停用，请重新选择。")
    if role and ROLE_CAPABILITIES[role] not in row["capabilities"]:
        raise ValueError("所选模型不支持此用途。")
    return deepcopy(row)


def default_bindings():
    return deepcopy(catalog()["defaults"])


def resolve_models(bindings=None, *, roles=None, historical=False, allow_missing=False):
    selected = default_bindings() if bindings is None else dict(bindings)
    roles = roles or ("text", "text_fast", "vision", "text_backup", "vision_backup")
    result = {"proxy": deepcopy(config.proxy), "_bindings": dict(selected), "context_tokens": 64000}
    for role in roles:
        identifier = selected.get(role, "")
        if role == "asr":
            try:
                result[role] = asr_profile(identifier, historical=historical)
            except ValueError:
                if not allow_missing:
                    raise
                result[role] = {}
            continue
        if not identifier and role.endswith("_backup"):
            result[role] = {"enabled": False}
            continue
        if not identifier:
            result[role] = {}
            continue
        try:
            row = get_profile(identifier, role, historical=historical)
        except ValueError:
            if not allow_missing:
                raise
            result[role] = {}
            continue
        result[role] = {"model": row["model"], "base": row["base"], "key": row["key"], **row.get("parameters", {})}
        if role.endswith("_backup"):
            result[role]["enabled"] = True
    context_roles = ("audio",) if "audio" in roles else ("text", "vision")
    budgets = [result[role].get("context_tokens", 64000) for role in context_roles if result.get(role)]
    if budgets:
        result["context_tokens"] = min(budgets)
    return result


def tts_profile(identifier=None, *, engine=None, historical=False):
    legacy_engine = engine
    if identifier is None:
        identifier = "legacy_tts_" + engine if engine else default_bindings().get("tts")
    try:
        row = get_profile(identifier, "tts", historical=historical)
    except ValueError:
        if legacy_engine is None:
            raise
        row = next(row for row in legacy_catalog()["profiles"] if row["id"] == "legacy_tts_" + legacy_engine)
    settings = deepcopy(row.get("settings", {}))
    if row["engine"] == "mimo_tts":
        settings.update(model=row["model"], base_url=row["base"], api_key=row["key"])
    elif row["engine"] == "qwen3_tts":
        settings.update(model_name=row["model"], api_key=row["key"])
        if row["base"]:
            settings["api_url"] = row["base"]
    elif "api_url" in settings or row["base"]:
        settings["api_url"] = row["base"]
    if row["engine"] not in {"edge_tts", "mimo_tts", "qwen3_tts"} and row["key"]:
        settings["api_key"] = row["key"]
    if row["engine"] == "soulvoice":
        settings["model"] = row["model"]
    return {"engine": row["engine"], "settings": settings, "proxy": deepcopy(config.proxy),
            "profile_id": row["id"], "model": row["model"], "base_url": row["base"]}


def validate_narration_voice(profile, *, require_credentials=True):
    """Shared, local-only checks for a manuscript project's saved speech choice."""
    if not isinstance(profile, dict) or profile.get("engine") not in TTS_GROUPS:
        raise ValueError("请选择已接入的配音服务配置。")
    if profile.get("schema") == 1:
        if not isinstance(profile.get("settings"), dict):
            raise ValueError("配音服务快照不完整。")
        for address in (profile.get("base_url"), profile["settings"].get("api_url"), profile["settings"].get("base_url")):
            if address:
                parsed = urlsplit(address)
                if parsed.username or parsed.password or parsed.query or parsed.fragment:
                    raise ValueError("配音接口地址不能内嵌凭据或查询参数，请在中央配置填写服务凭据。")
    for field in ("rate", "pitch"):
        number = profile.get(field, 1.0 if field == "rate" else 0.0)
        if isinstance(number, bool) or not isinstance(number, (int, float)) or not isfinite(number) or (field == "rate" and not .5 <= number <= 2):
            raise ValueError("配音语速需在0.5至2倍之间，音调需为有效数字。")
    if not require_credentials:
        return
    engine = profile["engine"]
    if engine == "mimo_tts":
        from app.services.mimo_tts import MODELS, VOICES
        if not profile.get("api_key"):
            raise ValueError("请填写小米语音 API Key。")
        if profile.get("model") not in MODELS:
            raise ValueError("请选择已支持的MiMo配音模型。")
        if profile["model"] == MODELS[0] and profile.get("voice") not in VOICES:
            raise ValueError("请选择有效MiMo音色。")
        if profile["model"] == MODELS[1] and not str(profile.get("instruction") or "").strip():
            raise ValueError("请填写音色设计描述。")
        if profile["model"] == MODELS[2] and not Path(profile.get("reference_audio") or "__missing__").is_file():
            raise ValueError("请选择有效参考音频。")
        return
    if engine == "edge_tts":
        if not str(profile.get("voice", "")).strip():
            raise ValueError("请填写Edge音色。")
        return
    if profile.get("schema") != 1 or not isinstance(profile.get("settings"), dict):
        raise ValueError("配音服务快照不完整，请重新选择配置或创建配音副本。")
    settings = profile["settings"]
    row = {"id": profile.get("profile_id", ""), "name": profile.get("profile_id", ""), "kind": "tts", "engine": engine,
           "model": profile.get("model", ""), "base": profile.get("base_url", ""), "key": settings.get("api_key", ""), "settings": settings}
    missing = profile_summary(row)["missing"]
    if missing:
        raise ValueError("配音配置待完善：" + "、".join(missing))
    if not str(profile.get("voice", "")).strip():
        raise ValueError("请填写所选服务的音色／说话人ID，或参考音频路径。")
    reference = str(profile.get("voice", ""))
    if engine in {"indextts", "indextts2", "indextts_macos"}:
        if reference.startswith(engine + ":"):
            reference = reference[len(engine) + 1:]
        if not Path(reference).is_file():
            raise ValueError("所选IndexTTS需要本机可读取的参考音频路径。")
    if engine == "tencent_tts" and not reference.removeprefix("tencent:").isdigit():
        raise ValueError("腾讯音色ID需为数字，例如101001。")
    if engine in {"azure_speech", "tencent_tts"} and profile.get("base_url"):
        raise ValueError("Azure／腾讯SDK通过服务区域选择接口，请将中央配置的接口地址留空并填写区域。")
    if (engine == "voxcpm_2b" and settings.get("mode", "design") == "clone") or (engine == "omnivoice" and settings.get("mode") == "voice_clone"):
        reference = reference if Path(reference).is_file() else settings.get("reference_audio", "")
        if not Path(reference or "__missing__").is_file():
            raise ValueError("所选克隆模式需要有效参考音频。")


def restore_narration_voice(recorded, current, *, profile_id=None):
    """Restore only credentials; recorded non-secret sound parameters stay immutable."""
    saved = deepcopy(recorded)
    if saved.get("schema") == 1:
        identifier = profile_id or saved.get("profile_id")
        service = tts_profile(identifier, historical=True)
        if any(service.get(k) != saved.get(k) for k in ("engine", "model")) or \
                service.get("base_url", "").rstrip("/") != saved.get("base_url", "").rstrip("/") or \
                service["profile_id"] != saved.get("profile_id"):
            raise ValueError("历史配音配置的引擎、模型或接口已改变，请恢复对应配置或创建副本。")
        saved["settings"] = with_private_values(saved["settings"], service["settings"])
    elif saved.get("engine", "mimo_tts") == "mimo_tts":
        saved["api_key"] = credentials_for(saved, "tts", identifier=profile_id, current=current)
    saved["proxy"] = deepcopy(current.get("proxy", config.proxy))
    return saved


def redact_tts_error(value, service=None):
    """Remove active task credentials from provider exceptions and legacy logs."""
    message = str(value)
    service = service if service is not None else (current_tts() or {})
    def scrub(data, secret=False):
        nonlocal message
        if isinstance(data, dict):
            for key, item in data.items():
                scrub(item, secret or _private_key(key))
        elif isinstance(data, list):
            for item in data:
                scrub(item, secret)
        elif secret and isinstance(data, str) and len(data) >= 4:
            message = message.replace(data, "[REDACTED]")
    scrub(service.get("settings", {}))
    def hide_url(match):
        try:
            parsed = urlsplit(match.group())
            return urlunsplit((parsed.scheme, parsed.netloc.rsplit("@", 1)[-1], parsed.path,
                               "[REDACTED]" if parsed.query else "", ""))
        except ValueError:
            return "[REDACTED URL]"
    message = re.sub(r"https?://[^\s'\"<>]+", hide_url, message)
    return message


def alignment_profile(identifier=None, *, historical=False):
    row = get_profile(identifier or default_bindings().get("alignment"), "alignment", historical=historical)
    return deepcopy(row.get("settings", {}))


def asr_profile(identifier=None, *, historical=False):
    row = get_profile(identifier or default_bindings().get("asr"), "asr", historical=historical)
    settings = deepcopy(row.get("settings", {}))
    settings.update(backend=row["engine"], model=row["model"], api_key=row["key"])
    if row["engine"] in {"local", "firered"}:
        settings["api_url" if row["engine"] == "local" else "firered_api_url"] = row["base"]
    return {"engine": row["engine"], "settings": settings}


def asr_settings():
    active = current_models()
    return active["asr"]["settings"] if active and active.get("asr", {}).get("settings") else config.fun_asr


def credentials_for(saved, role, *, identifier=None, current=None):
    """Preserve recorded endpoint/parameters; restore only that service's credentials."""
    def matches(row):
        if role == "tts":
            engine = saved.get("engine", "mimo_tts")
            base = saved.get("base_url", "").rstrip("/")
            return row.get("engine") == engine and row.get("model") == saved.get("model", engine) and row.get("base", "").rstrip("/") == base
        return row.get("kind") == "chat" and row.get("base", "").rstrip("/") == saved.get("base", "").rstrip("/") and row.get("model") == saved.get("model")
    rows = catalog()["profiles"]
    if identifier:
        row = get_profile(identifier, role, historical=True)
        if not matches(row):
            raise ValueError("项目保存的模型接口与中央定义不同，请恢复对应配置，或明确使用新选择。")
        return row.get("key", "")
    if current and (current.get("base", current.get("base_url", "")).rstrip("/") == saved.get("base", saved.get("base_url", "")).rstrip("/")):
        credential = current.get("key", current.get("api_key", ""))
        if credential:
            return credential
    matches_with_key = {row.get("key", "") for row in rows if matches(row) and row.get("key")}
    if len(matches_with_key) > 1:
        raise ValueError("该接口存在多个账号，请明确选择项目对应模型配置。")
    return next(iter(matches_with_key), "")


@contextmanager
def runtime_configuration(*, models=None, tts=None):
    """Context-local compatibility for existing async/legacy calls; never change config."""
    previous = _runtime.get() or {}
    value = {**previous}
    if models is not None:
        value["models"] = deepcopy(models)
    if tts is not None:
        value["tts"] = deepcopy(tts)
    token = _runtime.set(value)
    try:
        yield value
    finally:
        _runtime.reset(token)


def current_models():
    return (_runtime.get() or {}).get("models")


def current_tts():
    return (_runtime.get() or {}).get("tts")


def tts_settings(group):
    profile = current_tts()
    if profile:
        if group == "proxy":
            return profile["proxy"]
        if TTS_GROUPS.get(profile["engine"]) == group:
            return profile["settings"]
    return getattr(config, group, {})

