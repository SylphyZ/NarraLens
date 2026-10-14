"""Shared configuration and wire parameters for SDK and material-library calls."""

from urllib.parse import urlparse

from app.config.defaults import DEFAULT_LLM_GENERATION_CONFIG


def generation_settings(app, model_type):
    return {name: app.get(f"{model_type}_openai_{name}", default)
            for name, default in DEFAULT_LLM_GENERATION_CONFIG.items()}


def model_family(model, base=""):
    name = str(model).lower().removeprefix("openai/")
    host = (urlparse(base).hostname or "").lower()
    if name.startswith("mimo-") or host.endswith("xiaomimimo.com"):
        return "mimo"
    if "deepseek" in name or host == "api.deepseek.com":
        return "deepseek"
    return "generic"


def thinking_mode(model, base, settings):
    level = str(settings.get("thinking_level") or "auto").lower()
    if level not in {"auto", "off", "low", "medium", "high", "max"}:
        raise ValueError("无效的思考等级。")
    family = model_family(model, base)
    if family == "mimo" and level not in {"auto", "off"}:
        raise ValueError("MiMo 当前公开接口只支持思考开关，请选择“自动”（模型默认开启）或“关闭”。")
    effective = level
    if family in {"mimo", "deepseek"} and level == "auto":
        effective = "default_enabled"
    elif family == "deepseek" and level == "medium":
        effective = "high"
    return family, level, effective


def build_generation_options(model, base, settings, *, max_tokens=None, json_output=False):
    family, level, effective = thinking_mode(model, base, settings)
    options = {}
    thinking = {}
    if family in {"mimo", "deepseek"}:
        if level == "off":
            thinking["thinking"] = {"type": "disabled"}
        elif level != "auto":
            thinking["thinking"] = {"type": "enabled"}
            thinking["reasoning_effort"] = effective
    elif level != "auto":
        thinking["reasoning_effort"] = "none" if level == "off" else level
    if thinking:
        options["extra_body"] = thinking
    # MiMo forces its sampling defaults while thinking; do not pretend UI values apply.
    if family != "mimo" or level == "off":
        options["temperature"] = float(settings.get("temperature", 1.0))
        options["top_p"] = float(settings.get("top_p", .95))
    limit = settings.get("max_tokens", 65536) if max_tokens is None else max_tokens
    if limit is not None and int(limit) > 0:
        options["max_completion_tokens" if family == "mimo" else "max_tokens"] = int(limit)
    if json_output:
        options["response_format"] = {"type": "json_object"}
    return options


def output_budget(model, base, settings, requested, input_tokens, context_tokens):
    """The UI value is a ceiling; retries grow without exceeding the remaining context."""
    family, level, effective = thinking_mode(model, base, settings)
    configured = int(settings.get("max_tokens") or 65536)
    ceiling = min(configured, int(context_tokens) - int(input_tokens) - 512)
    if ceiling < 256:
        raise ValueError("输入已占满本流程的上下文预算，请缩小分析窗口或候选数量。")
    initial = 4096 if level == "off" or (family == "generic" and level == "auto") else 16384 if effective == "low" else 32768
    return min(ceiling, max(initial, int(requested))), ceiling
