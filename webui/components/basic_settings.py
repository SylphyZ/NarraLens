import traceback

import streamlit as st
import os
from app.config import config
from app.config.defaults import (
    DEFAULT_LLM_GENERATION_CONFIG,
    DEFAULT_LLM_THINKING_LEVELS,
    DEFAULT_OPENAI_COMPATIBLE_BASE_URL,
    DEFAULT_OPENAI_COMPATIBLE_PROVIDER,
    DEFAULT_TEXT_LLM_PROVIDER,
    DEFAULT_TEXT_OPENAI_FAST_MODEL_NAME,
    DEFAULT_TEXT_OPENAI_MODEL_NAME,
    DEFAULT_VISION_LLM_PROVIDER,
    DEFAULT_VISION_OPENAI_MODEL_NAME,
    get_openai_compatible_ui_values,
    normalize_openai_compatible_model_name as normalize_openai_compatible_model_id,
)
from app.utils.openai_base_url_security import (
    openai_compatible_base_url_warning,
    validate_openai_compatible_base_url,
)
from app.utils import utils
from webui.utils.file_utils import folder_input
from loguru import logger
from app.services.llm.unified_service import UnifiedLLMService

# 需要用户手动填写 Base URL 的 OpenAI 兼容网关及其默认接口
OPENAI_COMPATIBLE_GATEWAY_BASE_URLS = {
    "siliconflow": "https://api.siliconflow.cn/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "moonshot": "https://api.moonshot.cn/v1",
    "gemini(openai)": "",
}


def build_base_url_help(provider: str, model_type: str, tr=lambda key: key) -> tuple[str, bool, str]:
    """
    根据 provider 返回 Base URL 的帮助文案

    Returns:
        help_text: 显示在输入框的帮助内容
        requires_base: 是否强制提示必须填写 Base URL
        placeholder: 推荐的默认值（可为空字符串）
    """
    default_help = tr("Custom API endpoint help")
    provider_key = (provider or "").lower()
    example_url = OPENAI_COMPATIBLE_GATEWAY_BASE_URLS.get(provider_key)

    if example_url is not None:
        extra = f"\n{tr('Recommended API endpoint')}: {example_url}" if example_url else ""
        help_text = (
            f"{tr('OpenAI compatible gateway help').format(model_type=model_type)}"
            f"{extra}"
        )
        return help_text, True, example_url

    return default_help, False, ""


def validate_api_key(api_key: str, provider: str) -> tuple[bool, str]:
    """验证API密钥格式"""
    if not api_key or not api_key.strip():
        return False, f"{provider} API密钥不能为空"

    # 基本长度检查
    if len(api_key.strip()) < 10:
        return False, f"{provider} API密钥长度过短，请检查是否正确"

    return True, ""


def validate_base_url(base_url: str, provider: str) -> tuple[bool, str]:
    """验证Base URL格式"""
    if not base_url or not base_url.strip():
        return True, ""  # base_url可以为空

    base_url = base_url.strip()
    if not (base_url.startswith('http://') or base_url.startswith('https://')):
        return False, f"{provider} Base URL必须以http://或https://开头"

    try:
        validate_openai_compatible_base_url(base_url)
    except ValueError as exc:
        return False, f"{provider} Base URL格式无效: {exc}"

    return True, ""


def show_base_url_security_warning(base_url: str) -> None:
    warning = openai_compatible_base_url_warning(base_url)
    if warning:
        st.warning(warning)


def validate_model_name(model_name: str, provider: str) -> tuple[bool, str]:
    """验证模型名称"""
    if not model_name or not model_name.strip():
        return False, f"{provider} 模型名称不能为空"

    return True, ""


def validate_openai_compatible_model_name(model_name: str, model_type: str) -> tuple[bool, str]:
    """验证 OpenAI 兼容 模型名称格式
    
    Args:
        model_name: 模型名称，应为 provider/model 格式
        model_type: 模型类型（如"视觉分析"、"文案生成"）
        
    Returns:
        (是否有效, 错误消息)
    """
    if not model_name or not model_name.strip():
        return False, f"{model_type} 模型名称不能为空"
    
    model_name = model_name.strip()
    
    # OpenAI 兼容 推荐格式：provider/model（如 gemini/gemini-2.0-flash-lite）
    # 但也支持直接的模型名称（如 gpt-4o，OpenAI 兼容 会自动推断 provider）
    
    # 检查是否包含 provider 前缀（推荐格式）
    if "/" in model_name:
        parts = model_name.split("/")
        if len(parts) < 2 or not parts[0] or not parts[1]:
            return False, f"{model_type} 模型名称格式错误。推荐格式: provider/model （如 gemini/gemini-2.0-flash-lite）"
        
        # 验证 provider 名称（只允许字母、数字、下划线、连字符）
        provider = parts[0]
        if not provider.replace("-", "").replace("_", "").isalnum():
            return False, f"{model_type} Provider 名称只能包含字母、数字、下划线和连字符"
    else:
        # 直接模型名称也是有效的（OpenAI 兼容 会自动推断）
        # 但给出警告建议使用完整格式
        logger.debug(f"{model_type} 模型名称未包含 provider 前缀，OpenAI 兼容 将自动推断")
    
    # 基本长度检查
    if len(model_name) < 3:
        return False, f"{model_type} 模型名称过短"
    
    if len(model_name) > 200:
        return False, f"{model_type} 模型名称过长"
    
    return True, ""


def normalize_openai_compatible_model_name(model_name: str) -> str:
    """仅剥离误保存的 openai/ 前缀，保留完整模型名称。"""
    return normalize_openai_compatible_model_id(
        model_name,
        provider=DEFAULT_OPENAI_COMPATIBLE_PROVIDER,
    )


def show_config_validation_errors(errors: list):
    """显示配置验证错误"""
    if errors:
        for error in errors:
            st.error(error)


def update_app_config_if_changed(key: str, value) -> bool:
    """Update app config only when the value really changed."""
    if config.app.get(key) == value:
        return False

    config.app[key] = value
    return True


def render_openai_compatible_protocol_field(tr, label_key: str, key: str) -> None:
    """Render the fixed OpenAI-compatible protocol as a non-selectable field."""
    st.text_input(
        tr(label_key),
        value=tr("OpenAI compatible protocol"),
        help=tr("OpenAI compatible protocol help"),
        disabled=True,
        key=key,
    )










def render_basic_settings(tr):
    """Global language, proxy and search settings; model services have their own workspace."""
    with st.expander(tr("Basic Settings"), expanded=False):
        render_language_settings(tr)
        render_proxy_settings(tr)
        render_tavily_search_settings(tr)
  # 文案生成模型设置


def render_generation_settings(tr):
    """渲染通用生成参数。"""
    st.divider()
    st.subheader(tr("Generation Settings"))
    if 'temperature' not in st.session_state:
        st.session_state['temperature'] = DEFAULT_LLM_GENERATION_CONFIG["temperature"]
    st.slider("temperature", 0.0, 2.0, key="temperature")


def render_language_settings(tr):
    st.subheader(tr("Proxy Settings"))

    """渲染语言设置"""
    system_locale = utils.get_system_locale()
    i18n_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "i18n")
    locales = utils.load_locales(i18n_dir)

    display_languages = []
    selected_index = 0
    for i, code in enumerate(locales.keys()):
        display_languages.append(f"{code} - {locales[code].get('Language')}")
        if code == st.session_state.get('ui_language', system_locale):
            selected_index = i

    selected_language = st.selectbox(
        tr("Language"),
        options=display_languages,
        index=selected_index
    )

    if selected_language:
        code = selected_language.split(" - ")[0].strip()
        st.session_state['ui_language'] = code
        config.ui['language'] = code


def render_proxy_settings(tr):
    """渲染代理设置"""
    # 获取当前代理状态
    proxy_enabled = config.proxy.get("enabled", False)
    proxy_url_http = config.proxy.get("http")
    proxy_url_https = config.proxy.get("https")

    # 添加代理开关
    proxy_enabled = st.checkbox(tr("Enable Proxy"), value=proxy_enabled)
    
    # 保存代理开关状态
    config.proxy["enabled"] = proxy_enabled

    # 只有在代理启用时才显示代理设置输入框
    if proxy_enabled:
        HTTP_PROXY = st.text_input(tr("HTTP_PROXY"), value=proxy_url_http)
        HTTPS_PROXY = st.text_input(tr("HTTPs_PROXY"), value=proxy_url_https)

        if HTTP_PROXY and HTTPS_PROXY:
            config.proxy["http"] = HTTP_PROXY
            config.proxy["https"] = HTTPS_PROXY
            os.environ["HTTP_PROXY"] = HTTP_PROXY
            os.environ["HTTPS_PROXY"] = HTTPS_PROXY
            # logger.debug(f"代理已启用: {HTTP_PROXY}")
    else:
        # 当代理被禁用时，清除环境变量和配置
        os.environ.pop("HTTP_PROXY", None)
        os.environ.pop("HTTPS_PROXY", None)
        config.proxy["http"] = ""
        config.proxy["https"] = ""

    # 剪映草稿地址设置
    st.subheader(tr("Jianying Draft Settings"))
    jianying_draft_path = folder_input(
        tr("Jianying Draft Folder Path"),
        value=config.ui.get("jianying_draft_path", ""),
        help=tr("Jianying Draft Folder Path Help"), key="jianying_draft_folder", tr=tr,
    )
    config.ui["jianying_draft_path"] = jianying_draft_path


def render_tavily_search_settings(tr):
    """Render Tavily API key settings used by short drama web search."""
    st.subheader(tr("Tavily Search Settings"))
    st.markdown(
        f"{tr('API Key URL')}: "
        "[https://app.tavily.com](https://app.tavily.com)"
    )

    tavily_api_key = st.text_input(
        tr("Tavily API Key"),
        value=config.app.get("tavily_api_key", ""),
        type="password",
        help=tr("Tavily API Key Help"),
        key="tavily_api_key_input",
    )

    if update_app_config_if_changed("tavily_api_key", str(tavily_api_key or "").strip()):
        try:
            config.save_config()
            st.session_state["tavily_api_key"] = str(tavily_api_key or "").strip()
            st.success(tr("Tavily config saved"))
        except Exception as e:
            st.error(f"{tr('Failed to save config')}: {str(e)}")
            logger.error(f"保存 Tavily 配置失败: {str(e)}")


def test_vision_model_connection(api_key, base_url, model_name, provider, tr):
    """测试视觉模型连接

    Args:
        api_key: API密钥
        base_url: 基础URL
        model_name: 模型名称
        provider: 提供商名称

    Returns:
        bool: 连接是否成功
        str: 测试结果消息
    """
    import requests
    logger.debug(f"大模型连通性测试: {base_url} 模型: {model_name} apikey: {api_key}")
    if provider.lower() == 'gemini':
        # 原生Gemini API测试
        try:
            # 构建请求数据
            request_data = {
                "contents": [{
                    "parts": [{"text": "直接回复我文本'当前网络可用'"}]
                }]
            }

            # 构建请求URL
            api_base_url = base_url
            url = f"{api_base_url}/models/{model_name}:generateContent"
            # 发送请求
            response = requests.post(
                url,
                json=request_data,
                headers={
                    "x-goog-api-key": api_key,
                    "Content-Type": "application/json"
                    },
                timeout=10
            )

            if response.status_code == 200:
                return True, tr("原生Gemini模型连接成功")
            else:
                return False, f"{tr('原生Gemini模型连接失败')}: HTTP {response.status_code}"
        except Exception as e:
            return False, f"{tr('原生Gemini模型连接失败')}: {str(e)}"
    elif provider.lower() == 'gemini(openai)':
        # OpenAI兼容的Gemini代理测试
        try:
            base_url = validate_openai_compatible_base_url(base_url)
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }

            test_url = f"{base_url.rstrip('/')}/chat/completions"
            test_data = {
                "model": model_name,
                "messages": [
                    {"role": "user", "content": "直接回复我文本'当前网络可用'"}
                ],
                "stream": False
            }

            response = requests.post(test_url, headers=headers, json=test_data, timeout=10)
            if response.status_code == 200:
                return True, tr("OpenAI兼容Gemini代理连接成功")
            else:
                return False, f"{tr('OpenAI兼容Gemini代理连接失败')}: HTTP {response.status_code}"
        except Exception as e:
            return False, f"{tr('OpenAI兼容Gemini代理连接失败')}: {str(e)}"
    else:
        from openai import OpenAI
        try:
            base_url = validate_openai_compatible_base_url(base_url)
            client = OpenAI(
                api_key=api_key,
                base_url=base_url,
            )

            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": "You are a helpful assistant."}],
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": "https://help-static-aliyun-doc.aliyuncs.com/file-manage-files/zh-CN/20241022/emyrja/dog_and_girl.jpeg"
                                },
                            },
                            {"type": "text", "text": "回复我网络可用即可"},
                        ],
                    },
                ],
            )
            if response and response.choices:
                return True, tr("QwenVL model is available")
            else:
                return False, tr("QwenVL model returned invalid response")

        except Exception as e:
            # logger.debug(api_key)
            # logger.debug(base_url)
            # logger.debug(model_name)
            return False, f"{tr('QwenVL model is not available')}: {str(e)}"




def test_openai_compatible_vision_model(api_key: str, base_url: str, model_name: str, tr) -> tuple[bool, str]:
    """测试 OpenAI 兼容视觉模型连接。"""
    try:
        import base64
        import io
        from openai import OpenAI
        from PIL import Image

        logger.debug(
            f"OpenAI 兼容视觉模型连通性测试: model={model_name}, base_url={base_url}"
        )

        base_url = validate_openai_compatible_base_url(base_url)
        client = OpenAI(
            api_key=api_key,
            base_url=base_url or None,
            timeout=10.0,
            max_retries=1,
        )

        # 创建测试图片（64x64 白色像素）
        test_image = Image.new("RGB", (64, 64), color="white")
        img_buffer = io.BytesIO()
        test_image.save(img_buffer, format="JPEG")
        base64_image = base64.b64encode(img_buffer.getvalue()).decode("utf-8")

        response = client.chat.completions.create(
            model=normalize_openai_compatible_model_name(model_name),
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "请直接回复'连接成功'"},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                        },
                    ],
                }
            ],
            temperature=0.1,
            max_tokens=50,
        )

        if response and response.choices and len(response.choices) > 0:
            return True, f"OpenAI 兼容视觉模型连接成功 ({model_name})"
        return False, "OpenAI 兼容视觉模型返回空响应"
    except Exception as e:
        error_msg = str(e)
        logger.error(f"OpenAI 兼容视觉模型测试失败: {error_msg}")
        if "authentication" in error_msg.lower() or "api_key" in error_msg.lower():
            return False, "认证失败，请检查 API Key 是否正确"
        if "not found" in error_msg.lower() or "404" in error_msg:
            return False, "模型不存在，请检查模型名称是否正确"
        if "rate limit" in error_msg.lower():
            return False, "超出速率限制，请稍后重试"
        return False, f"连接失败: {error_msg}"


def test_openai_compatible_text_model(api_key: str, base_url: str, model_name: str, tr) -> tuple[bool, str]:
    """测试 OpenAI 兼容文本模型连接。"""
    try:
        from openai import OpenAI

        logger.debug(
            f"OpenAI 兼容文本模型连通性测试: model={model_name}, base_url={base_url}"
        )

        base_url = validate_openai_compatible_base_url(base_url)
        client = OpenAI(
            api_key=api_key,
            base_url=base_url or None,
            timeout=10.0,
            max_retries=1,
        )

        response = client.chat.completions.create(
            model=normalize_openai_compatible_model_name(model_name),
            messages=[{"role": "user", "content": "请直接回复'连接成功'"}],
            temperature=0.1,
            max_tokens=20,
        )

        if response and response.choices and len(response.choices) > 0:
            return True, f"OpenAI 兼容文本模型连接成功 ({model_name})"
        return False, "OpenAI 兼容文本模型返回空响应"
    except Exception as e:
        error_msg = str(e)
        logger.error(f"OpenAI 兼容文本模型测试失败: {error_msg}")
        if "authentication" in error_msg.lower() or "api_key" in error_msg.lower():
            return False, "认证失败，请检查 API Key 是否正确"
        if "not found" in error_msg.lower() or "404" in error_msg:
            return False, "模型不存在，请检查模型名称是否正确"
        if "rate limit" in error_msg.lower():
            return False, "超出速率限制，请稍后重试"
        return False, f"连接失败: {error_msg}"



def test_text_model_connection(api_key, base_url, model_name, provider, tr):
    """测试文本模型连接
    
    Args:
        api_key: API密钥
        base_url: 基础URL
        model_name: 模型名称
        provider: 提供商名称
    
    Returns:
        bool: 连接是否成功
        str: 测试结果消息
    """
    import requests
    logger.debug(f"大模型连通性测试: {base_url} 模型: {model_name} apikey: {api_key}")

    try:
        # 构建统一的测试请求（遵循OpenAI格式）
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        # 特殊处理Gemini
        if provider.lower() == 'gemini':
            # 原生Gemini API测试
            try:
                # 构建请求数据
                request_data = {
                    "contents": [{
                        "parts": [{"text": "直接回复我文本'当前网络可用'"}]
                    }]
                }

                # 构建请求URL
                api_base_url = base_url
                url = f"{api_base_url}/models/{model_name}:generateContent"

                # 发送请求
                response = requests.post(
                    url,
                    json=request_data,
                    headers={
                        "x-goog-api-key": api_key,
                        "Content-Type": "application/json"
                    },
                    timeout=10
                )

                if response.status_code == 200:
                    return True, tr("原生Gemini模型连接成功")
                else:
                    return False, f"{tr('原生Gemini模型连接失败')}: HTTP {response.status_code}"
            except Exception as e:
                return False, f"{tr('原生Gemini模型连接失败')}: {str(e)}"

        elif provider.lower() == 'gemini(openai)':
            # OpenAI兼容的Gemini代理测试
            test_url = f"{base_url.rstrip('/')}/chat/completions"
            test_data = {
                "model": model_name,
                "messages": [
                    {"role": "user", "content": "直接回复我文本'当前网络可用'"}
                ],
                "stream": False
            }

            response = requests.post(test_url, headers=headers, json=test_data, timeout=10)
            if response.status_code == 200:
                return True, tr("OpenAI兼容Gemini代理连接成功")
            else:
                return False, f"{tr('OpenAI兼容Gemini代理连接失败')}: HTTP {response.status_code}"
        else:
            test_url = f"{base_url.rstrip('/')}/chat/completions"

            # 构建测试消息
            test_data = {
                "model": model_name,
                "messages": [
                    {"role": "user", "content": "直接回复我文本'当前网络可用'"}
                ],
                "stream": False
            }

            # 发送测试请求
            response = requests.post(
                test_url,
                headers=headers,
                json=test_data,
            )
            # logger.debug(model_name)
            # logger.debug(api_key)
            # logger.debug(test_url)
            if response.status_code == 200:
                return True, tr("Text model is available")
            else:
                return False, f"{tr('Text model is not available')}: HTTP {response.status_code}"
            
    except Exception as e:
        logger.error(traceback.format_exc())
        return False, f"{tr('Connection failed')}: {str(e)}"



    # # Cloudflare 特殊配置
    # if text_provider == 'cloudflare':
    #     st_account_id = st.text_input(
    #         tr("Account ID"),
    #         value=config.app.get(f"text_{text_provider}_account_id", "")
    #     )
    #     if st_account_id:
    #         config.app[f"text_{text_provider}_account_id"] = st_account_id
