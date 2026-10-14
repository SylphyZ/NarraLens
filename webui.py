from copy import deepcopy
from contextvars import copy_context
from pathlib import Path
import streamlit as st
import os
import sys
import time
from html import escape
from loguru import logger
from app.config import config
from app.utils import utils
from app.utils import ffmpeg_utils
from app.models import const
from app.models.schema import VideoClipParams, VideoAspect

APP_NAME_ZH = "叙镜"
APP_NAME_EN = "NarraLens"
APP_TITLE = f"{APP_NAME_ZH} · {APP_NAME_EN}"
UPSTREAM_URL = "https://github.com/linyqh/NarratoAI"
WORKSPACE_LABELS = {"home": "首页", "narration_create": "项目管理", "narration_progress": "制作工作台",
                    "library": "视频素材库", "audio_library": "音频素材库", "models": "模型配置", "legacy": "视频制作"}


# 初始化配置 - 必须是第一个 Streamlit 命令
st.set_page_config(
    page_title=APP_TITLE,
    page_icon="📽️",
    layout="wide",
    initial_sidebar_state="auto",
    menu_items={
        "Report a bug": None,
        'About': f"# {APP_TITLE}\n\nVersion: v{config.project_version}\n\n"
                 "文案驱动的视频制作工作台 · Manuscript-driven video production.\n\n"
                 f"基于 [NarratoAI]({UPSTREAM_URL}) 的独立衍生版本。 Independently maintained derivative.\n\n"
                 "MIT License · Copyright (c) 2024 linyq\n\n"
                 f"[上游许可证 / Upstream license]({UPSTREAM_URL}/blob/main/LICENSE)"
    },
)

# 设置页面样式
hide_streamlit_style = """
<style>#root > div:nth-child(1) > div > div > div > div > section > div {padding-top: 2rem; padding-bottom: 10px; padding-left: 20px; padding-right: 20px;}</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def init_log():
    """每个进程初始化一次日志配置。"""
    logger.remove()
    _lvl = "INFO"  # 改为 INFO 级别，过滤掉 DEBUG 日志

    def format_record(record):
        # 简化日志格式化处理，不尝试按特定字符串过滤torch相关内容
        file_path = record["file"].path
        relative_path = os.path.relpath(file_path, config.root_dir)
        record["file"].path = f"./{relative_path}"
        record['message'] = record['message'].replace(config.root_dir, ".")

        _format = '<green>{time:%Y-%m-%d %H:%M:%S}</> | ' + \
                  '<level>{level}</> | ' + \
                  '"{file.path}:{line}":<blue> {function}</> ' + \
                  '- <level>{message}</>' + "\n"
        return _format

    # 添加日志过滤器
    def log_filter(record):
        """过滤不必要的日志消息"""
        # 过滤掉启动时的噪音日志（即使在 DEBUG 模式下也可以选择过滤）
        ignore_patterns = [
            "Examining the path of torch.classes raised",
            "torch.cuda.is_available()",
            "CUDA initialization"
        ]
        return not any(pattern in record["message"] for pattern in ignore_patterns)

    return logger.add(
        sys.stdout,
        level=_lvl,
        format=format_record,
        colorize=True,
        filter=log_filter
    )


def init_global_state():
    """初始化全局状态"""
    if 'video_clip_json' not in st.session_state:
        st.session_state['video_clip_json'] = []
    if 'video_plot' not in st.session_state:
        st.session_state['video_plot'] = ''
    if 'ui_language' not in st.session_state:
        st.session_state['ui_language'] = config.ui.get("language", utils.get_system_locale())
    # 移除subclip_videos初始化 - 现在使用统一裁剪策略


def tr(key):
    """翻译函数"""
    i18n_dir = os.path.join(os.path.dirname(__file__), "webui", "i18n")
    locales = utils.load_locales(i18n_dir)
    loc = locales.get(st.session_state['ui_language'], {})
    return loc.get("Translation", {}).get(key, key)


VIDEO_GENERATION_STEP_LABELS = [
    "正在加载剪辑脚本",
    "正在生成 TTS 配音",
    "正在按脚本裁剪视频片段",
    "正在合并配音和字幕",
    "正在合并视频片段",
    "正在合成最终视频",
]


def _safe_int(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _format_optional_percent(value):
    try:
        percent = max(0.0, min(100.0, float(value)))
    except (TypeError, ValueError):
        return None
    if percent.is_integer():
        return str(int(percent))
    return f"{percent:.1f}"


def _render_generation_status(task: dict | None) -> str:
    task = task or {}
    state = task.get("state")
    current_step = _safe_int(task.get("step_current"), 0)
    step_total = _safe_int(task.get("step_total"), len(VIDEO_GENERATION_STEP_LABELS))
    message = str(task.get("message") or "")
    ffmpeg_percent = _format_optional_percent(task.get("ffmpeg_progress"))

    if current_step <= 0:
        return f"<div style='font-weight:650;color:#262730;'>{escape(message or '正在生成视频，请稍候...')}</div>"

    lines = []
    for index, default_label in enumerate(VIDEO_GENERATION_STEP_LABELS, start=1):
        is_current = index == current_step
        is_complete = state == const.TASK_STATE_COMPLETE
        is_done = is_complete or index < current_step
        label = message if is_current and message else default_label

        suffix = f"{index}/{step_total}"
        if (
            is_current
            and index == step_total
            and ffmpeg_percent is not None
            and not is_complete
        ):
            suffix = f"{suffix}，ffmpeg {ffmpeg_percent}%"

        color = "#262730" if is_current else "#8b9099" if is_done else "#b9bec7"
        weight = "650" if is_current else "500"
        lines.append(
            "<div style='"
            "font-size:1.02rem;"
            "line-height:1.85;"
            "margin:0.28rem 0;"
            f"color:{color};"
            f"font-weight:{weight};"
            "'>"
            f"{escape(label)} <span style='white-space:nowrap;'>({escape(suffix)})</span>"
            "</div>"
        )

    return "".join(lines)


def get_help_text():
    """返回带当前项目版本号的帮助文案"""
    return tr("Get Help").replace("🎉🎉🎉", f" v{config.project_version}")


def render_generate_button():
    """渲染生成按钮和处理逻辑"""
    if st.button(tr("Generate Video"), use_container_width=True, type="primary"):
        from webui.components import audio_settings, script_settings, subtitle_settings, video_settings
        from app.services import task as tm
        from app.services import state as sm
        from app.models import const
        import threading
        import time
        import uuid

        config.save_config()

        # 移除task_id检查 - 现在使用统一裁剪策略，不再需要预裁剪
        # 直接检查必要的文件是否存在
        if not st.session_state.get('video_clip_json_path'):
            st.error(tr("Script file cannot be empty"))
            return
        if not st.session_state.get('video_origin_path'):
            st.error(tr("Video file cannot be empty"))
            return

        # 获取所有参数
        script_params = script_settings.get_script_params()
        video_params = video_settings.get_video_params()
        audio_params = audio_settings.get_audio_params()
        subtitle_params = subtitle_settings.get_subtitle_params()

        # 合并所有参数
        all_params = {
            **script_params,
            **video_params,
            **audio_params,
            **subtitle_params
        }

        # 创建参数对象
        params = VideoClipParams(**all_params)

        # 生成一个新的task_id用于本次处理
        task_id = str(uuid.uuid4())

        @st.dialog(tr("Generating Video"), width="large")
        def generate_video_dialog():
            st.markdown(
                """
                <style>
                    div[data-testid="stDialog"] div[data-testid="stStatusWidget"] {
                        margin-top: 0.25rem;
                    }
                    div[data-testid="stDialog"] div[data-testid="stProgress"] {
                        margin-bottom: 0.75rem;
                    }
                    div[data-testid="stDialog"] video {
                        max-height: 62vh;
                        object-fit: contain;
                        background: #000;
                    }
                </style>
                """,
                unsafe_allow_html=True,
            )

            progress_bar = st.progress(0)
            status_panel = st.status(tr("Generating Video"), expanded=True)
            with status_panel:
                status_placeholder = st.empty()
                status_placeholder.markdown(
                    _render_generation_status(None),
                    unsafe_allow_html=True,
                )

            service_profile = deepcopy(st.session_state.get("legacy_tts_runtime"))

            def run_task():
                try:
                    tm.start_subclip_unified(
                        task_id=task_id,
                        params=params,
                        service_profile=service_profile,
                    )
                except Exception as e:
                    logger.error(f"任务执行失败: {e}")
                    current_task = sm.state.get_task(task_id) or {}
                    sm.state.update_task(
                        task_id,
                        state=const.TASK_STATE_FAILED,
                        progress=current_task.get("progress", 0),
                        message=str(e),
                    )

            # 在新线程中启动任务
            thread = threading.Thread(target=copy_context().run, args=(run_task,))
            thread.start()

            last_status_key = None

            # 轮询任务状态
            while True:
                task = sm.state.get_task(task_id)
                if task:
                    progress = task.get("progress", 0)
                    state = task.get("state")

                    try:
                        progress = int(progress)
                    except (TypeError, ValueError):
                        progress = 0
                    progress = max(0, min(progress, 100))

                    # 更新进度条和阶段状态
                    progress_bar.progress(progress / 100)
                    current_message = task.get("message") or f"Processing... {progress}%"
                    status_key = (
                        state,
                        progress,
                        current_message,
                        task.get("step_current"),
                        task.get("step_total"),
                        task.get("ffmpeg_progress"),
                    )
                    if status_key != last_status_key:
                        status_placeholder.markdown(
                            _render_generation_status(task),
                            unsafe_allow_html=True,
                        )
                        last_status_key = status_key

                    if state == const.TASK_STATE_COMPLETE:
                        status_panel.update(
                            label=tr("Video Generation Completed"),
                            state="complete",
                            expanded=False,
                        )
                        progress_bar.progress(1.0)

                        # 显示结果
                        video_files = task.get("videos", [])
                        try:
                            if video_files:
                                aspect = getattr(params, "video_aspect", "")
                                aspect = getattr(aspect, "value", aspect)
                                preview_width = 320 if aspect in {
                                    VideoAspect.portrait.value,
                                    VideoAspect.portrait_2.value,
                                } else 600
                                for url in video_files:
                                    _, preview_col, _ = st.columns([1, 2, 1])
                                    with preview_col:
                                        st.video(url, width=preview_width)
                        except Exception as e:
                            logger.error(f"播放视频失败: {e}")

                        st.success(tr("Video Generation Completed"))
                        break

                    if state == const.TASK_STATE_FAILED:
                        status_panel.update(
                            label=f"{tr('Task failed')}: {task.get('message', 'Unknown error')}",
                            state="error",
                            expanded=True,
                        )
                        st.error(f"{tr('Task failed')}: {task.get('message', 'Unknown error')}")
                        break

                time.sleep(0.5)

        generate_video_dialog()


def get_voice_name_for_tts_engine(tts_engine: str) -> str:
    """根据TTS引擎获取用户选择的音色"""
    if tts_engine == 'edge_tts':
        return config.ui.get('edge_voice_name', 'zh-CN-XiaoxiaoNeural-Female')
    if tts_engine == 'azure_speech':
        return config.ui.get('azure_voice_name', 'zh-CN-XiaoxiaoMultilingualNeural')
    if tts_engine == 'tencent_tts':
        return f"tencent:{config.ui.get('tencent_voice_type', '101001')}"
    if tts_engine == 'qwen3_tts':
        return f"qwen3:{config.ui.get('qwen_voice_type', 'Cherry')}"
    if tts_engine == 'mimo_tts':
        model = st.session_state.get('legacy_tts_runtime', {}).get('settings', config.mimo_tts).get('model', 'mimo-v2.5-tts')
        selected_voice = config.mimo_tts.get('voice', '冰糖') if model == 'mimo-v2.5-tts' else model
        return f"mimo:{selected_voice}"
    if tts_engine == config.INDEXTTS2_ENGINE:
        reference_audio = config.indextts2.get('reference_audio', '')
        if reference_audio:
            return f"{config.INDEXTTS2_VOICE_PREFIX}{reference_audio}"
        return config.ui.get('voice_name', '')
    if config.normalize_tts_engine_name(tts_engine) == config.INDEXTTS_ENGINE:
        reference_audio = config.indextts.get('reference_audio', '')
        if reference_audio:
            return f"{config.INDEXTTS_VOICE_PREFIX}{reference_audio}"
        return config.ui.get('voice_name', '')
    if tts_engine == config.OMNIVOICE_ENGINE:
        mode = config.omnivoice.get('mode', 'auto')
        reference_audio = config.omnivoice.get('reference_audio', '')
        if mode == 'voice_clone' and reference_audio:
            return f"{config.OMNIVOICE_VOICE_PREFIX}{reference_audio}"
        return f"{config.OMNIVOICE_VOICE_PREFIX}{mode}"
    if tts_engine == 'doubaotts':
        return config.ui.get('doubaotts_voice_type', 'BV700_streaming')
    if tts_engine == 'soulvoice':
        voice_uri = config.soulvoice.get('voice_uri', '')
        if voice_uri and not voice_uri.startswith(('soulvoice:', 'speech:')):
            return f"soulvoice:{voice_uri}"
        return voice_uri
    return config.ui.get('voice_name', config.ui.get('edge_voice_name', 'zh-CN-XiaoxiaoNeural-Female'))


def get_jianying_export_params(draft_name=None) -> VideoClipParams:
    """获取导出到剪映草稿的参数"""
    tts_engine = st.session_state.get('tts_engine', config.ui.get('tts_engine', 'edge_tts'))
    voice_name = get_voice_name_for_tts_engine(tts_engine)
    voice_rate = st.session_state.get('voice_rate', 1.0)
    voice_pitch = st.session_state.get('voice_pitch', 1.0)
    subtitle_paths = st.session_state.get('subtitle_paths', [])
    if isinstance(subtitle_paths, str):
        subtitle_paths = [subtitle_paths]
    subtitle_paths = [
        path for path in subtitle_paths
        if isinstance(path, str) and path.strip()
    ]
    if not subtitle_paths and st.session_state.get('subtitle_path'):
        subtitle_paths = [st.session_state.get('subtitle_path')]
    
    return VideoClipParams(
        video_clip_json_path=st.session_state['video_clip_json_path'],
        video_origin_path=st.session_state['video_origin_path'],
        video_origin_paths=st.session_state.get('video_origin_paths', []),
        original_subtitle_path=subtitle_paths[0] if subtitle_paths else "",
        original_subtitle_paths=subtitle_paths,
        tts_engine=tts_engine,
        voice_name=voice_name,
        voice_rate=voice_rate,
        voice_pitch=voice_pitch,
        n_threads=config.app.get('n_threads', 4),
        video_aspect=VideoAspect.landscape,
        subtitle_enabled=st.session_state.get('subtitle_enabled', False),
        font_name=st.session_state.get('font_name', 'SourceHanSansCN-Regular.otf'),
        font_size=st.session_state.get('font_size', 24),
        text_fore_color=st.session_state.get('text_fore_color', '#FFFFFF'),
        subtitle_position=st.session_state.get('subtitle_position', 'bottom'),
        custom_position=st.session_state.get('custom_position', 70.0),
        tts_volume=st.session_state.get('tts_volume', 1.0),
        original_volume=st.session_state.get('original_volume', 0.7),
        bgm_volume=st.session_state.get('bgm_volume', 0.3),
        draft_name=(
            draft_name
            if draft_name is not None
            else st.session_state.get('draft_name_input', f"{APP_NAME_EN}_{int(time.time())}")
        )
    )


def _render_jianying_export_status():
    """渲染剪映导出的结果提示。"""
    result = st.session_state.get('jianying_export_result')
    error = st.session_state.get('jianying_export_error')

    if result:
        st.success(tr("Jianying draft exported successfully").format(name=result['draft_name']))
        st.info(tr("Draft saved to").format(path=result['draft_path']))
    elif error:
        st.error(f"{tr('Failed to export Jianying draft')}: {error}")


def _render_jianying_export_dialog():
    """使用弹窗确认剪映草稿名称。"""
    import uuid
    from loguru import logger

    @st.dialog(tr("Export to Jianying Draft"), width="small")
    def jianying_export_dialog():
        jianying_draft_path = config.ui.get("jianying_draft_path", "")
        dialog_title = escape(tr("Jianying export dialog title"))
        dialog_description = escape(tr("Jianying export dialog description"))
        destination_label = escape(tr("Jianying export destination"))
        destination_path = escape(jianying_draft_path or "-")

        st.markdown(
            f"""
            <style>
                .jianying-export-panel {{
                    display: flex;
                    gap: 12px;
                    align-items: flex-start;
                    padding: 14px;
                    margin: 2px 0 18px;
                    border: 1px solid rgba(255, 75, 75, 0.24);
                    border-radius: 8px;
                    background: linear-gradient(135deg, rgba(255, 75, 75, 0.10), rgba(255, 255, 255, 0.96));
                }}
                .jianying-export-icon {{
                    width: 38px;
                    height: 38px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    flex: 0 0 auto;
                    border-radius: 8px;
                    color: #ffffff;
                    background: #ff4b4b;
                    font-size: 20px;
                    line-height: 1;
                }}
                .jianying-export-title {{
                    color: #202534;
                    font-size: 17px;
                    font-weight: 700;
                    line-height: 1.35;
                    margin-bottom: 4px;
                }}
                .jianying-export-description {{
                    color: #5f6575;
                    font-size: 13px;
                    line-height: 1.55;
                }}
                .jianying-export-path {{
                    padding: 10px 12px;
                    margin: 2px 0 16px;
                    border: 1px solid #e4e7ef;
                    border-radius: 8px;
                    background: #f8f9fc;
                    color: #323846;
                    font-size: 13px;
                    line-height: 1.45;
                    word-break: break-all;
                }}
                .jianying-export-path-label {{
                    display: block;
                    color: #7a8192;
                    font-size: 12px;
                    margin-bottom: 4px;
                }}
            </style>
            <div class="jianying-export-panel">
                <div class="jianying-export-icon">📤</div>
                <div>
                    <div class="jianying-export-title">{dialog_title}</div>
                    <div class="jianying-export-description">{dialog_description}</div>
                </div>
            </div>
            <div class="jianying-export-path">
                <span class="jianying-export-path-label">{destination_label}</span>
                {destination_path}
            </div>
            """,
            unsafe_allow_html=True,
        )

        draft_name = st.text_input(
            tr("Jianying draft name"),
            key="draft_name_input",
            placeholder=f"{APP_NAME_EN}_",
        )

        error = st.session_state.get('jianying_export_error')
        if error:
            st.error(f"{tr('Failed to export Jianying draft')}: {error}")

        cancel_col, confirm_col = st.columns(2)
        with cancel_col:
            if st.button(tr("Cancel"), key="cancel_export", use_container_width=True):
                st.session_state['jianying_export_error'] = None
                st.rerun()

        with confirm_col:
            if st.button(tr("Confirm Export"), key="confirm_export", type="primary", use_container_width=True):
                draft_name = (draft_name or "").strip()
                if not draft_name:
                    st.error(tr("Please enter draft name"))
                    return

                # 创建任务ID
                task_id = str(uuid.uuid4())
                st.session_state['task_id'] = task_id

                # 构建参数
                try:
                    params = get_jianying_export_params(draft_name)
                except Exception as e:
                    logger.error(f"构建参数失败: {e}")
                    st.session_state['jianying_export_error'] = f"{tr('Failed to build parameters')}: {e}"
                    st.error(st.session_state['jianying_export_error'])
                    return

                with st.spinner(tr("Exporting to Jianying draft...")):
                    try:
                        from app.services import jianying_task

                        # 调用导出到剪映草稿的任务
                        result = jianying_task.start_export_jianying_draft(task_id, params)

                        # 记录日志
                        logger.info(f"成功导出到剪映草稿: {result['draft_name']}")
                        logger.info(f"草稿已保存到: {result['draft_path']}")

                        # 保存结果到session state
                        st.session_state['jianying_export_result'] = result
                        st.session_state['jianying_export_error'] = None
                        st.rerun()
                    except Exception as e:
                        logger.error(f"导出到剪映草稿失败: {e}")
                        import traceback
                        logger.error(f"错误详情: {traceback.format_exc()}")
                        st.session_state['jianying_export_error'] = str(e)
                        st.session_state['jianying_export_result'] = None
                        st.error(f"{tr('Failed to export Jianying draft')}: {e}")

    jianying_export_dialog()


def render_export_jianying_button():
    """渲染导出到剪映草稿按钮和处理逻辑"""
    import os
    import time
    
    # 初始化session state
    if 'jianying_export_result' not in st.session_state:
        st.session_state['jianying_export_result'] = None
    if 'jianying_export_error' not in st.session_state:
        st.session_state['jianying_export_error'] = None
    
    if st.button(tr("Export to Jianying Draft"), use_container_width=True, type="secondary"):
        config.save_config()
        
        if not st.session_state.get('video_clip_json_path'):
            st.error(tr("Script file cannot be empty"))
            return
        if not st.session_state.get('video_origin_path'):
            st.error(tr("Video file cannot be empty"))
            return
        
        jianying_draft_path = config.ui.get("jianying_draft_path", "")
        if not jianying_draft_path:
            st.error(tr("Please configure Jianying draft folder in basic settings"))
            return
        
        if not os.path.exists(jianying_draft_path):
            st.error(tr("Jianying draft folder does not exist").format(path=jianying_draft_path))
            return
        
        st.session_state['jianying_export_result'] = None
        st.session_state['jianying_export_error'] = None
        st.session_state['draft_name_input'] = f"{APP_NAME_EN}_{int(time.time())}"
        _render_jianying_export_dialog()
    
    _render_jianying_export_status()



def _render_home():
    """Project introduction and navigation; local documents are read on explicit expansion."""
    from webui.utils.file_utils import remembered_expander, remember_widget_values

    prefix = "home"
    st.session_state["_narration_settings_prefix"] = prefix
    remember_widget_values(prefix)

    def open_workspace(page, *, new_project=False):
        remember_widget_values(prefix)
        if new_project:
            st.session_state["narration_manage_new_open"] = True
        if page == "models":
            st.session_state["model_config_return_page"] = "home"
        st.session_state["narration_pending_workspace"] = page
        st.rerun(scope="app")

    st.title(APP_TITLE)
    st.caption(tr("文案驱动的视频制作工作台") + f" · v{config.project_version}")
    st.write(tr("以完整文案为核心，利用可复用素材库组织画面，结合连续配音、字幕对齐、配乐和解释动画，制作视频。"))

    st.subheader(tr("快速开始"))
    actions = (("新建项目", "登记完整文案和视频素材库，开始一个项目。", "narration_create"),
               ("继续制作", "保留当前项目选择，进入制作设置和生成进度。", "narration_progress"),
               ("配置模型", "查看并维护模型接口、用途与参数。", "models"))
    for column, (label, description, page) in zip(st.columns(3), actions):
        with column:
            with st.container(border=True):
                st.write(tr(description))
                if st.button(tr(label), key=prefix + "_go_" + page, width="stretch",
                             type="primary" if page == "narration_create" else "secondary"):
                    open_workspace(page, new_project=page == "narration_create")

    st.divider()
    st.subheader(tr("项目来源与演进"))
    st.write(tr("{project} 基于开源项目 NarratoAI 开发，由本项目维护者独立维护。").format(project=APP_TITLE))
    st.write(tr("NarratoAI 提供影视解说、文案生成、配音、字幕与视频制作基础；本项目围绕固定文案制作，扩展素材库、连续配音与本地字幕对齐、自动 BGM、解释动画、项目管理和阶段继续。"))
    st.caption(tr("感谢 NarratoAI 原作者及贡献者。原版权声明与完整 MIT 许可保留，第三方组件和模型遵守各自许可。"))
    st.link_button(tr("NarratoAI 上游项目"), UPSTREAM_URL)
    st.caption("MIT License · Copyright (c) 2024 linyq")
    with remembered_expander(tr("查看完整 MIT 许可证"), key=prefix + "_license_open") as panel:
        if panel.open:
            try:
                st.code((Path(config.root_dir) / "LICENSE").read_text(encoding="utf-8"), language=None)
            except OSError:
                st.warning(tr("许可证文件无法读取，请查看项目目录下的 LICENSE。"))

    st.subheader(tr("制作流程"))
    st.markdown(tr("**配置模型 → 建立视频素材库 → 登记完整文案 → 推进制作 → 输出视频**"))
    st.caption(tr("音频素材库供自动配乐使用；BGM 和解释动画可按项目需要开启。"))
    video_col, audio_col = st.columns(2)
    if video_col.button(tr("视频素材库"), key=prefix + "_go_library", width="stretch"):
        open_workspace("library")
    if audio_col.button(tr("音频素材库"), key=prefix + "_go_audio_library", width="stretch"):
        open_workspace("audio_library")

    st.subheader(tr("核心能力"))
    features = (("素材分析与复用", "保存视频和音乐的分析结果，后续项目继续使用。"),
                ("文案驱动画面组织", "多个素材库联合检索，在连续旁白下编排多个镜头。"),
                ("连续配音与字幕", "长段配音、手动分块、本地字词对齐和可调整字号的单行字幕。"),
                ("自动配乐", "利用整首音乐的分析资料，按文案主题和情绪选曲。"),
                ("解释动画", "按原文范围自动或手动插入章节、概念与观点动画。"),
                ("项目复用与阶段继续", "管理历史项目，通过副本调整设置，从编号阶段继续制作。"))
    for offset in range(0, len(features), 3):
        for column, (label, description) in zip(st.columns(3), features[offset:offset + 3]):
            with column:
                with st.container(border=True):
                    st.markdown("**" + tr(label) + "**")
                    st.write(tr(description))

    with remembered_expander(tr("使用说明"), key=prefix + "_guide_open") as panel:
        if panel.open:
            st.write(tr("先在模型配置和素材库准备资料，再到项目管理登记文案；制作工作台保存参数、推进生成并查看结果。"))
            try:
                guide = (Path(config.root_dir) / "docs" / "manuscript-video.md").read_text(encoding="utf-8")
                st.download_button(tr("下载详细使用说明"), guide, file_name=APP_NAME_EN + "-guide.md",
                    mime="text/markdown", key=prefix + "_download_guide")
            except OSError:
                st.warning(tr("使用说明文件无法读取，请查看项目 docs 目录。"))
    remember_widget_values(prefix)


def main():
    """主函数"""
    init_log()
    init_global_state()

    if st.session_state.get("_narration_settings_prefix"):
        from webui.utils.file_utils import remember_widget_values
        remember_widget_values()
    destination = st.session_state.pop("narration_pending_workspace", None)
    if destination in WORKSPACE_LABELS:
        st.session_state["workspace_page"] = destination
    elif st.session_state.get("workspace_page") == "narration":
        st.session_state["workspace_page"] = "narration_create"
    current_page = st.session_state.get("workspace_page")
    st.session_state["workspace_page"] = current_page if current_page in WORKSPACE_LABELS else "home"
    st.sidebar.caption(APP_TITLE + f" · v{config.project_version}")
    page = st.sidebar.radio(
        tr("工作区"), options=list(WORKSPACE_LABELS), index=None,
        format_func=lambda value: tr(WORKSPACE_LABELS[value]), key="workspace_page",
    )
    if page == "home":
        _render_home()
        return
    if page == "models":
        from webui.components.model_settings import render_model_workspace
        render_model_workspace(tr)
        return
    if page == "library":
        from webui.components.library_settings import render_library_panel
        render_library_panel(tr)
        return
    if page == "audio_library":
        from webui.components.audio_library_settings import render_audio_library_panel
        render_audio_library_panel(tr)
        return
    if page in {"narration_create", "narration_progress"}:
        from webui.components.narration_settings import render_project_management, render_project_progress
        if page == "narration_create":
            render_project_management(tr)
        else:
            render_project_progress(tr)
        return

    from webui.components.model_settings import seed_model_choices, render_model_choices, selected_models, selected_bindings
    from webui.utils.file_utils import remembered_expander
    from app.services.model_profiles import runtime_configuration
    prefix = "legacy_models"
    st.session_state["_narration_settings_prefix"] = prefix
    seed_model_choices(prefix, bindings=config.ui.get("legacy_model_bindings"))
    with remembered_expander(tr("模型选择"), key=prefix + "_open") as panel:
        if panel.open:
            render_model_choices(tr, prefix, backups=False)
            if st.button(tr("保存本页模型选择"), key=prefix + "_save"):
                config.ui["legacy_model_bindings"] = selected_bindings(prefix)
                config.save_config()
    with runtime_configuration(models=selected_models(prefix, roles=("text", "text_fast", "vision", "asr"))):
        _render_legacy_workspace()


def _render_legacy_workspace():
    from webui.components import basic_settings, video_settings, audio_settings, subtitle_settings, script_settings, system_settings

    # ===== 显式注册 LLM 提供商（最佳实践）=====
    # 在应用启动时立即注册，确保所有 LLM 功能可用
    if 'llm_providers_registered' not in st.session_state:
        try:
            from app.services.llm.providers import register_all_providers
            register_all_providers()
            st.session_state['llm_providers_registered'] = True
            logger.info("✅ LLM 提供商注册成功")
        except Exception as e:
            logger.error(f"❌ LLM 提供商注册失败: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            st.error(tr("LLM initialization failed").format(error=str(e)))
            # 不抛出异常，允许应用继续运行（但 LLM 功能不可用）

    # 检测FFmpeg硬件加速，但只打印一次日志（使用 session_state 持久化）
    if 'hwaccel_logged' not in st.session_state:
        st.session_state['hwaccel_logged'] = False
    
    hwaccel_info = ffmpeg_utils.detect_hardware_acceleration()
    if not st.session_state['hwaccel_logged']:
        if hwaccel_info["available"]:
            logger.info(f"FFmpeg硬件加速检测结果: 可用 | 类型: {hwaccel_info['type']} | 编码器: {hwaccel_info['encoder']} | 独立显卡: {hwaccel_info['is_dedicated_gpu']}")
        else:
            logger.warning(f"FFmpeg硬件加速不可用: {hwaccel_info['message']}, 将使用CPU软件编码")
        st.session_state['hwaccel_logged'] = True

    # 仅初始化基本资源，避免过早地加载依赖PyTorch的资源
    # 检查是否能分解utils.init_resources()为基本资源和高级资源(如依赖PyTorch的资源)
    try:
        utils.init_resources()
    except Exception as e:
        logger.warning(f"资源初始化时出现警告: {e}")

    st.title(APP_TITLE)
    st.write(get_help_text())

    # 首先渲染不依赖PyTorch的UI部分
    # 渲染基础设置面板
    basic_settings.render_basic_settings(tr)

    # 渲染主面板
    panel = st.columns(3)
    with panel[0]:
        script_settings.render_script_panel(tr)
    with panel[1]:
        audio_settings.render_audio_panel(tr)
    with panel[2]:
        video_settings.render_video_panel(tr)
        subtitle_settings.render_subtitle_panel(tr)

    # 放到最后渲染可能使用PyTorch的部分
    # 渲染系统设置面板
    with panel[2]:
        system_settings.render_system_panel(tr)

    # 放到最后渲染生成按钮和处理逻辑
    render_generate_button()
    render_export_jianying_button()


if __name__ == "__main__":
    main()
