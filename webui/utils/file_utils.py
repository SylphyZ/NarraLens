import os
import glob
import time
import platform
import shutil
import base64
import subprocess
import threading
import json
import sys
from copy import deepcopy
from uuid import uuid4
from loguru import logger
from app.utils import utils

_folder_dialog_lock = threading.Lock()


def probe_file(path, timeout=8):
    """Explicit read-only probe in a disposable process, including blocked UNC paths."""
    script = """
import errno, json, os, stat, sys
path = json.loads(sys.stdin.read())
try:
    info = os.stat(path)
    if not stat.S_ISREG(info.st_mode):
        result = {"state": "missing", "reason": "路径不是普通文件"}
    else:
        with open(path, "rb") as stream:
            stream.read(1)
        result = {"state": "available", "reason": ""}
except OSError as exc:
    code = getattr(exc, "winerror", None)
    if isinstance(exc, PermissionError):
        state = "denied"
    elif code in (53, 64, 67, 121, 1203, 1231, 1232) or exc.errno in (errno.ENETUNREACH, errno.EHOSTUNREACH, errno.ETIMEDOUT):
        state = "unreachable"
    elif isinstance(exc, FileNotFoundError):
        state = "missing"
    else:
        state = "unconfirmed"
    result = {"state": state, "reason": str(exc)}
print(json.dumps(result, ensure_ascii=True))
"""
    try:
        result = subprocess.run(
            [sys.executable, "-I", "-S", "-c", script], input=json.dumps(str(path)),
            capture_output=True, text=True, encoding="utf-8", timeout=timeout,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        if result.returncode:
            return {"state": "unconfirmed", "reason": result.stderr[-400:]}
        return json.loads(result.stdout)
    except subprocess.TimeoutExpired:
        # subprocess.run kills and reaps the child; there is no abandoned probe thread.
        return {"state": "timeout", "reason": f"超过 {timeout} 秒，结果未确认"}
    except (OSError, ValueError) as exc:
        return {"state": "unconfirmed", "reason": str(exc)}


def remembered_widget_value(key, default=None):
    """Restore a page's widget default after Streamlit removes inactive widgets."""
    import streamlit as st
    return st.session_state.get(key, st.session_state.get("narration_widget_memory", {}).get(key, default))


def remember_widget_values(prefix=None):
    """Capture settings before a full or fragment rerun removes hidden widgets."""
    import streamlit as st
    prefix = prefix or st.session_state.get("_narration_settings_prefix")
    if not prefix:
        return
    memory = st.session_state.setdefault("narration_widget_memory", {})
    global_keys = ("narration_resume_stages_", "narration_motion_plan_", "narration_motion_disabled_",
                   "narration_motion_preview_", "narration_bgm_plan_open_", "narration_original_open_",
                   "narration_results_open_", "narration_matches_open_", "narration_diag_open_",
                   "narration_layout_open_", "narration_model_log_open_", "resume_current_models_", "fork_keep_audio_", "weak_shots_")
    for key in list(st.session_state):
        owned = key.startswith(prefix + "_") or key.startswith(global_keys) or key == "narration_task_manager_open"
        owned |= prefix == "narration_create" and key == "narration_library_choice_multi"
        if owned and not key.endswith(("_upload", "_browse", "_preview")) and "_download_" not in key:
            memory[key] = deepcopy(st.session_state[key])


def remembered_expander(label, *, key, default=False):
    """Keep the expander identity fixed while restoring its saved open state."""
    import streamlit as st
    memory = st.session_state.setdefault("narration_widget_memory", {})
    if key not in st.session_state and (key in memory or default):
        st.session_state[key] = bool(memory.get(key, default))
    def remember():
        memory[key] = bool(st.session_state.get(key, False))
    return st.expander(label, expanded=False, key=key, on_change=remember)


def folder_input(label, *, key, value="", help=None, disabled=False, tr=lambda text: text):
    """Editable directory with a native Windows picker on the WebUI host."""
    import streamlit as st

    def browse():
        error_key = key + "_browse_error"
        st.session_state.pop(error_key, None)
        if os.name != "nt":
            st.session_state[error_key] = tr("文件夹选择按钮仅支持 Windows，请手动输入路径。")
            return
        if not _folder_dialog_lock.acquire(blocking=False):
            st.session_state[error_key] = tr("已有文件夹选择窗口，请先完成或取消。")
            return
        try:
            initial = os.path.expanduser(str(st.session_state.get(key, value)).strip().strip('"'))
            while initial and not os.path.isdir(initial):
                parent = os.path.dirname(initial)
                if parent == initial:
                    break
                initial = parent
            # Pass user paths through the environment, never interpolate them as shell code.
            script = """
$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false)
Add-Type -AssemblyName System.Windows.Forms
$dialog = New-Object System.Windows.Forms.FolderBrowserDialog
$dialog.Description = $env:NARRATO_PICKER_TITLE
$dialog.ShowNewFolderButton = $true
if ($dialog.PSObject.Properties.Name -contains 'AutoUpgradeEnabled') { $dialog.AutoUpgradeEnabled = $true }
if ($env:NARRATO_PICKER_INITIAL) { $dialog.SelectedPath = $env:NARRATO_PICKER_INITIAL }
$owner = New-Object System.Windows.Forms.Form
$owner.TopMost = $true
$owner.ShowInTaskbar = $false
$owner.Opacity = 0
try {
    $owner.Show()
    if ($dialog.ShowDialog($owner) -eq [System.Windows.Forms.DialogResult]::OK) {
        [Console]::Write($dialog.SelectedPath)
    }
} finally { $dialog.Dispose(); $owner.Dispose() }
"""
            result = subprocess.run(
                ["powershell.exe", "-NoProfile", "-STA", "-WindowStyle", "Hidden", "-EncodedCommand",
                 base64.b64encode(script.encode("utf-16-le")).decode("ascii")],
                env={**os.environ, "NARRATO_PICKER_TITLE": label, "NARRATO_PICKER_INITIAL": initial},
                capture_output=True, timeout=600, creationflags=subprocess.CREATE_NO_WINDOW,
            )
            if result.returncode:
                raise OSError(result.stderr.decode("utf-8", errors="replace")[:400])
            selected = result.stdout.decode("utf-8-sig").strip()
            if selected:
                if not os.path.isdir(selected):
                    raise ValueError(tr("选择的文件夹不存在。"))
                st.session_state[key] = selected
        except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
            st.session_state[error_key] = tr("文件夹选择失败，可手动输入：") + str(exc)
        finally:
            _folder_dialog_lock.release()

    if key not in st.session_state:
        st.session_state[key] = str(value or "")
    text_col, button_col = st.columns([5, 1], vertical_alignment="bottom")
    with text_col:
        result = st.text_input(label, value=None, key=key, help=help, disabled=disabled)
    with button_col:
        st.button(tr("选择文件夹"), key=key + "_browse", on_click=browse, disabled=disabled,
                  help=tr("在运行本应用的电脑上打开 Windows 文件夹选择窗口。"), width="stretch")
    if st.session_state.get(key + "_browse_error"):
        st.warning(st.session_state[key + "_browse_error"])
    return result or ""

def open_task_folder(root_dir, task_id):
    """打开任务文件夹
    Args:
        root_dir: 项目根目录
        task_id: 任务ID
    """
    try:
        sys = platform.system()
        path = os.path.join(root_dir, "storage", "tasks", task_id)
        if os.path.exists(path):
            if sys == 'Windows':
                os.system(f"start {path}")
            if sys == 'Darwin':
                os.system(f"open {path}")
            if sys == 'Linux':
                os.system(f"xdg-open {path}")
    except Exception as e:
        logger.error(f"打开任务文件夹失败: {e}")

def cleanup_temp_files(temp_dir, max_age=3600):
    """清理临时文件
    Args:
        temp_dir: 临时文件目录
        max_age: 文件最大保存时间(秒)
    """
    if os.path.exists(temp_dir):
        for file in os.listdir(temp_dir):
            file_path = os.path.join(temp_dir, file)
            try:
                if os.path.getctime(file_path) < time.time() - max_age:
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                    logger.debug(f"已清理临时文件: {file_path}")
            except Exception as e:
                logger.error(f"清理临时文件失败: {file_path}, 错误: {e}")

def get_file_list(directory, file_types=None, sort_by='ctime', reverse=True):
    """获取指定目录下的文件列表
    Args:
        directory: 目录路径
        file_types: 文件类型列表，如 ['.mp4', '.mov']
        sort_by: 排序方式，支持 'ctime'(创建时间), 'mtime'(修改时间), 'size'(文件大小), 'name'(文件名)
        reverse: 是否倒序排序
    Returns:
        list: 文件信息列表
    """
    if not os.path.exists(directory):
        return []

    files = []
    if file_types:
        for file_type in file_types:
            files.extend(glob.glob(os.path.join(directory, f"*{file_type}")))
    else:
        files = glob.glob(os.path.join(directory, "*"))

    file_list = []
    for file_path in files:
        try:
            file_stat = os.stat(file_path)
            file_info = {
                "name": os.path.basename(file_path),
                "path": file_path,
                "size": file_stat.st_size,
                "ctime": file_stat.st_ctime,
                "mtime": file_stat.st_mtime
            }
            file_list.append(file_info)
        except Exception as e:
            logger.error(f"获取文件信息失败: {file_path}, 错误: {e}")

    # 排序
    if sort_by in ['ctime', 'mtime', 'size', 'name']:
        file_list.sort(key=lambda x: x.get(sort_by, ''), reverse=reverse)

    return file_list

def save_uploaded_file(uploaded_file, save_dir, allowed_types=None):
    """保存上传的文件
    Args:
        uploaded_file: StreamlitUploadedFile对象
        save_dir: 保存目录
        allowed_types: 允许的文件类型列表，如 ['.mp4', '.mov']
    Returns:
        str: 保存后的文件路径，失败返回None
    """
    try:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        file_name, file_extension = os.path.splitext(uploaded_file.name)

        # 检查文件类型
        if allowed_types and file_extension.lower() not in allowed_types:
            logger.error(f"不支持的文件类型: {file_extension}")
            return None

        # 如果文件已存在，添加时间戳
        save_path = os.path.join(save_dir, uploaded_file.name)
        if os.path.exists(save_path):
            timestamp = time.strftime("%Y%m%d%H%M%S")
            new_file_name = f"{file_name}_{timestamp}{file_extension}"
            save_path = os.path.join(save_dir, new_file_name)

        # 保存文件
        with open(save_path, "wb") as f:
            f.write(uploaded_file.read())

        logger.info(f"文件保存成功: {save_path}")
        return save_path

    except Exception as e:
        logger.error(f"保存上传文件失败: {e}")
        return None

def create_temp_file(prefix='tmp', suffix='', directory=None):
    """创建临时文件
    Args:
        prefix: 文件名前缀
        suffix: 文件扩展名
        directory: 临时文件目录，默认使用系统临时目录
    Returns:
        str: 临时文件路径
    """
    try:
        if directory is None:
            directory = utils.storage_dir("temp", create=True)

        if not os.path.exists(directory):
            os.makedirs(directory)

        temp_file = os.path.join(directory, f"{prefix}-{str(uuid4())}{suffix}")
        return temp_file

    except Exception as e:
        logger.error(f"创建临时文件失败: {e}")
        return None

def get_file_size(file_path, format='MB'):
    """获取文件大小
    Args:
        file_path: 文件路径
        format: 返回格式，支持 'B', 'KB', 'MB', 'GB'
    Returns:
        float: 文件大小
    """
    try:
        size_bytes = os.path.getsize(file_path)

        if format.upper() == 'B':
            return size_bytes
        elif format.upper() == 'KB':
            return size_bytes / 1024
        elif format.upper() == 'MB':
            return size_bytes / (1024 * 1024)
        elif format.upper() == 'GB':
            return size_bytes / (1024 * 1024 * 1024)
        else:
            return size_bytes

    except Exception as e:
        logger.error(f"获取文件大小失败: {file_path}, 错误: {e}")
        return 0

def ensure_directory(directory):
    """确保目录存在，如果不存在则创建
    Args:
        directory: 目录路径
    Returns:
        bool: 是否成功
    """
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
        return True
    except Exception as e:
        logger.error(f"创建目录失败: {directory}, 错误: {e}")
        return False

def create_zip(files: list, zip_path: str, base_dir: str = None, folder_name: str = "demo") -> bool:
    """
    创建zip文件
    Args:
        files: 要打包的文件列表
        zip_path: zip文件保存路径
        base_dir: 基础目录，用于保持目录结构
        folder_name: zip解压后的文件夹名称，默认为frames
    Returns:
        bool: 是否成功
    """
    try:
        import zipfile

        # 确保目标目录存在
        os.makedirs(os.path.dirname(zip_path), exist_ok=True)

        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file in files:
                if not os.path.exists(file):
                    logger.warning(f"文件不存在，跳过: {file}")
                    continue

                # 计算文件在zip中的路径，添加folder_name作为前缀目录
                if base_dir:
                    arcname = os.path.join(folder_name, os.path.relpath(file, base_dir))
                else:
                    arcname = os.path.join(folder_name, os.path.basename(file))

                try:
                    zipf.write(file, arcname)
                except Exception as e:
                    logger.error(f"添加文件到zip失败: {file}, 错误: {e}")
                    continue

        return True

    except Exception as e:
        logger.error(f"创建zip文件失败: {e}")
        return False
