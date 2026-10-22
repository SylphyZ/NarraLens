"""Frame-counted FFmpeg cuts, concatenation and the existing final compositor."""

import json
import subprocess
from dataclasses import asdict
from pathlib import Path
from uuid import uuid4


from app.services.library.frame_extractor import execute_ffmpeg as _execute, ffmpeg_binary, ffprobe_binary, inspect_video
from app.services.library.runner import check_cancel
from app.services.library.store import stable_hash, write_json
from app.services.timeline_planner import validate_timeline




def _video_info(path):
    path = Path(path)
    try:
        if not path.is_file() or not path.stat().st_size:
            return {}
        result = subprocess.run([ffprobe_binary(), "-v", "error", "-show_entries",
                                 "stream=codec_type,nb_frames,duration:format=duration:chapter=id",
                                 "-of", "json", str(path)], capture_output=True, timeout=30)
        if result.returncode != 0:
            return {}
        data = json.loads(result.stdout)
        streams = data.get("streams", [])
        video = next(item for item in streams if item.get("codec_type") == "video")
        return {"frames": int(video.get("nb_frames") or 0), "video_duration": float(video.get("duration") or 0),
                "duration": float(data.get("format", {}).get("duration") or 0),
                "audio_duration": float(next((item.get("duration") for item in streams if item.get("codec_type") == "audio"), 0) or 0),
                "video_only": len(streams) == 1 and not data.get("chapters")}
    except (OSError, ValueError, TypeError, StopIteration, subprocess.TimeoutExpired):
        return {}


def _duration_matches(info, duration, fps):
    # Container rounding and frame quantization are normal; average FPS need not be exact.
    return bool(info and duration > 0 and all(abs(info[key] - duration) <= 2 / fps
                                             for key in ("video_duration", "duration")))


def _video_matches(info, frames, fps):
    return bool(info and info["frames"] == frames and info["video_only"]
                and _duration_matches(info, frames / fps, fps))


def render_timeline(segments, shots, assets, audio_path, subtitle_path, directory, options, stop=None, progress=None, *, visual_policy=None):
    from app.services import generate_video
    from app.utils import ffmpeg_utils
    validate_timeline(segments, shots, assets, policy=visual_policy)
    for asset_id in {shot.asset_id for shot in shots}:
        check_cancel(stop)
        asset = next(item for item in assets if item["id"] == asset_id)
        actual = inspect_video(asset["path"], stop)
        if actual.fingerprint != asset["fingerprint"] or any(
                shot.source_stream_index != actual.stream_index for shot in shots if shot.asset_id == asset_id):
            raise ValueError("来源视频已变化，请更新素材库后继续。")
    directory = Path(directory)
    clips_dir = directory / "clips"
    clips_dir.mkdir(exist_ok=True)
    width, height = (1080, 1920) if options.get("aspect") == "portrait" else (1920, 1080)
    fps = 30
    encoder = ffmpeg_utils.get_optimal_ffmpeg_encoder()
    paths = []
    frame_counts = []
    for index, shot in enumerate(shots):
        check_cancel(stop)
        frames = round(shot.timeline_end * fps) - round(shot.timeline_start * fps)
        if frames <= 0:
            continue
        key = stable_hash((asdict(shot), width, height, fps, "render-source-frames-v2"))
        output = clips_dir / f"{index:05d}-{key[:16]}.mp4"
        info = _video_info(output)
        if not _video_matches(info, frames, fps):
            temporary = output.with_suffix(".tmp.mp4")
            absolute_start = shot.source_origin_seconds + shot.source_start
            absolute_end = shot.source_origin_seconds + shot.source_end
            video_filter = (f"trim=start={absolute_start:.9f}:end={absolute_end:.9f},setpts=PTS-STARTPTS,"
                            f"scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,"
                            f"setsar=1,fps={fps},tpad=stop_mode=clone:stop_duration=1")
            basic = [ffmpeg_binary(), "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-copyts"]
            if absolute_start > 0:
                basic += ["-seek_timestamp", "1", "-ss", f"{absolute_start:.9f}"]
            basic += ["-i", shot.video_path, "-map", f"0:{shot.source_stream_index}", "-an", "-sn", "-dn", "-map_chapters", "-1", "-map_metadata", "-1",
                      "-vf", video_filter, "-frames:v", str(frames), "-pix_fmt", "yuv420p"]
            chosen = encoder if encoder in {"h264_amf", "h264_nvenc", "h264_videotoolbox", "libx264"} else "libx264"
            args = ["-c:v", chosen, "-b:v", "6M"] if chosen != "libx264" else ["-c:v", "libx264", "-preset", "fast", "-crf", "21"]
            if info and info["frames"] == frames and abs(info["video_duration"] - frames / fps) <= 2 / fps:
                # Repair chapter-polluted derived clips without reading or re-encoding the original video.
                return_code, ffmpeg_output = _execute(
                    [ffmpeg_binary(), "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-i", str(output),
                     "-map", "0:v:0", "-an", "-sn", "-dn", "-map_chapters", "-1", "-map_metadata", "-1", "-c:v", "copy", str(temporary)],
                    stop, directory=directory, stage=f"clip-{index+1:05d}-chapters")
            else:
                return_code, ffmpeg_output = _execute(basic + args + [str(temporary)], stop, directory=directory, stage=f"clip-{index+1:05d}")
                if return_code != 0 and chosen != "libx264":
                    return_code, ffmpeg_output = _execute(basic + ["-c:v", "libx264", "-preset", "fast", "-crf", "21", str(temporary)],
                                                         stop, directory=directory, stage=f"clip-{index+1:05d}-software")
            if return_code != 0 or not temporary.is_file() or temporary.stat().st_size == 0:
                raise RuntimeError(f"镜头 {index+1} 裁剪失败（FFmpeg 返回码 {return_code}）。\n{ffmpeg_output[-1500:]}")
            if not _video_matches(_video_info(temporary), frames, fps):
                raise RuntimeError(f"镜头 {index+1} 的帧数、时长或章节信息异常，未替换已有缓存；预期 {frames / fps:.3f} 秒。")
            temporary.replace(output)
        paths.append(output)
        frame_counts.append(frames)
        if progress:
            progress((index + 1) / len(shots) * .7, f"裁剪镜头 {index + 1}/{len(shots)}")
    check_cancel(stop)
    if not paths:
        raise ValueError("整段配音不足一帧，无法生成视频。")
    concat_file = directory / "concat.txt"
    concat_file.write_text("\n".join("file '" + path.resolve().as_posix().replace("'", "'\\''") + f"'\nduration {frames / fps:.9f}"
                                     for path, frames in zip(paths, frame_counts)), encoding="utf-8")
    joined = directory / "visual.mp4"
    total_frames = sum(frame_counts)
    manifest_key = stable_hash(("video-only-v2", [str(path) for path in paths], frame_counts))
    manifest = directory / "visual_manifest.json"
    previous = json.loads(manifest.read_text(encoding="utf-8")) if manifest.is_file() else {}
    if previous.get("key") != manifest_key or not _video_matches(_video_info(joined), total_frames, fps):
        temporary = joined.with_suffix(".tmp.mp4")
        from app.services.merger_video import _can_concat_video_copy
        codec = ["-c:v", "copy"] if _can_concat_video_copy([str(path) for path in paths]) else ["-c:v", "libx264", "-preset", "fast", "-crf", "21"]
        return_code, ffmpeg_output = _execute([ffmpeg_binary(), "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0",
                                              "-i", str(concat_file), "-map", "0:v:0", "-an", "-sn", "-dn", "-map_chapters", "-1",
                                              "-map_metadata", "-1", *codec, str(temporary)], stop, directory=directory, stage="concat", timeout=14400)
        if return_code != 0 or not temporary.is_file() or temporary.stat().st_size == 0:
            raise RuntimeError(f"画面拼接失败（FFmpeg 返回码 {return_code}）。\n{ffmpeg_output[-1500:]}")
        if not _video_matches(_video_info(temporary), total_frames, fps):
            raise RuntimeError(f"画面拼接帧数或时长与时间轴不符，未进入最终合成；预期 {total_frames / fps:.3f} 秒。")
        temporary.replace(joined)
        write_json(manifest, {"key": manifest_key})
    check_cancel(stop)
    final = directory / "final.mp4"
    temporary = directory / "final.tmp.mp4"
    merge_options = {
        "target_duration": float(segments[-1].audio_end),
        "keep_original_audio": False, "voice_volume": options.get("voice_volume", 1.0),
        "bgm_volume": options.get("bgm_volume", .15), "subtitle_enabled": bool(options.get("subtitles", True)),
        "bgm_prepared": bool(options.get("bgm_prepared", False)),
        "subtitle_font": options.get("font", ""), "subtitle_font_size": options.get("font_size", 40),
        "subtitle_single_line": bool(options.get("subtitle_single_line", False)),
        "subtitle_width_ratio": options.get("subtitle_width_ratio", .9),
        "subtitle_color": "#FFFFFF", "subtitle_position": "bottom", "threads": 4,
        "subtitle_mask_enabled": bool(options.get("mask_source_subtitles", False)),
    }
    temp_files = []
    try:
        if options.get("bgm_prepared") and not Path(options.get("bgm_path") or "").is_file():
            raise ValueError("已编排的音乐音轨不存在，请重新制作配乐。")
        command, temp_files, video_duration = generate_video._build_ffmpeg_merge_command(
            str(joined), str(audio_path), str(temporary), subtitle_path, options.get("bgm_path") or None, merge_options)
        # The shared command builder supplies fonts, masks and subtitle-filter fallbacks.
        while "-filter_complex" in command:
            filter_index = command.index("-filter_complex")
            filter_script = directory / "ffmpeg-diagnostics" / f"filter-{uuid4().hex}.txt"
            filter_script.parent.mkdir(parents=True, exist_ok=True)
            filter_script.write_text(command[filter_index + 1], encoding="utf-8", newline="\n")
            command[filter_index:filter_index + 2] = ["-filter_complex_script", str(filter_script)]
        return_code, ffmpeg_output = _execute(command, stop, directory=directory, stage="final", timeout=21600, duration=video_duration,
                                              progress=lambda value: progress(.7 + value * .3, "合成字幕、配音和背景音乐") if progress else None)
        if return_code != 0 and "-c:v" in command and command[command.index("-c:v") + 1] not in {"copy", "libx264"}:
            begin = command.index("-c:v")
            end = command.index("-map", begin + 2)
            command = command[:begin] + generate_video._build_video_encoder_args("libx264", 4) + command[end:]
            return_code, ffmpeg_output = _execute(command, stop, directory=directory, stage="final-software", timeout=21600, duration=video_duration,
                                                  progress=lambda value: progress(.7 + value * .3, "使用软件编码合成") if progress else None)
        if return_code != 0:
            raise RuntimeError(f"最终合成失败（FFmpeg 返回码 {return_code}）。\n{ffmpeg_output[-1500:]}")
    finally:
        for filename in temp_files:
            path = Path(filename)
            if path.resolve().is_relative_to(directory.resolve()):
                path.unlink(missing_ok=True)
    check_cancel(stop)
    if not temporary.is_file() or temporary.stat().st_size == 0:
        raise RuntimeError("未生成有效成片。")
    info = _video_info(temporary)
    target_duration = merge_options["target_duration"]
    if not _duration_matches(info, target_duration, fps):
        raise RuntimeError("成片时长与配音时间轴不符，未覆盖已有成片；请查看合成日志。")
    audio_duration = info.get("audio_duration", 0)
    if not 0 < audio_duration < float("inf") or abs(audio_duration - target_duration) > max(.1, 2 / fps):
        raise RuntimeError(f"成片音轨时长{audio_duration:.3f}秒，与配音目标{target_duration:.3f}秒不符；"
                           "已有成片、原配音和有效镜头保留，请查看最终合成日志。")
    temporary.replace(final)
    return str(final)
