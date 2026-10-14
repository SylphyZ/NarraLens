"""Isolated HF forced-alignment worker. No NarratoAI or cloud-provider imports."""

import argparse
import contextlib
import json
import os
from pathlib import Path
import sys
import time
import traceback
import wave

_PROTOCOL = sys.stdout


def emit(event, **values):
    _PROTOCOL.write("\n" + json.dumps({"event": event, **values}, ensure_ascii=False, allow_nan=False) + "\n")
    _PROTOCOL.flush()


def run(job_file):
    job = json.loads(Path(job_file).read_text(encoding="utf-8-sig"))
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1",
                      HF_HOME=str(Path(job["model_path"]).parent / ".hf-cache"))
    # ROCm's native cache cannot reliably use a Windows Unicode user-profile path.
    cache_root = Path(__file__).resolve().parents[3] / ".conda" / "miopen-cache"
    for variable, folder in (("MIOPEN_CUSTOM_CACHE_DIR", "kernels"), ("MIOPEN_USER_DB_PATH", "db")):
        target = cache_root / folder
        target.mkdir(parents=True, exist_ok=True)
        os.environ[variable] = target.as_posix()
    with contextlib.redirect_stdout(sys.stderr):
        import numpy as np
        import torch
        import transformers
        from transformers import AutoModelForTokenClassification, AutoProcessor
        from transformers.models.qwen3_asr.processing_qwen3_asr import _is_kept_char

        if not torch.cuda.is_available():
            raise RuntimeError("对齐环境没有识别到GPU，请检查ROCm环境。")
        if not torch.cuda.is_bf16_supported(including_emulation=False):
            raise RuntimeError("当前GPU不支持本次配置的BF16对齐。")
        started = time.perf_counter()
        processor = AutoProcessor.from_pretrained(job["model_path"], local_files_only=True, token=False,
                                                 trust_remote_code=False)
        model = AutoModelForTokenClassification.from_pretrained(
            job["model_path"], local_files_only=True, token=False, trust_remote_code=False,
            dtype=torch.bfloat16, attn_implementation="sdpa",
        ).to("cuda:0").eval()
        emit("ready", gpu=torch.cuda.get_device_name(0), dtype=str(model.dtype),
             load_seconds=round(time.perf_counter() - started, 3),
             torch=str(torch.__version__), transformers=transformers.__version__)
        for item in job["jobs"]:
            emit("start", id=item["id"])
            with wave.open(item["input_path"], "rb") as source:
                if (source.getnchannels(), source.getsampwidth(), source.getframerate()) != (1, 2, 16000):
                    raise ValueError("对齐输入必须是16kHz单声道PCM16 WAV。")
                frames = source.getnframes()
                if not 0 < frames <= 16000 * 300:
                    raise ValueError(f"配音块{item['id']}时长超过5分钟或为空，请降低配音块字数后新建任务。")
                audio = np.frombuffer(source.readframes(frames), dtype="<i2").astype(np.float32) / 32768.0
            started = time.perf_counter()
            torch.cuda.reset_peak_memory_stats()
            inputs, word_lists = processor.prepare_forced_aligner_inputs(
                audio=audio, transcript=item["text"], language=job.get("language", "Chinese"),
                processor_kwargs={"audio_kwargs": {"sampling_rate": 16000, "truncation": False}},
            )
            tokens = int(inputs["input_ids"].shape[-1])
            limit = int(model.config.text_config.max_position_embeddings)
            if tokens > limit:
                raise ValueError(f"配音块{item['id']}编码长度{tokens}超过{limit}，请降低配音块字数后新建任务。")
            if not word_lists[0]:
                raise ValueError(f"配音块{item['id']}没有可对齐文字。")
            positions = [i for i, char in enumerate(item["text"]) if _is_kept_char(char)]
            canonical = "".join(item["text"][i] for i in positions)
            if canonical != "".join(word_lists[0]):
                raise ValueError("处理器清洗文字无法按原文顺序映射。")
            inputs = inputs.to(model.device, model.dtype)
            with torch.inference_mode():
                outputs = model(**inputs)
            words = processor.decode_forced_alignment(
                logits=outputs.logits, input_ids=inputs["input_ids"], word_lists=word_lists,
                timestamp_token_id=model.config.timestamp_token_id,
                timestamp_segment_time=model.config.timestamp_segment_time,
            )[0]
            consumed = 0
            for word in words:
                word["text_start"] = positions[consumed]
                consumed += len(word["text"])
                word["text_end"] = positions[consumed - 1] + 1
            torch.cuda.synchronize()
            stats = {"duration": frames / 16000, "encoded_tokens": tokens,
                     "inference_seconds": round(time.perf_counter() - started, 3),
                     "peak_vram_gib": round(torch.cuda.max_memory_allocated() / 1024**3, 3),
                     "timestamp_grid_ms": model.config.timestamp_segment_time,
                     "zero_duration_words": sum(w["end_time"] <= w["start_time"] for w in words)}
            emit("result", id=item["id"], words=words, stats=stats)
            del outputs, inputs, audio
        emit("complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", required=True)
    args = parser.parse_args()
    try:
        run(args.job)
    except Exception as error:
        emit("error", message=f"{type(error).__name__}: {error}")
        traceback.print_exc(file=sys.stderr)
        raise SystemExit(1)
