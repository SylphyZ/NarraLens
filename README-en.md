# NarraLens · 叙镜

> Turn your manuscript into a video.

**Manuscript-driven video production · v2.6.0**

[简体中文](README.md) · [MIT license](LICENSE)

## About

NarraLens creates videos around a complete manuscript and local media libraries. It retrieves footage, arranges shots under continuous narration, aligns subtitles, and optionally adds background music and explanatory animations.

It supports long-form film analysis and commentary. A manuscript does not need to follow each episode closely: projects can search multiple video libraries. Review the final video and matching details against your source material.

## Relationship to NarratoAI

NarraLens is an independently maintained derivative of [NarratoAI](https://github.com/linyqh/NarratoAI).

NarratoAI provides the foundations for LLM-assisted film commentary, script generation, speech, subtitles, and video processing. NarraLens builds on those implementations and extends workflows centered on an existing complete manuscript.

| Origin | Capabilities |
|---|---|
| NarratoAI foundations | Film commentary and video production, model interfaces, existing speech services, subtitle and media processing |
| NarraLens extensions | Video and audio libraries, manuscript-driven retrieval and shot planning, project management, long-form speech and local alignment, automatic BGM, explanatory animations, centralized model configuration, and stage resume |

Thanks to the original NarratoAI authors and contributors. Their copyright notice and MIT license are retained. This derivative and its extensions are maintained independently; the upstream links identify the source project.

[Upstream project](https://github.com/linyqh/NarratoAI) · [Upstream MIT license](https://github.com/linyqh/NarratoAI/blob/main/LICENSE)

## Features

- **Reusable media analysis:** register footage and music, save analysis, and reuse successful results in later projects.
- **Complete manuscript input:** paste text or upload TXT/Markdown while retaining the source document.
- **Shot planning:** retrieve from multiple libraries and arrange several shots under continuous speech.
- **Long-form narration:** start with paragraph blocks and manually merge or split them. The current default pause between blocks is three seconds.
- **Local alignment:** a separate Qwen3 ForcedAligner environment provides word timestamps. Text-based timing estimates remain available when local alignment is disabled.
- **Single-line subtitles:** adjustable font size, measured text width, and available speech time guide subtitle changes.
- **Automatic BGM:** use whole-track analysis to choose one track or change music by manuscript themes and emotions.
- **Explanatory animations:** automatic, manual, or mixed placement, anchored to original text ranges, with timing and audio previews.
- **Project management:** creation, inspection, reuse, and deletion are separate from production settings.
- **Stage resume:** use the general continue action or numbered stages 1–5, preserving matching saved results.
- **Model configuration:** centralized endpoints, credentials, parameters, roles, and independent text/vision backups.

## Workspaces

Version 2.6.0 provides seven workspaces:

| Workspace | Purpose |
|---|---|
| Home | Project introduction, NarratoAI origins, license, workflow, and quick entry points |
| Project Management | Register manuscripts and libraries; inspect, reuse, and delete projects |
| Production Workspace | Configure narration, music, models, output, and animations; generate and review |
| Video Library | Register local videos and subtitles; build and resume analysis |
| Audio Library | Register, preview, and analyze music for BGM |
| Model Configuration | Inspect saved models and edit service definitions and parameters |
| Video Production | Use the inherited NarratoAI production workflow |

A new session starts on Home. Existing sessions and explicit workspace navigation keep their selection. Home exposes the upstream relationship, the full local MIT license, and the production workflow.

## Runtime and models

Prepare Python 3.12, FFmpeg, and the application dependencies. In a prepared environment, run these commands from the source directory:

```powershell
python -m pip install -r requirements.txt
python -m streamlit run webui.py --server.address=127.0.0.1 --server.port=8501
```

Open `http://127.0.0.1:8501`. The repository does not include local Conda environments, model weights, source media, or generated outputs. Local alignment and explanatory animations require their own model/runtime and Node/Remotion dependencies.

This repository starts its independent commit history at NarraLens v2.5.0. Upstream attribution and the license are retained; implementation documents remain on the maintainer's computer.

Media, projects, and generated outputs are managed locally. Text, vision, audio analysis, and speech services may call external APIs according to the selected configuration. Analysis sends the relevant manuscript, subtitles, images, or full music track to that service.

Configure services in Model Configuration. Filled fields do not verify connectivity or input capabilities; use the explicit checks and supported workflow to verify them.

Manuscript projects retain MiMo/Edge TTS support. The inherited Video Production workspace keeps its existing speech engines. Text and vision backups can use independent endpoints, credentials, and parameters.

### Subtitle timestamp model

Long-form narration in the manuscript workflow uses [Qwen/Qwen3-ForcedAligner-0.6B-hf](https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B-hf) for local forced alignment. It takes synthesized speech and the corresponding manuscript, predicts spoken character/word boundaries, and maps them to subtitle timestamps. Chinese alignment primarily uses characters. The selected text/subtitle model plans semantic segmentation.

- An independent worker loads the model only when unfinished blocks need alignment. Successful timestamp caches are reused; exiting the worker releases GPU resources.
- Speech blocks longer than five minutes are located and split into alignment windows while retaining the existing speech audio.
- Prepare a separate PyTorch/Transformers environment and local model weights. The current GPU worker uses BF16 and the cuda device interface; AMD GPUs require a working ROCm environment.
- Local alignment is enabled by default. Turning it off keeps subtitles and estimates timing from audio duration and text weights, which can differ from actual speech.

Explanatory animations use separately prepared Node, Remotion, and browser dependencies.

## Production workflow

1. **Configure models:** maintain service definitions and select them by role in each workspace.
2. **Build video libraries:** scan video/subtitle directories, review matches, register, and analyze. ASS/SSA and SRT are supported. Select the intended subtitle file/style when several translations are present.
3. **Prepare optional audio libraries:** register music and analyze whole tracks through an audio-capable service, or choose music manually.
4. **Register a project:** paste or upload the full manuscript in Project Management and select one or more video libraries.
5. **Configure and produce:** in Production Workspace, choose speech, blocks and pauses, model roles, subtitle style, output, BGM, and animation placement. Save and generate.
6. **Review and adjust:** inspect the video, matching details, subtitle/animation plans, and project diagnostics. Apply changes through the available copy actions when earlier output exists.

Video and music originals normally remain at their registered paths. Libraries save analysis and derived data. Successful analysis can be reused, and interrupted runs can continue.


## Storage

Paths below are relative to the existing code directory:

| Location | Contents |
|---|---|
| config.toml | Application and model definitions, including service credentials |
| storage/libraries/ | Video registrations, subtitle copies, frames, and analysis |
| storage/audio_libraries/ | Music registrations, analysis, and derived data |
| storage/narration_tasks/project-ID/ | Manuscript input, settings, speech, subtitles, plans, caches, and diagnostics |
| storage/narration_tasks/project-ID/final.mp4 | Manuscript project video |
| storage/narration_tasks/project-ID/motion/ | Animation plans, assets, and diagnostics |
| storage/tasks/ | Inherited Video Production task output |

Moving or deleting originals affects later reads. Update registrations or use the supported rebind actions.

## Resume, reuse, and delete

Use the general continue action or numbered stages 1–5. Each stage validates its prerequisites and can reuse matching saved output. Use the workspace copy actions to apply changes to existing speech, subtitles, animations, or music.

After upgrading to 2.6, resume old timelines from **3. Find footage** to apply the new selection policy, reusing narration, captions, and alignment. The previous video and timeline are retained under the project's `history/visual-*` directory. Disabling visual relevance confirmation still requires a vision model for the independent credits/usability check.

Delete a manuscript project through **Project Management → select project → Manage Project → inspect files and deletion scope → confirm → permanently delete**. This removes the current project and its output while retaining original media and external manuscript files. Associated running work must stop first.

Code archives normally exclude storage, media, environments, and weights. Back up project data separately when preserving audio, videos, and analysis.

## Version and documentation

Version 2.6.0 prohibits reuse of the same source shot within one output using content fingerprints and detected cut boundaries, expands retrieval when footage runs short, and checks actual source ranges to exclude credits, mixed footage, and uncertain results. Range checks are cached. Animation cards, arrows, progress, and subtitle space use separate layout regions. Visual near-duplicates across episodes or differently encoded files are not detected; sampled-frame checks also depend on model accuracy.

Version 2.5.4 reports animation format issues together and uses the previous JSON to repair failed fields while retaining valid content and original-text bindings. Correction messages are recorded and included in input budgets.

Version 2.5.3 allows chapter title animations from 2 seconds while flow and comparison animations require 4 seconds. The Production Workspace shows actual narration times and checks all manual nodes using the same duration rules as automatic planning.

Version 2.5.2 fixes animation browser paths after directory moves, stores project-local browser paths relative to the Remotion directory, identifies missing runtime files, and removes unused .github configuration.

Version 2.5.1 documents the local subtitle timestamp model and synchronizes the independent repository release metadata.

Version 2.5.0 adds Home, the bilingual NarraLens identity, new README files, and explicit upstream/license information. Prior project creation fixes and the persistent manual animation Add Node button remain in place.

Implementation notes distinguish source changes, static checks, and functional acceptance. Some changes have not yet been validated through the full UI or generation workflow.


## License and acknowledgements

NarraLens retains NarratoAI's MIT license. See the complete local [LICENSE](LICENSE) and [upstream copy](https://github.com/linyqh/NarratoAI/blob/main/LICENSE). The original notice is retained:

```text
Copyright (c) 2024 linyq
```

Retain the original copyright notice and full license notice, including the disclaimer, when distributing inherited software or substantial portions. Attribution and links in Home or this README accompany those notices; distribution packages must still include the required text. [MIT standard text](https://opensource.org/license/mit)

Third-party components, models, renderers, and API services follow their own licenses or terms. The new display name does not change those conditions.

Thanks to NarratoAI and its contributors. NarraLens versions and extensions are maintained independently. A dedicated maintenance channel has not yet been supplied; upstream links identify the source and its documentation.
