# NarraLens · 叙镜

> Turn your manuscript into a video.

**Manuscript-driven video production · v2.9.1**

[简体中文](README.md) · User guide · [MIT license](LICENSE)

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
- **Meaning-based shot planning:** retrieve actual shots from multiple libraries, review their relationship to the current manuscript span, and arrange them under continuous speech. Missing coverage is reported.
- **Long-form narration:** start with paragraph blocks and manually merge or split them. The current default pause between blocks is three seconds.
- **Local alignment:** a separate Qwen3 ForcedAligner environment provides word timestamps. Text-based timing estimates remain available when local alignment is disabled.
- **Single-line subtitles:** adjustable font size, measured text width, and available speech time guide subtitle changes.
- **Automatic BGM:** use whole-track analysis to choose one track or change music by manuscript themes and emotions.
- **Explanatory animations:** automatic, manual, or mixed placement, anchored to original text ranges, with timing and audio previews.
- **Project management:** creation, inspection, reuse, and deletion are separate from production settings.
- **Stage resume:** use the general continue action or numbered stages 1–5, preserving matching saved results.
- **Model configuration:** centralized endpoints, credentials, parameters, roles, and independent text/vision backups.

## Workspaces

Version 2.9.1 provides seven workspaces:

| Workspace | Purpose |
|---|---|
| Home | Project introduction, NarratoAI origins, license, workflow, and quick entry points |
| Project Management | Register manuscripts and libraries; inspect, reuse, and delete projects |
| Production Workspace | Configure narration, music, models, output, and animations; generate and review |
| Video Library | Register local videos and subtitles; build and resume analysis |
| Audio Library | Register, preview, and analyze music for BGM |
| Model Configuration | Inspect saved models and edit service definitions and parameters |
| Video Production | Use the inherited NarratoAI production workflow |

A new session starts on Home. Existing sessions and explicit workspace navigation keep their selection. Home exposes the upstream relationship, the full local MIT license, and a downloadable guide.

## Runtime and models

The current Windows setup uses this layout; adapt paths to your own installation:

```text
workspace/
├─ start.bat
├─ .conda/
│  ├─ narratoai/              Main application environment
│  └─ narratoai-align/        Local alignment environment
├─ models/
│  └─ Qwen3-ForcedAligner-0.6B-hf/
└─ NarraLens/                Actual code directory
   ├─ webui.py
   ├─ config.toml
   └─ storage/
```

The actual code directory is NarraLens, and the old directory entry has been removed. Video and audio library paths now use the new directory while retaining material IDs, fingerprints, subtitle/frame caches, and existing analyses. Historical generation projects no longer receive old-path compatibility. The launcher uses NarraLens; existing environments and the internal package name are retained. See the library migration and old directory removal notes.

In an already prepared workspace, run `start.bat` from the parent directory and open `http://127.0.0.1:8501`. Keep the launch window open. Press Ctrl+C to stop.

A new machine requires the application dependencies, FFmpeg, and the selected model services. The existing alignment deployment notes describe the current machine and need adaptation for other hardware.

Media, projects, and generated outputs are managed locally. Text, vision, audio analysis, and speech services may call external APIs according to the selected configuration. Analysis sends the relevant manuscript, subtitles, images, or full music track to that service.

Configure services in Model Configuration. Filled fields do not verify connectivity or input capabilities; use the explicit checks and supported workflow to verify them.

Manuscript projects can select registered MiMo, Edge, Doubao, Tencent, Qwen3 TTS, Azure, SoulVoice, IndexTTS, OmniVoice, and VoxCPM adapters under **Production Workspace → TTS settings**. Incomplete configurations are visible and must be completed in Model Configuration. Local TTS runs through its existing HTTP service; NarraLens does not install or start those models. Connectivity and voice quality still require provider-specific validation.

Logical speech blocks retain the 12,000-Han-character limit and three-second gaps. Provider requests may be split further and joined without additional gaps: Qwen uses 600 characters, Tencent conservatively uses 150 for mixed text, the existing Doubao v1 route uses 1,024 UTF-8 bytes, and other new adapters default to 500 characters. Advanced service parameters `request_max_chars` / `request_max_bytes` adjust request sizes within known hard limits. Long-form subtitle alignment continues to use the actual audio.

To change speech for an existing project, choose the new service and voice, open **Reuse copy**, uncheck **Keep historical speech settings**, and create the copy. Ordinary resume retains recorded speech settings. New speech requires fresh alignment and dependent visual, animation, and music timing; the original project remains available.

Azure and Tencent SDKs select their endpoints by region; leave the interface address empty. A custom Qwen address uses the native DashScope API root. Doubao reuses the existing v1 adapter; v3 / 2.0 voice protocols were not added. Advanced parameters follow each existing adapter's capabilities. Text and vision backups retain their independent endpoints and credentials.

### Subtitle timestamp model

Long-form narration in the manuscript workflow uses [Qwen/Qwen3-ForcedAligner-0.6B-hf](https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B-hf) for local forced alignment. It takes synthesized speech and the corresponding manuscript, predicts spoken character/word boundaries, and maps them to subtitle timestamps. Chinese alignment primarily uses characters. The selected text/subtitle model plans semantic segmentation.

- An independent worker loads the model only when unfinished blocks need alignment. Successful timestamp caches are reused; exiting the worker releases GPU resources.
- Speech blocks longer than five minutes are located and split into alignment windows while retaining the existing speech audio.
- Prepare a separate PyTorch/Transformers environment and local model weights. The current GPU worker uses BF16 and the cuda device interface; AMD GPUs require a working ROCm environment.
- Local alignment is enabled by default. Turning it off keeps subtitles and estimates timing from audio duration and text weights, which can differ from actual speech.

Explanatory animations use the configured Node, Remotion, and browser rendering environment. See the animation setup and rollback notes.

## Production workflow

1. **Configure models:** maintain service definitions and select them by role in each workspace.
2. **Build video libraries:** scan video/subtitle directories, review matches, register, and analyze. ASS/SSA and SRT are supported. Select the intended subtitle file/style when several translations are present.
3. **Prepare optional audio libraries:** register music and analyze whole tracks through an audio-capable service, or choose music manually.
4. **Register a project:** paste or upload the full manuscript in Project Management and select one or more video libraries.
5. **Configure and produce:** in Production Workspace, choose speech, blocks and pauses, model roles, subtitle style, output, BGM, and animation placement. Save and generate.
6. **Review and adjust:** inspect the video, matching details, subtitle/animation plans, and project diagnostics. Apply changes through the available copy actions when earlier output exists.

Visual units use existing narration timestamps and do not change TTS blocks. Paginated details show the corresponding manuscript span, source range, direct or supporting relationship, and evidence. Missing coverage remains visible before a complete timeline exists. Concrete events require direct support; supporting footage is restricted to suitable state or abstract units. Unselected footage no longer automatically fills gaps. Add suitable media or animations and resume while retaining speech, subtitles, and the previous video.

Video and music originals normally remain at their registered paths. Libraries save analysis and derived data. Successful analysis can be reused, and interrupted runs can continue.

Detailed instructions are in the workflow guide.

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

## Upgrading to 2.8

1. Libraries already using the content schema (schema 2) can be opened directly. Only legacy window libraries show **Create a 2.8 analysis copy** (创建 2.8 拉片副本): copy registrations and owned subtitles into a separate database while referencing the same original videos and retaining the old library.
2. Confirm the adaptation, episode mapping and subtitles. Analyze one or two episodes first, inspect events and evidence, then continue the remaining assets.
3. Create a new project or a reusable copy with the new libraries. Compatible narration, subtitle and alignment caches remain reusable. Requirements and shot matches use the new 2.8 protocol.
4. Continue from **3. Find footage**. Successful observations and matches are cached independently; continuation uses saved retrieval cursors. Shortages distinguish incomplete analysis, unresolved requests, missing event matches, duration and source allocation conflicts.

The video library combines subtitles and sampled frames into attributed observations, then groups actual shot ranges into content events. Request batches are not semantic boundaries. The long-take target duration divides analysis ranges; it is not a frame sampling interval or a fixed event duration.

Optional work notes include Markdown, character aliases, periods, local reference shots, episode mapping and attributed Bangumi data. External information guides interpretation and never proves that a local shot contains an event. Local analysis works without external data.

The content browser shows event summaries, source previews and evidence references. Individual ranges can be reanalyzed or corrected. Similar sampled frames are suggestions; confirmed duplicate groups participate in timeline deduplication.

Concrete plot events retain direct evidence requirements. Commentary and interpretation can use grounded illustrations. Explicit event chronology is preserved; lists of properties or opinions do not automatically impose shot order. Source reuse, credits and minimum shot duration remain checked.

Libraries and historical narration, captions, animation and final videos are not automatically rewritten. This release does not rebuild every library in the background. Use a project copy when switching libraries.

## Independent agent analysis (2.9)

In the video library analysis section, choose **Automatic analysis / External Agent**:

1. Select assets, expand **External Agent tasks**, and create a local analysis task.
2. Local preparation saves shot ranges, subtitle snapshots and sparse previews. Copy the task instruction into an agent with local file, image-viewing and command capabilities.
3. The agent follows the generated `TASK.md`, reviews episodes, requests extra frames or crops, and submits observation and event drafts through the CLI.
4. Refresh the task in NarraLens, inspect the result check, then import accepted results. Existing results, input conflicts and unfinished items are reported separately.
5. Imported observations use the existing content browser and manuscript retrieval. Partial observations may be imported before the whole directory is complete.

Tasks live in `storage/agent_analysis_tasks/TASK_ID/`. Original videos remain read-only path references. Each task owns its subtitle snapshot, evidence, proposals and receipts. Imported evidence is copied into library ownership and survives removal of the task directory.

The CLI entry is `python -m app.services.library.agent_tasks`, with `describe`, `overview`, `ranges`, `view`, `capture`, `references`, `submit`, `stop` and `resume`. Generated instructions include the actual Python environment, working directory and commands. CLI submission saves drafts; the WebUI performs library imports.

Local preparation and checks do not call models. The external agent uses its own model and account; cloud models still require a network connection. This version uses manual task handoff and does not include an MCP server or a client-specific agent launcher. A stop flag prevents subsequent local tool operations; stop external model requests in the agent application.

## Version 2.9.1

Source frames now use presentation-order ordinals together with integer PTS and a rational time base. Physical shots and analysis ranges use half-open frame intervals. Shot identities and source deduplication no longer estimate frame numbers from average FPS.

- One sequential scan records the frame table and scene cuts. Only metadata and requested evidence images are stored. Completed indices are reused; an interrupted video needs a new scan.
- Automatic analysis and external agents share frame capture. The agent `capture` command accepts either `--time` or `--frame`, verifies decoded source positions and keeps crops attached to the same parent frame.
- Long-take duration remains a setting in seconds, with subtitle-adjacent boundaries mapped onto real frames. Range details display source ordinals and timestamps.
- Targeted retrieval retains the full range and evidence baseline. Editing carries the selected video stream and source time origin while narration and output-frame clocks remain independent.

**Previous indices, analysis results and agent tasks are not compatible.** Keep video/subtitle registrations and work notes, then rebuild analysis and create new agent tasks. Old source caches cannot drive new editing. Opening a page does not delete existing data. Original media, external subtitle files and audio libraries are unaffected by this protocol change.

Static source checks have been performed. Automated tests, actual frame extraction, model-powered analysis, agent runs, live WebUI interaction and final-video acceptance have not been run. The initial scan still decodes the full video on the CPU. Precise indexing does not guarantee correct scene detection or visual interpretation.

## Version 2.9.0

Adds one agent task service and CLI while sharing existing media processing, observation/event validators and library transactions. Task operations use process-level file locks and source-bound receipts. Extra frames record decoder timestamps. Partial imports retain unrelated records, detect later edits and skip duplicate results.

The analysis baseline is distinct from evidence revisions, allowing normal continuation to reuse imported observations even when extra images were obtained. The automatic workflow, schema 2, TTS, alignment, BGM, explanatory animation and final rendering remain available.

Static source checks have been performed. Automated tests, actual frame extraction, external-agent runs, live WebUI interaction, model-powered library builds and final-video acceptance have not been run for this release. Quality and performance improvements need a real comparison.

## Version 2.8.1

This patch updates video library controls and state handling:

- Overview, work notes, registered assets, analysis jobs, browsing and maintenance have separate sections with remembered expansion state.
- Forms and tables edit character cards, episode mapping and observations. Frame/caption references are selected from the current range. Advanced JSON is collapsed by default and imports into the same validated form draft.
- Observations remain browsable when the event directory is incomplete. Missing or outdated references show an explanation.
- The **long-take analysis target duration** defaults to 15 seconds, with up to 50 ranges per request and a concurrency limit of 32. Scope is explicitly all assets or selected assets; the target duration is not a sampling interval.
- Source availability is checked explicitly in the background with an eight-second per-file timeout. Permission, network, missing-file and timeout results are separate; page refreshes do not probe every source path.
- Subtitle restoration marks related observations and events for rebuilding. Targeted repairs preserve other observations; directory continuation reuses successful results.

The database stays at schema 2. Nothing is migrated, cleared or rebuilt automatically. Older 2.8 event records without observation revision links need a normal continuation for the affected assets; saved observations remain reusable. Import, analysis, subtitle restoration and source checks require explicit user actions.

The 2.8 core continues to use event-to-shot retrieval, stable media range IDs, combined subtitle/frame observations, optional work knowledge, local corrections and targeted enrichment. It removes the old candidate expansion and multiple identifier translation/recovery branches while reusing the existing API gateway and concurrency coordinator.

No new local model, vector database or CUDA requirement is introduced. Existing TTS, forced alignment, subtitle layout, audio library, BGM, Remotion and FFmpeg rendering remain in use.

Static source checks are complete. Automated tests, live WebUI interaction, model-powered library builds, token comparisons and final-video acceptance have not been run for this patch. Inspect a small sample before expanding the analysis scope; sampled frames and model judgments can remain uncertain.

Local implementation notes stay in docs. The public repository excludes those machine-specific records, tests, credentials, media, environments and generated output.

## License and acknowledgements

NarraLens retains NarratoAI's MIT license. See the complete local [LICENSE](LICENSE) and [upstream copy](https://github.com/linyqh/NarratoAI/blob/main/LICENSE). The original notice is retained:

```text
Copyright (c) 2024 linyq
```

Retain the original copyright notice and full license notice, including the disclaimer, when distributing inherited software or substantial portions. Attribution and links in Home or this README accompany those notices; distribution packages must still include the required text. [MIT standard text](https://opensource.org/license/mit)

Third-party components, models, renderers, and API services follow their own licenses or terms. The new display name does not change those conditions.

Thanks to NarratoAI and its contributors. NarraLens versions and extensions are maintained independently. A dedicated maintenance channel has not yet been supplied; upstream links identify the source and its documentation.
