# NarraLens · 叙镜

> Turn your manuscript into a video.

**Manuscript-driven video production · v2.7.11**

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

Version 2.7.11 provides seven workspaces:

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

Explanatory animations use separately prepared Node, Remotion, and browser dependencies.

## Production workflow

1. **Configure models:** maintain service definitions and select them by role in each workspace.
2. **Build video libraries:** scan video/subtitle directories, review matches, register, and analyze. ASS/SSA and SRT are supported. Select the intended subtitle file/style when several translations are present.
3. **Prepare optional audio libraries:** register music and analyze whole tracks through an audio-capable service, or choose music manually.
4. **Register a project:** paste or upload the full manuscript in Project Management and select one or more video libraries.
5. **Configure and produce:** in Production Workspace, choose speech, blocks and pauses, model roles, subtitle style, output, BGM, and animation placement. Save and generate.
6. **Review and adjust:** inspect the video, matching details, subtitle/animation plans, and project diagnostics. Apply changes through the available copy actions when earlier output exists.

Visual units use existing narration timestamps and do not change TTS blocks. Paginated details show the corresponding manuscript span, source range, direct or supporting relationship, and evidence. Missing coverage remains visible before a complete timeline exists. Concrete events require direct support; supporting footage is restricted to suitable state or abstract units. Unselected footage no longer automatically fills gaps. Add suitable media or animations and resume while retaining speech, subtitles, and the previous video.

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

After upgrading to 2.7, resume old timelines from **3. Find footage**, reusing narration, captions, and alignment. Stages 4 and 5 cannot bypass the new content checks. The previous video and timeline are retained under the project's `history/visual-*` directory. When candidate visual confirmation is disabled, content relationships are still reviewed using saved descriptions and range-specific subtitles and marked as not visually rechecked. The independent credits/usability check still uses a vision model.

Version 2.7.1 accepts valid zero-duration alignment entries and uses positive-duration speech anchors while retaining the full manuscript. Projects stopped by this issue can resume from stage 3 using existing narration and captions. Missing positive-duration anchors and invalid timing data still produce explicit errors.

Version 2.7.2 uses range-scoped subtitle IDs and restores the original quotes in code. Identifiable row errors reject that candidate while retaining valid peers; unconfirmed formats and content shortages are recorded separately. Resume from **3. Find footage** to apply the new protocol, reusing narration, captions, alignment, and independent usability checks. Previous videos and timelines are retained; library reanalysis is unnecessary.

Version 2.7.3 uses token-conscious footage scheduling from stage 3: restore approved candidates, prepare candidates locally, optionally run text-only model preselection, propose the full timeline, then inspect proposed footage as needed. Per-unit expansion pauses after successive rounds without new approved duration; continuing reuses caches and advances the search cursor. The workspace displays reported tokens for the current run and project history. Narration, captions, alignment, and previous output are retained; provisional plans cannot be rendered.

Version 2.7.4 reconciles shared visual cuts after restoring semantic groups, checking original-text bindings, rendered animation reservations, and speech gaps with a half-frame rounding tolerance. Projects stopped by the three reported subsecond intervals can resume from **3. Find footage** using existing narration, captions, and animations. Search pools and cursors are reused only when the original plan key, library, and model settings match. Genuine short intervals and binding conflicts still pause with a reason.

Version 2.7.5 fixes the pending-state conflict that blocked new candidate approval. Valid peers are retained while necessary malformed rows get one small correction pass. Saved gaps can receive targeted review of commentary mistaken for concrete facts while retaining actual plot constraints. Unchanged units retain search progress; changed requirements revalidate existing ranges. Resume from **3. Find footage**. Necessary corrections and requirement reviews call the selected models and count toward token usage; genuine footage shortages are still reported.

Version 2.7.6 filters invalid preselection IDs individually and retains valid choices. Empty selections retain only a few pending local alternatives; text protocol errors no longer mark whole groups as visually unconfirmed. Revised requirements reuse old approvals only when manuscript, source range, model cache identity and final admission checks prove compatibility. No-gain rounds require completed valid content assessment, with separate protocol, allocation and visual-review pause reasons. Resume from **3. Find footage**. New reviews still consume tokens; genuine footage gaps remain visible.

Version 2.7.7 saves temporary source conflicts and inconclusive reviews per candidate, then tries other ranges. Local retrieval continues past empty window batches. Resume advances the saved candidate queue and window cursor while reusing paid results. Unknown reply IDs are rejected individually and valid peers are retained. Strictly identified negative results may be validated locally without another correction request. User-configured valid no-gain stopping and final factual, pending-review, unique-source and credits gates remain required. Resume from **3. Find footage**; new reviews still consume tokens.

Version 2.7.8 separates ASS language hints from dialogue, lyrics and translator annotations, preserving Chinese dialogue in neutral styles. Video Library adds subtitle checks, local dialogue/index recovery and per-asset rollback, retaining old files and visual analysis. Restored subtitle semantics are marked pending. Known dialogue loss blocks paid footage search. Each actual review handles one manuscript unit with multiple candidates. Grounded atomic facts may be covered by several shots, but only evidence inside the footage actually used counts. Unique-source, credits and minimum-duration checks remain required. Check/recover subtitle data before resuming from **3. Find footage**. Model review and full output still require live acceptance.

Version 2.7.9 uses original-text part IDs for gap requirement revision. The model selects ranges and code restores quotes, offsets and fact IDs, without repeated quotation copying or legacy fact-index remapping. Valid peers are saved immediately. Invalid units receive one local correction with field-specific feedback; unresolved work pauses with saved progress. Parsed replies, request references and application validation remain in one batch audit. Compatible successful requirements can be reused. Resume from **3. Find footage**; new revisions and corrections consume tokens. No model calls or generation were run for this release; semantic quality and full output still require live acceptance.

Version 2.7.10 schedules footage reviews across independent manuscript units with bounded concurrency, defaulting to 4 and adjustable in advanced output settings. Set 1 for the same serial path. Each request still handles one unit and a small set of proposed candidates. Inputs are frozen; the task thread persists replies, updates usage and commits shared state before batch allocation. Remaining duration bounds candidate submissions, with temporary source reservations and all factual, credits, source uniqueness and minimum-shot gates retained. New reply receipts support recovery after interrupted cache application; compatible caches remain reusable. Restart and resume from **3. Find footage**. Live concurrency, speed/token comparisons and full output have not been tested; no fixed speedup is promised.

Version 2.7.11 repairs requirement replies with uniquely resolvable zero-padding differences and JSON-identical duplicates. Conflicting results, wrong or ambiguous IDs and invalid original-text ranges remain blocked. Correction feedback includes the actual returned IDs and rows. Pending requirements may reuse the latest saved reply only after verifying its complete input identity and passing the existing validator. Compatible successful caches and 2.7.10 concurrency remain unchanged. Restart and resume from **3. Find footage**. Offline debugging with saved replies passed; no model calls or full generation were run.

Delete a manuscript project through **Project Management → select project → Manage Project → inspect files and deletion scope → confirm → permanently delete**. This removes the current project and its output while retaining original media and external manuscript files. Associated running work must stop first.

Code archives normally exclude storage, media, environments, and weights. Back up project data separately when preserving audio, videos, and analysis.

## Version and documentation

Version 2.7.11 fixes requirement ID matching, identical duplicate rows and missing correction data, with controlled recovery of paid pending replies. Offline debugging and static review are recorded separately; live model and full-video acceptance remain pending.

Version 2.7.10 added configurable footage review concurrency, defaulting to 4 with the same serial path at 1. Frozen inputs, shared validation and task-thread commits support batch allocation and paid-result recovery. Static review is recorded separately from live concurrency, token and full-generation acceptance, which remain pending.

Version 2.7.9 simplified requirement revision and validation, reuses compatible successful results, saves complete parsed replies and corrects only invalid units once. Recoverable work pauses with progress retained. Implementation and static review are recorded separately from live model and full-generation acceptance, which have not been run.

Version 2.7.8 repaired lost Chinese ASS dialogue, adds reversible local index recovery, isolates actual reviews by manuscript unit and supports grounded multi-shot fact coverage. Existing narration, subtitle timing, compatible caches and old output are retained. Implementation and static review are complete; full generation and token effects have not been validated.

Version 2.7.7 repaired premature unit pauses, empty local batch handling and repeated selection of blocked candidates on resume. Saved candidate progress tracks allocation changes and inconclusive reviews. Valid reply peers and narrowly normalized negative results retain their audit records. Implementation and static review are complete; live gap improvement, token consumption and full video output have not been validated.

Version 2.7.6 repaired invalid preselection IDs, strictly compatible approval reuse and valid no-gain accounting while retaining existing audio, subtitles and footage ranges.

Version 2.7.5 fixed candidate approval state, local format correction, legacy cache scope and targeted gap requirement review, while retaining original grouping and unchanged-unit search progress.

Version 2.7.4 reconciles animation frame edges and visual cuts with the same original-text binding while retaining narration, captions, and animations. Saved project recovery records show the three reported remnants resolved; full video output still needs validation.

Version 2.7.3 added optional text preselection, provisional allocation and on-demand footage confirmation, shared request catalogues with canonical ID restoration, approved-result reuse, search checkpoints, no-gain pauses, and reported token totals. Final factual, source deduplication, credits and timeline checks remain required. Token savings and video quality still need validation.

Version 2.7.2 binds subtitle evidence to candidate-specific IDs and rejects local errors while retaining valid peers. Abstract claims and concrete constraints are separated, with zero-duration alignment compatibility retained.

Version 2.7.1 fixes visual planning rejecting otherwise valid alignment data containing zero-duration entries. The manuscript, audio, captions, and source caches are retained; only positive-duration points serve as speech anchors. Invalid ranges, malformed times, and missing voiced anchors remain errors. Existing valid 2.7 visual plans remain compatible.

Version 2.7.0 adds visual units bound to existing narration timing, reviews content relationships for actual source ranges, and removes automatic admission of unselected fallback footage. Whole-video allocation reserves footage for concrete events and scarce evidence before arranging duration. Gap diagnostics and paginated details preserve long-form TTS, captions, manual animation nodes, source-shot deduplication, and credits filtering. The implementation is in place; improvements still require review of generated footage. Three-frame sampling and model judgments cannot guarantee correct identification in every case.

Version 2.6.1 connects manuscript projects to the existing TTS adapters, with per-configuration voice drafts, provider-sized requests, cached internal audio, task-scoped credentials, and recursive credential removal from saved snapshots. Existing MiMo/Edge caches, three-second logical block gaps, and local alignment remain supported. Provider connectivity and voice quality have not been validated for every service.

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
