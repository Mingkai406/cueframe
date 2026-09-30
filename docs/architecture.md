# Runtime design

CueFrame v0.1 is a standalone C++20 visual evidence runtime. It provides a real detector, bounded streaming pipeline, finite reference queries, and a response publication contract. A future dialogue model can consume the same timestamped evidence.

## Scope and decisions

The first deliverable is an inspectable engineering prototype, with no dependency on VOICE, patient data, cloud credentials, or a language-model provider. Camera inputs stay local. Nothing accesses a camera until the user explicitly passes `--camera`.

Chosen architecture: one capture producer, one detector owner, a mutex-protected evidence store, and one response worker. A monolithic blocking loop was considered: it is simpler, but prevents a live source from advancing during inference and conceals backpressure. Multiple inference workers are deferred until there is evidence of a bottleneck that justifies out-of-order completion and extra memory. The current topology should be revisited for multiple cameras or accelerator batching.

The accepted initial scope is a new, independent repository. Existing VOICE and TurnServe code is unchanged. Rollback consists of reverting a CueFrame commit; no database migration or production deployment is involved.

## Data path

1. **Acquire.** Read BGR pixels from an image, image directory, video backend, or camera. Images are bounded to 1280 px on the longest side. Source frames carry a sequence number and media timestamp. Replays use the explicit `--fps` constant-rate clock; this is not a variable-frame-rate PTS-preserving decoder.
2. **Queue.** A bounded move-owned queue either evicts the oldest waiting frame (`latest`) or rejects the newest arrival (`fifo`). Both preserve processing order for admitted frames. Capture does not wait for inference. Closing drains surviving items and wakes consumers.
3. **Infer.** One thread owns YOLOX-s / OpenCV DNN. RGB conversion, 640-pixel top-left letterboxing, anchor decoding, class-aware NMS, and clipping map detections back to normalized source coordinates. No external inference service is used.
4. **Associate.** Same-class greedy IoU association retains local track IDs, marks missed tracks unobserved, and keeps up to 512 tracks and 96 immutable observation snapshots. This is not robust re-identification. Occlusion, crossing same-class objects, and detector jitter can break identity.
5. **Resolve.** A finite grammar selects an object and copies a ticket containing source epoch, turn, track/revision, box, and evidence time. `previous LABEL` selects the newest older retained snapshot containing that label. It does not infer an arbitrary past event or conversational referent.
6. **Commit.** Immediately before publication, check source epoch, superseding turns, evidence age, visibility, and target movement. An immutable historical ticket intentionally survives current-scene changes, but not a new turn or source reset. The decision linearizes while holding the store mutex; changes not yet observed cannot be detected.

## Contracts

| Condition | Decision |
|---|---|
| Same current target, age <= 750 ms | Publish a grounded template |
| Another query supersedes the pending turn | Suppress old response |
| Source reset | Suppress old current and historical responses |
| Current target missing, moved, or evidence expired | Suppress; caller may query again |
| Multiple candidates, invalid point, unknown class/reference | Request clarification |
| Historical reference with retained observation | Publish with its evidence timestamp |

Position change tolerance is 0.025 in normalized image coordinates; matching requires IoU > 0.25 and last observation within 1500 ms. These are engineering defaults, not learned or validated optimal values. Missed detections do not prove physical absence. A query consumes a new turn even when ambiguous, so an older pending response cannot override a correction.

## Response worker

The renderer uses grounded templates, not an LLM. `--response-delay-ms` injects a cancellable delay to test what happens when evidence changes while an answer is pending. This is explicitly different from measured detector inference. The work queue holds eight requests, evicts oldest under overload, and reports dropped queries. Publication checks defend against known stale results; they do not guarantee conversational correctness.

## Trace contract

- `observations.jsonl`: source frame, media time, track ID/revision, class, detector confidence, visibility, normalized box.
- `responses.jsonl`: query, turn, selected evidence time, completion time, historical flag, decision, reason, and text.
- `timings.csv`: frame ID, media time, queue wait, preprocessing, inference, postprocessing, acquisition-to-observation latency, visible tracks.
- `summary.json`: queue bounds/drops, process peak RSS, model/runtime details, warm-up, counts, and injected delay.
- `frames/*.jpg`: detections drawn on the exact frame used for inference. Old boxes are never rendered onto a newer captured frame.

`observation_latency_ms` ends after the evidence-store update, before JPEG writing or dialogue rendering. It includes decode and queue wait. Files and image export can influence throughput. Peak RSS is the whole process, not just queue memory. Historical memory is bounded by observations, not seconds; history coverage depends on throughput. Model warm-up is recorded separately.

## Verification and extension boundaries

The core tests cover queue overload/shutdown, invalid inputs, out-of-order observations, ambiguity, expiry, cumulative motion, unrelated scene changes, source reset, history eviction, and turn replacement. The model test checks actual ONNX inference and inverse-letterbox geometry in landscape and portrait layouts.

The camera path is implemented but requires a user-run hardware check. First-release measurements use a transformed public photograph, not human interaction footage. Free-form language, speech recognition, pointing gestures, facial behavior, learned cue fusion, patient policies, and clinical evaluation remain future work.
