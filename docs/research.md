# From visual evidence to shared understanding

**Research question:** How should an interactive agent associate a spoken reference with visual evidence, and revise that association when the person or scene changes?

CueFrame is independent of a particular avatar, patient simulator, or dialogue model. The runtime is the instrument; a research contribution must come from a tested method or finding.

## Next controlled study

Record consented, staged desktop interactions with object reference, correction, occlusion, and delayed response. Annotate which observation supports each utterance. Split by participant and scene, not random frames. Compare text only, latest frame, recent-frame windows, and an event-based evidence policy at matched model and inference budgets. Measure referent accuracy, recovery after correction, stale-evidence errors, clarification usefulness, and response latency. Include cumulative movement and references to earlier objects, where a latest-only policy can fail. These measurements are planned, not results of v0.1.

The current transformed-photo replay tests engineering behavior under overload. It cannot establish social understanding, user benefit, robust object tracking, or a new state-of-the-art result.

## Connections to prior work

- [ENACT (2025)](https://arxiv.org/abs/2511.20937): reasoning about action/observation sequences and changes in state. It motivates evaluating temporal relations rather than isolated image recognition.
- [CaP-X (2026)](https://arxiv.org/abs/2603.22435): embodied agents using perception, execution feedback, and interaction. CueFrame does not implement robot control.
- [Real-Time Reasoning Agents in Evolving Environments (2025)](https://realtimegym.saltlab.stanford.edu/): the world changes while an agent reasons; timeliness matters alongside decision quality.
- [Roleplay-doh (EMNLP 2024)](https://aclanthology.org/2024.emnlp-main.591/): experts provide principles for simulated-patient behavior. Visual evidence could become another input to such policies.
- [Towards Streaming Perception (ECCV 2020)](https://arxiv.org/abs/2005.10420): perception quality depends on when the output is available. Freshness-aware processing alone is not a new research claim.

## A future VOICE adapter

CueFrame would supply timestamped observations and uncertainty. VOICE would apply expert-defined patient capabilities and behavior principles to choose a response. A detector recognizing a cup must not force a simulated patient to name it correctly. Object/gesture cues can be evaluated first; expression and social interpretation require their own contextual labels and evaluation. No VOICE integration or human-subject validation is claimed in this release.

## Engineering roadmap

1. First release: C++ model inference, replay, camera input path, bounded queues, finite references, publication checks, public traces.
2. Interactive evidence: user-selected image point and utterance timestamps, better tracking, explicit evidence spans, live monitor.
3. Model adapter: interchangeable vision-language/dialogue backends with evidence citations and cancellation.
4. Research evaluation: unseen people/scenes, expert-defined policies, and a separate VOICE deployment study.
