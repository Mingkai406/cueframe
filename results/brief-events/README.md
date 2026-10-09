# Brief events: freshness and missed evidence

Dropping queued frames can lower latency and still miss an event. This experiment measures
both. A controlled 288-frame sequence contains 16 appearances of the same CC0 coffee photo,
lasting 1, 2, 4 or 8 frames (33–267 ms at 30 fps), separated by blank backgrounds. The runtime
performs real YOLOX-s inference; the input is **synthetic photo appearances, not natural video**.

Recorded 2026-10-08, OpenCV CPU with 2 threads, 30 fps, queue capacities 1 and 4, and three
fresh-process repetitions per policy/capacity (12 runs). Policy order alternates. Model and
binary hashes, fixture provenance, exact frame ranges and all event outcomes are retained.
No other project benchmark ran concurrently during these reference measurements.

| Queue capacity | Policy | Event coverage range across 3 runs | Median timely coverage | Median per-run p95 observation latency ms |
|---|---|---:|---:|---:|
| 1 | latest | 75.0%–93.8% | 43.8% | 108.6 |
| 1 | fifo | 75.0%–93.8% | 25.0% | 145.3 |
| 4 | latest | 81.2%–87.5% | 25.0% | 205.1 |
| 4 | fifo | 75.0%–87.5% | 0.0% | 368.6 |

**Lower latency does not imply complete event coverage.** Both policies miss some appearances.
At capacity 4, FIFO detects some events only after they have disappeared; none met the fixture's
strict disappearance deadline in these three runs. The smaller latest queue improves timely
coverage here, but still misses brief events. This is a design tradeoff, not a general object-
detection accuracy result. Only 16 scripted episodes per run are used; no statistical superiority
or natural-video recall claim is supported.

## Definitions and independent checks

- Event coverage: at least one processed frame inside the annotated interval has a **visible**
  cup detection. A remembered but invisible track does not count.
- Timely event coverage: that detection becomes available before the event's scheduled end.
  It uses acquisition lag plus observation latency, so delayed capture cannot hide latency.
- Miss cause: either every event frame was dropped, or retained frames had no cup detection.
  This distinguishes queue loss from detector failure instead of attributing both to scheduling.
- Background detections and per-event first-detection delay are also recorded.
- Produced/processed/dropped counts must reconcile; timing and observation frame IDs must match.
  Unpaced or mismatched-rate runs are rejected. Each episode counts once, not once per frame.

## Reproduce

Build the vision executable as described in the main README and install Pillow for fixtures.

```sh
python3 -m pip install Pillow
python3 scripts/event_benchmark.py --binary build/cueframe_run \
  --fps 30 --capacities 1 4 --repeats 3 --out runs/brief-events
python3 -m unittest discover -s tests -p 'test_*.py'
```

The script creates a new output directory and will not overwrite an existing experiment.
Use another `--fps` or queue-capacity list to explore the tradeoff; retain every run. Source
photo and frame-sequence hashes make the controlled input auditable. Frame previews remain
local; the checked-in CSV/JSONL files are sufficient to recompute all reported metrics.
