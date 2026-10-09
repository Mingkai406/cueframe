#!/usr/bin/env python3
"""Measure event coverage alongside freshness using controlled brief cup appearances."""

import argparse
import csv
import hashlib
import json
import math
import platform
import subprocess
from pathlib import Path


def evaluate(manifest, summary, timings, observations):
    n = manifest["frames"]
    if not summary["paced"] or summary["source_fps"] != manifest["fps"]:
        raise ValueError(
            "Event deadlines require paced replay at the annotated frame rate"
        )
    if not timings:
        raise ValueError("No processed frames")
    if summary["produced"] != n or summary["processed"] + summary["dropped"] != n:
        raise ValueError("Incomplete run: frame accounting does not match fixture")
    times = {int(t["frame"]): t for t in timings}
    obs = {int(o["frame"]): o for o in observations}
    if (
        len(times) != len(timings)
        or len(obs) != len(observations)
        or times.keys() != obs.keys()
        or len(times) != summary["processed"]
    ):
        raise ValueError("Incomplete or duplicate observation/timing records")
    if any(not 0 <= frame < n for frame in times):
        raise ValueError("Frame outside fixture")
    for t in times.values():
        if any(
            not math.isfinite(float(t[k]))
            for k in ("media_ms", "capture_lag_ms", "observation_latency_ms")
        ):
            raise ValueError("Nonfinite timing")
    events = []
    for event in manifest["events"]:
        begin, end = event["start_frame"], event["end_frame"]
        candidates = [i for i in times if begin <= i < end]
        detected = [
            i
            for i in candidates
            if any(t["visible"] and t["label"] == "cup" for t in obs[i]["tracks"])
        ]
        first_ms = (
            min(
                (
                    float(times[i]["media_ms"])
                    + float(times[i]["capture_lag_ms"])
                    + float(times[i]["observation_latency_ms"])
                    for i in detected
                )
            )
            if detected
            else None
        )
        events.append(
            {
                **event,
                "retained_frames": len(candidates),
                "detected_frames": len(detected),
                "detected": bool(detected),
                "detected_before_disappearance": first_ms is not None
                and first_ms < end * 1000 / manifest["fps"],
                "first_detection_delay_ms": first_ms - begin * 1000 / manifest["fps"]
                if first_ms is not None
                else None,
                "miss_reason": None
                if detected
                else "queue_dropped_all"
                if not candidates
                else "detector_miss_on_retained_frames",
            }
        )
    background = [
        i
        for i in obs
        if not any(e["start_frame"] <= i < e["end_frame"] for e in manifest["events"])
    ]
    latencies = sorted(float(t["observation_latency_ms"]) for t in timings)
    return {
        "event_coverage": sum(e["detected"] for e in events) / len(events),
        "timely_event_coverage": sum(e["detected_before_disappearance"] for e in events)
        / len(events),
        "background_frames_processed": len(background),
        "background_frames_with_cup_detection": sum(
            any(t["visible"] and t["label"] == "cup" for t in obs[i]["tracks"])
            for i in background
        ),
        "observation_latency_p95_ms": latencies[math.ceil(0.95 * len(latencies)) - 1],
        "dropped_frames": summary["dropped"],
        "events": events,
    }


def fixture(out, fps):
    from PIL import Image

    out.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).resolve().parents[1] / "examples/coffee.png"
    base = Image.open(source).convert("RGB")
    blank = Image.new("RGB", base.size, (235, 234, 230))
    # Repeated on/off episodes with fixed phase offsets, including one-frame appearances.
    events = []
    for i in range(16):
        width = (1, 2, 4, 8)[i % 4]
        begin = 10 + i * 16 + (i % 3)
        events.append(
            {
                "id": f"cup-{i}",
                "start_frame": begin,
                "end_frame": begin + width,
                "duration_frames": width,
            }
        )
    n = 288
    hashes = []
    for i in range(n):
        visible = any(e["start_frame"] <= i < e["end_frame"] for e in events)
        path = out / f"{i:05d}.jpg"
        (base if visible else blank).save(path, quality=90)
        hashes.append(hashlib.sha256(path.read_bytes()).hexdigest())
    manifest = {
        "frames": n,
        "fps": fps,
        "events": events,
        "kind": "Controlled CC0 photo appearances, not natural video or a general accuracy benchmark",
        "source": "Rachel Michetti / scikit-image coffee, CC0",
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "frame_sequence_sha256": hashlib.sha256("".join(hashes).encode()).hexdigest(),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--binary", default="build/cueframe_run")
    p.add_argument("--model", default="models/yolox_s.onnx")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--capacities", type=int, nargs="+", default=[1, 4])
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--out", type=Path, default=Path("runs/events"))
    a = p.parse_args()
    if (
        not 1 <= a.fps <= 240
        or a.repeats < 1
        or any(not 1 <= c <= 128 for c in a.capacities)
    ):
        p.error("invalid experiment bounds")
    a.out.mkdir(parents=True, exist_ok=False)
    manifest = fixture(a.out / "fixture", a.fps)
    rows = []
    for capacity in a.capacities:
        for repeat in range(a.repeats):
            for policy in ["latest", "fifo"] if repeat % 2 == 0 else ["fifo", "latest"]:
                dest = a.out / f"{policy}-c{capacity}-r{repeat}"
                subprocess.run(
                    [
                        a.binary,
                        "--input",
                        str(a.out / "fixture"),
                        "--model",
                        a.model,
                        "--out",
                        str(dest),
                        "--frames",
                        str(manifest["frames"]),
                        "--fps",
                        str(a.fps),
                        "--capacity",
                        str(capacity),
                        "--policy",
                        policy,
                        "--threads",
                        "2",
                    ],
                    check=True,
                    timeout=360,
                )
                summary = json.loads((dest / "summary.json").read_text())
                with (dest / "timings.csv").open() as f:
                    timings = list(csv.DictReader(f))
                observations = [
                    json.loads(s)
                    for s in (dest / "observations.jsonl").read_text().splitlines()
                ]
                rows.append(
                    {
                        "policy": policy,
                        "capacity": capacity,
                        "repeat": repeat,
                        **evaluate(manifest, summary, timings, observations),
                    }
                )
    report = {
        "environment": platform.platform(),
        "threads": 2,
        "binary_sha256": hashlib.sha256(Path(a.binary).read_bytes()).hexdigest(),
        "model_sha256": hashlib.sha256(Path(a.model).read_bytes()).hexdigest(),
        "fixture": manifest,
        "runs": rows,
        "interpretation": "Each episode counts once. Coverage can include late evidence; timely coverage requires output before the scheduled disappearance. Report repeats separately, not independent frames.",
    }
    (a.out / "event-report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    for row in rows:
        print({k: v for k, v in row.items() if k != "events"})


if __name__ == "__main__":
    main()
