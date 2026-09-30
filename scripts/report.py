#!/usr/bin/env python3
"""Publish auditable trace excerpts and figures; every plotted value comes from a run."""

import argparse, csv, json, shutil
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

p = argparse.ArgumentParser()
p.add_argument("--runs", default="runs/benchmark-final")
a = p.parse_args()
root = Path(__file__).resolve().parents[1]
runs = Path(a.runs)
assets = root / "docs/assets"
data = root / "docs/data"
raw = root / "results/reference"
for d in (assets, data, raw):
    d.mkdir(parents=True, exist_ok=True)
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.labelcolor": "#243b48",
        "text.color": "#182c37",
        "xtick.color": "#536874",
        "ytick.color": "#536874",
        "axes.edgecolor": "#c9d2d5",
        "svg.fonttype": "none",
        "savefig.facecolor": "#fafbf9",
    }
)
colors = {"latest": "#087f82", "fifo": "#bb603b"}
sets = {}
summaries = []
for policy in colors:
    rows = []
    for folder in sorted(runs.glob(policy + "-*")):
        dest = raw / folder.name
        dest.mkdir(exist_ok=True)
        for filename in [
            "timings.csv",
            "summary.json",
            "observations.jsonl",
            "responses.jsonl",
        ]:
            shutil.copyfile(folder / filename, dest / filename)
        r = list(csv.DictReader((folder / "timings.csv").open()))
        rows += r
        s = json.loads((folder / "summary.json").read_text())
        s["run"] = folder.name
        summaries.append(s)
    sets[policy] = {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}
shutil.copyfile(runs / "environment.json", raw / "environment.json")
metrics = {}
for policy, d in sets.items():
    y = d["observation_latency_ms"]
    metrics[policy] = {
        "n_processed": len(y),
        "latency_p50_ms": float(np.median(y)),
        "latency_p95_ms": float(np.percentile(y, 95)),
        "inference_p50_ms": float(np.median(d["inference_ms"])),
        "queue_p50_ms": float(np.median(d["queue_ms"])),
    }
env = json.loads((runs / "environment.json").read_text())
metrics["environment"] = {
    "cpu": env.get("cpu", env["processor"]),
    "architecture": env["machine"],
    "opencv": "4.12.0",
    "model": "YOLOX-s FP32",
    "threads": 2,
    "source_fps": 30,
    "queue_capacity": 4,
    "runs_per_policy": env["repeats"],
    "input_frames_per_run": 240,
    "workload": "controlled transformations of a CC0 photograph",
    "date": env.get("date", "unspecified"),
}
(raw / "metrics.json").write_text(json.dumps(metrics, indent=2))
(data / "metrics.json").write_text(json.dumps(metrics, indent=2))
# Select observations from one actual run for a browsable time-synchronized trace.
folder = runs / "latest-1"
obs = [json.loads(l) for l in (folder / "observations.jsonl").read_text().splitlines()]
responses = [
    json.loads(l) for l in (folder / "responses.jsonl").read_text().splitlines()
]
selected = []
for t in np.arange(0, 8000, 350):
    o = min(obs, key=lambda x: abs(x["media_ms"] - t))
    if selected and o["frame"] == selected[-1]["frame"]:
        continue
    dst = f"frame-{o['frame']}.jpg"
    shutil.copyfile(folder / "frames" / f"{o['frame']}.jpg", data / dst)
    o = dict(o)
    o["image"] = "data/" + dst
    selected.append(o)
(data / "trace.json").write_text(
    json.dumps(
        {
            "observations": selected,
            "responses": responses,
            "summary": json.loads((folder / "summary.json").read_text()),
        },
        indent=2,
    )
)


def save(fig, name):
    fig.savefig(assets / (name + ".svg"), bbox_inches="tight", pad_inches=0.22)
    fig.savefig(assets / (name + ".png"), dpi=180, bbox_inches="tight", pad_inches=0.22)
    plt.close(fig)


def head(fig, kicker, title, sub):
    fig.text(0.055, 0.97, kicker, fontsize=9, color="#087f82", weight="bold", va="top")
    fig.text(0.055, 0.91, title, fontsize=23, weight="medium", va="top")
    fig.text(0.055, 0.84, sub, fontsize=10, color="#536874", va="top")


# A three-panel visual argument, rendered from actual inference outputs.
fig, axs = plt.subplots(1, 3, figsize=(13.5, 5.8), facecolor="#fafbf9")
fig.subplots_adjust(top=0.70, bottom=0.22, wspace=0.11, left=0.055, right=0.96)
head(
    fig,
    "CUEFRAME  /  OBSERVATION → REFERENCE → RESPONSE",
    "Keep the evidence. Update the answer.",
    "Three moments from an actual local detector run · Controlled photo transformations, not human footage",
)
for ax, t, title, caption in zip(
    axs,
    [100, 3500, 4700],
    ["01  Observe", "02  Remember", "03  Clarify"],
    [
        "One cup grounds a current reference.",
        "The frame is blank; earlier evidence remains.",
        "Two cups make “cup” ambiguous.",
    ],
):
    o = min(obs, key=lambda x: abs(x["media_ms"] - t))
    ax.imshow(Image.open(folder / "frames" / f"{o['frame']}.jpg"))
    ax.set_axis_off()
    ax.set_title(title, loc="left", fontsize=13, pad=13)
    ax.text(
        0,
        -0.12,
        f"{o['media_ms'] / 1000:.2f} s  /  frame {o['frame']}",
        transform=ax.transAxes,
        color="#087f82",
        fontsize=10,
    )
    ax.text(0, -0.24, caption, transform=ax.transAxes, fontsize=9)
fig.text(
    0.055,
    0.025,
    "YOLOX-s · OpenCV CPU · Coffee photo: Rachel Michetti / CC0. Boxes are drawn on their own source frames.",
    fontsize=8,
    color="#697b84",
)
save(fig, "overview")
# Distributions instead of a single headline number. No cross-frame confidence intervals.
fig, axs = plt.subplots(1, 3, figsize=(13.5, 5.6), facecolor="#fafbf9")
fig.subplots_adjust(top=0.69, bottom=0.22, wspace=0.38, left=0.07, right=0.96)
head(
    fig,
    "CUEFRAME  /  REFERENCE MEASUREMENTS",
    "Fresh observations under load.",
    f"Same detector, input rate and queue capacity · {env['repeats']} runs per policy · {env.get('cpu', env['processor'])} / 2 CPU threads",
)
for policy, d in sets.items():
    x = np.sort(d["observation_latency_ms"])
    axs[0].plot(
        x, np.arange(1, len(x) + 1) / len(x), color=colors[policy], label=policy, lw=2
    )
axs[0].set(
    xlabel="Acquisition → observation (ms)",
    ylabel="Cumulative fraction",
    xlim=(0, None),
    ylim=(0, 1),
)
axs[0].legend(frameon=False, loc="lower right")
axs[0].set_title("A  Latency distribution", loc="left", pad=12)
for i, (policy, d) in enumerate(sets.items()):
    vals = d["observation_latency_ms"]
    v = axs[1].violinplot(vals, positions=[i], widths=0.65, showextrema=False)
    for b in v["bodies"]:
        b.set_facecolor(colors[policy])
        b.set_alpha(0.2)
        b.set_edgecolor("none")
    q = np.percentile(vals, [5, 50, 95])
    axs[1].plot([i, i], [q[0], q[2]], color=colors[policy], lw=2)
    axs[1].scatter([i], [q[1]], color=colors[policy], s=35, zorder=3)
    axs[1].text(i + 0.14, q[1], f"{q[1]:.0f}", fontsize=9, color=colors[policy])
    axs[1].text(i + 0.14, q[2], f"p95 {q[2]:.0f}", fontsize=8, color=colors[policy])
axs[1].set(
    xticks=[0, 1],
    xticklabels=["latest", "fifo"],
    ylabel="Observation latency (ms)",
    ylim=(0, None),
)
axs[1].set_title("B  Median + 5–95% range", loc="left", pad=12)
for i, (policy, d) in enumerate(sets.items()):
    for j, key in enumerate(["queue_ms", "inference_ms"]):
        vals = d[key]
        axs[2].scatter(
            np.full(len(vals), j) + (i - 0.5) * 0.25,
            vals,
            s=4,
            alpha=0.13,
            color=colors[policy],
            rasterized=True,
        )
        axs[2].scatter(
            j + (i - 0.5) * 0.25,
            np.median(vals),
            s=40,
            color=colors[policy],
            edgecolor="white",
            linewidth=0.6,
            zorder=3,
        )
axs[2].set(
    xticks=[0, 1],
    xticklabels=["Queue wait", "Inference"],
    ylabel="Time (ms)",
    ylim=(0, None),
)
axs[2].set_title("C  Where time goes", loc="left", pad=12)
fig.text(
    0.055,
    0.07,
    "latest evicts oldest queued frames; fifo rejects new arrivals when full. Both queues hold at most 4 frames.",
    fontsize=9,
)
fig.text(
    0.055,
    0.025,
    "Pooled frame distributions; frames are correlated. Includes decode + queue + inference + evidence update; excludes image export and response delay.",
    fontsize=8,
    color="#697b84",
)
save(fig, "performance")
# Observed response decisions, not a theoretical architecture mock-up.
fig, ax = plt.subplots(figsize=(13.5, 4.8), facecolor="#fafbf9")
fig.subplots_adjust(top=0.7, bottom=0.22, left=0.19, right=0.94)
head(
    fig,
    "CUEFRAME  /  RESPONSE TRACE",
    "A reference can outlive a frame. An old turn cannot.",
    "Actual decisions from latest-1 · Response worker uses templates and an injected 400 ms delay",
)
cs = {"published": "#087f82", "suppressed": "#bb603b", "clarify": "#887139"}
for i, r in enumerate(responses):
    end = r["finished_ms"] / 1000
    start = r["evidence_ms"] / 1000
    if r["reason"] == "ambiguous_reference":
        start = end
    ax.plot([start, end], [i, i], color=cs[r["status"]], lw=2, alpha=0.7)
    ax.scatter([end], [i], s=48, color=cs[r["status"]], zorder=3)
    if start != end:
        ax.scatter(
            [start],
            [i],
            s=35,
            facecolors="#fafbf9",
            edgecolors=cs[r["status"]],
            zorder=3,
        )
    ax.text(
        end + 0.12,
        i,
        r["reason"].replace("_", " "),
        va="center",
        fontsize=9,
        color=cs[r["status"]],
    )
ax.set(
    yticks=range(len(responses)),
    yticklabels=[r["query"] for r in responses],
    xlabel="Replay time (seconds)",
    xlim=(0, 9),
    ylim=(len(responses) - 0.5, -0.6),
)
ax.spines["left"].set_visible(False)
ax.tick_params(axis="y", length=0)
ax.grid(axis="x", alpha=0.13)
fig.text(
    0.055,
    0.025,
    "Open marker: selected evidence time. Filled marker: decision time. Historical responses explicitly name their source observation.",
    fontsize=8,
    color="#697b84",
)
save(fig, "response-trace")
print(json.dumps(metrics, indent=2))
