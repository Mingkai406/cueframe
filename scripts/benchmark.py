#!/usr/bin/env python3
"""Equal-capacity sequential policy comparison; each run has a fresh process."""

import argparse, subprocess, pathlib, json, platform, datetime

p = argparse.ArgumentParser()
p.add_argument("--binary", default="build/cueframe_run")
p.add_argument("--repeats", type=int, default=3)
p.add_argument("--out", default="runs/benchmark")
a = p.parse_args()
out = pathlib.Path(a.out)
out.mkdir(parents=True, exist_ok=True)
subprocess.run(["python3", "scripts/make_fixture.py"], check=True)
for repeat in range(a.repeats):
    # Alternate order to reduce, but not eliminate, order/thermal effects.
    for policy in ["latest", "fifo"] if repeat % 2 == 0 else ["fifo", "latest"]:
        dest = out / f"{policy}-{repeat + 1}"
        subprocess.run(
            [
                a.binary,
                "--input",
                "runs/fixture",
                "--out",
                str(dest),
                "--policy",
                policy,
                "--capacity",
                "4",
                "--frames",
                "240",
                "--fps",
                "30",
                "--queries",
                "examples/queries.tsv",
                "--response-delay-ms",
                "400",
            ],
            check=True,
        )
(out / "environment.json").write_text(
    json.dumps(
        {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "cpu": subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
            ).strip()
            if platform.system() == "Darwin"
            else platform.processor(),
            "date": datetime.date.today().isoformat(),
            "repeats": a.repeats,
            "workload": "8 s transformed CC0 photograph at 30 fps; not a human-interaction dataset",
            "comparison": "same model, same frame input, 2 CPU threads, queue capacity 4; latest evicts oldest; FIFO rejects newest",
            "response_worker": "template renderer with injected 400 ms delay, not LLM inference",
        },
        indent=2,
    )
)
