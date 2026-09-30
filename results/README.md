# Reference run provenance

The committed `reference/` traces were generated locally on 2026-09-29, using the C++ executable, `scripts/benchmark.py`, and the controlled fixture in `scripts/make_fixture.py`.

Each policy has three independent process executions. Both use capacity 4, YOLOX-s FP32, two CPU threads, a 30 fps source, and the same 240 transformed-photo frames. Queue policies are run sequentially and in alternating order. The 400 ms response delay is deliberately injected into a template worker. It is not the detector's inference latency and not a measured LLM latency.

`metrics.json` pools processed-frame latencies for descriptive medians/percentiles; it does not treat correlated frames as independent experimental subjects. The first inference warm-up is separate. `summary.json` reports peak process RSS, which includes the model and runtime. Detector confidence is not interaction accuracy.

Raw observations and responses are retained to audit the figures. Only public fixture imagery is published. The interactive site includes a subset of actual detection frames from `latest-1`.
