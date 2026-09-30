"use strict";
const $ = (id) => document.getElementById(id);
let trace = null,
  timer = null,
  time = 0;
function stop() {
  clearInterval(timer);
  timer = null;
  $("play").textContent = "Play";
  $("play").setAttribute("aria-label", "Play recorded trace");
}
function render(t) {
  time = t;
  $("scrubber").value = t;
  $("time").textContent = (t / 1000).toFixed(2) + " s";
  const past = trace.observations.filter((o) => o.media_ms <= t);
  const o = past.length ? past[past.length - 1] : trace.observations[0];
  $("scene").src = o.image;
  $("scene").alt =
    `Actual detector output at ${(o.media_ms / 1000).toFixed(2)} seconds`;
  $("frame-label").textContent =
    `FRAME ${o.frame} · ${(o.media_ms / 1000).toFixed(2)} s`;
  $("tracks").replaceChildren();
  const visible = o.tracks.filter((v) => v.visible);
  if (!visible.length) {
    const p = document.createElement("p");
    p.className = "empty";
    p.textContent = "No objects detected in this frame.";
    $("tracks").append(p);
  }
  visible.forEach((v) => {
    const el = document.createElement("div");
    el.className = "track";
    const name = document.createElement("strong");
    name.textContent = v.label;
    const meta = document.createElement("span");
    meta.textContent = `#${v.id} · ${(v.confidence * 100).toFixed(0)}%`;
    el.append(name, meta);
    $("tracks").append(el);
  });
  [...$("responses").children].forEach((el, i) => {
    const r = trace.responses[i];
    el.classList.toggle("future", r.finished_ms > t);
    el.classList.toggle(
      "active",
      r.finished_ms <= t && r.finished_ms > t - 900,
    );
  });
}
Promise.all([
  fetch("data/trace.json").then((r) => {
    if (!r.ok) throw Error("trace unavailable");
    return r.json();
  }),
  fetch("data/metrics.json").then((r) => r.json()),
])
  .then(([tr, m]) => {
    trace = tr;
    tr.responses.forEach((r) => {
      const el = document.createElement("article");
      el.className = `response ${r.status}`;
      const status = document.createElement("div");
      status.className = "status";
      status.textContent = `${r.status} / ${(r.finished_ms / 1000).toFixed(2)} s`;
      const q = document.createElement("strong");
      q.textContent = "“" + r.query + "”";
      const p = document.createElement("p");
      p.textContent = r.reason.replaceAll("_", " ");
      el.append(status, q, p);
      el.title = r.text || r.reason;
      $("responses").append(el);
    });
    const entries = [
      [
        "LATEST / MEDIAN",
        m.latest.latency_p50_ms.toFixed(0) + " ms",
        "Acquisition to observation",
      ],
      [
        "FIFO / MEDIAN",
        m.fifo.latency_p50_ms.toFixed(0) + " ms",
        "Same bounded capacity",
      ],
      ["REPEATED RUNS", "3 + 3", "Measured locally · source traces included"],
    ];
    entries.forEach(([label, value, note]) => {
      const el = document.createElement("div");
      el.className = "number";
      for (const [tag, text] of [
        ["span", label],
        ["b", value],
        ["small", note],
      ]) {
        const part = document.createElement(tag);
        part.textContent = text;
        el.append(part);
      }
      $("numbers").append(el);
    });
    $("scrubber").addEventListener("input", (e) => {
      stop();
      render(Number(e.target.value));
    });
    $("play").addEventListener("click", () => {
      if (timer) {
        stop();
        return;
      }
      if (time >= 8000) render(0);
      $("play").textContent = "Pause";
      $("play").setAttribute("aria-label", "Pause recorded trace");
      timer = setInterval(() => {
        render(Math.min(8000, time + 50));
        if (time >= 8000) stop();
      }, 50);
    });
    render(0);
  })
  .catch((e) => {
    $("frame-label").textContent =
      "Trace unavailable — serve this directory over HTTP.";
    console.error(e);
  });
