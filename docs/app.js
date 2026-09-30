"use strict";
const $ = (id) => document.getElementById(id);
let trace = null,
  timer = null,
  time = 900;
function stop() {
  clearInterval(timer);
  timer = null;
  $("play").textContent = "Play";
  $("play").setAttribute("aria-label", "Play recorded trace");
}
function seek(t) {
  stop();
  render(t);
}
function render(t) {
  time = t;
  $("scrubber").value = t;
  $("time").textContent = (t / 1000).toFixed(2) + " s";
  const past = trace.observations.filter((o) => o.media_ms <= t);
  const o = past.length ? past[past.length - 1] : trace.observations[0];
  $("scene").src = o.image;
  $("scene").alt =
    `Recorded detector output at ${(o.media_ms / 1000).toFixed(2)} seconds`;
  $("frame-label").textContent =
    `${(o.media_ms / 1000).toFixed(2)} s · frame ${o.frame}`;
  const visible = o.tracks.filter((v) => v.visible);
  $("object-count").textContent = `${visible.length} observed`;
  $("tracks").replaceChildren();
  visible.forEach((v) => {
    const el = document.createElement("div");
    el.className = "track";
    const label = document.createElement("strong");
    label.textContent = v.label;
    const meta = document.createElement("span");
    meta.textContent = `#${v.id} / ${(v.confidence * 100).toFixed(0)}%`;
    el.append(label, meta);
    $("tracks").append(el);
  });
  $("empty-scene").hidden = visible.length > 0;
  const finished = trace.responses.filter((r) => r.finished_ms <= t);
  const latest = finished[finished.length - 1];
  const retained = latest?.historical
    ? trace.retained_observations.find(
        (s) => Math.abs(s.media_ms - latest.evidence_ms) < 0.01,
      )
    : past
        .filter((s) => s.tracks.some((v) => v.visible && v.label === "cup"))
        .at(-1);
  const showRetained = retained && (!visible.length || latest?.historical);
  $("retained").hidden = !showRetained;
  if (showRetained) {
    $("retained-image").src = retained.image;
    $("retained-time").textContent =
      (retained.media_ms / 1000).toFixed(2) + " s";
    $("retained-image").alt =
      `Retained detector observation at ${(retained.media_ms / 1000).toFixed(2)} seconds`;
  }
  $("decision-status").textContent = latest
    ? latest.status.charAt(0).toUpperCase() + latest.status.slice(1)
    : "Waiting for a response";
  $("decision-status").dataset.status = latest?.status || "waiting";
  const explanations = {
    target_unobserved:
      "The target is no longer observed. The pending response was withheld.",
    superseded_turn:
      "A newer query replaced this turn. The earlier response was withheld.",
    evidence_too_old: "The observation expired before the response was ready.",
    ambiguous_reference:
      "Two cups are visible. The system asks which one you mean.",
  };
  $("decision-text").textContent = latest
    ? explanations[latest.reason] ||
      latest.text ||
      latest.reason.replaceAll("_", " ")
    : "Play the trace or select a recorded event.";
  $("decision-detail").textContent = latest
    ? `Query: “${latest.query}” · ${(latest.finished_ms / 1000).toFixed(2)} s`
    : "";
  [...$("responses").children].forEach((el, i) => {
    const r = trace.responses[i];
    el.classList.toggle("future", r.finished_ms > t);
    el.classList.toggle("current", latest === r);
    el.setAttribute("aria-pressed", String(latest === r));
  });
  document.querySelectorAll("[data-seek]").forEach((el) => {
    const target = Number(el.dataset.seek);
    const match =
      target === 900
        ? t < 3000
        : target === 4120
          ? t >= 3000 && t < 4600
          : t >= 4600 && t < 6100;
    el.setAttribute("aria-pressed", String(match));
  });
}
async function load() {
  const [tr, m] = await Promise.all(
    ["data/trace.json", "data/metrics.json"].map(async (url) => {
      const response = await fetch(url);
      if (!response.ok) throw new Error("Could not load " + url);
      return response.json();
    }),
  );
  trace = tr;
  tr.responses.forEach((r) => {
    const el = document.createElement("button");
    el.type = "button";
    el.className = "event";
    const stamp = document.createElement("time");
    stamp.textContent = (r.finished_ms / 1000).toFixed(2) + " s";
    const query = document.createElement("strong");
    query.textContent = "“" + r.query + "”";
    const status = document.createElement("span");
    status.textContent =
      r.status +
      " · " +
      (r.historical ? "history" : r.reason.replaceAll("_", " "));
    el.append(stamp, query, status);
    el.setAttribute(
      "aria-label",
      `${r.query}, ${r.status}, at ${(r.finished_ms / 1000).toFixed(2)} seconds`,
    );
    el.addEventListener("click", () =>
      seek(Math.ceil(r.finished_ms / 10) * 10),
    );
    $("responses").append(el);
  });
  $("metrics").replaceChildren();
  for (const [policy, label] of [
    ["latest", "Latest frame"],
    ["fifo", "Bounded FIFO"],
  ]) {
    const row = document.createElement("tr");
    const name = document.createElement("th");
    name.scope = "row";
    name.textContent = label;
    row.append(name);
    for (const key of ["latency_p50_ms", "latency_p95_ms"]) {
      const cell = document.createElement("td");
      cell.textContent = m[policy][key].toFixed(0) + " ms";
      row.append(cell);
    }
    $("metrics").append(row);
  }
  $("play").disabled = false;
  $("scrubber").disabled = false;
  $("scrubber").addEventListener("input", (e) => seek(Number(e.target.value)));
  document
    .querySelectorAll("[data-seek]")
    .forEach((el) =>
      el.addEventListener("click", () => seek(Number(el.dataset.seek))),
    );
  $("play").addEventListener("click", () => {
    if (timer) {
      stop();
      return;
    }
    if (time >= 8000) render(0);
    $("play").textContent = "Pause";
    $("play").setAttribute("aria-label", "Pause recorded trace");
    const started = performance.now() - time;
    timer = setInterval(() => {
      render(Math.min(8000, performance.now() - started));
      if (time >= 8000) stop();
    }, 50);
  });
  render(900);
}
load().catch((error) => {
  $("frame-label").textContent = "Recorded trace unavailable";
  $("decision-text").textContent =
    "Reload the page to try again. The complete run is available in the GitHub repository.";
  document
    .querySelectorAll("[data-seek]")
    .forEach((el) => (el.disabled = true));
  console.error(error);
});
$("copy-command").addEventListener("click", async () => {
  try {
    await navigator.clipboard.writeText($("run-command").textContent);
    $("copy-command").textContent = "Copied";
  } catch {
    $("copy-command").textContent = "Select the command to copy";
  }
  setTimeout(() => ($("copy-command").textContent = "Copy command"), 2500);
});
