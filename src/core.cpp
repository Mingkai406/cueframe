#include "cueframe/core.hpp"
#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>

namespace cueframe {
float iou(const Box &a, const Box &b) {
  float w = std::max(0.f, std::min(a.x + a.w, b.x + b.w) - std::max(a.x, b.x));
  float h = std::max(0.f, std::min(a.y + a.h, b.y + b.h) - std::max(a.y, b.y));
  float u = a.w * a.h + b.w * b.h - w * h;
  return u > 0 ? w * h / u : 0;
}
EvidenceStore::EvidenceStore(std::size_t n, double f) : limit_(n), freshness_(f) {
  if (!n || !std::isfinite(f) || f <= 0)
    throw std::invalid_argument("invalid evidence limits");
}
Snapshot EvidenceStore::observe(std::uint64_t frame, double time,
                                const std::vector<Detection> &ds) {
  std::lock_guard lock(mu_);
  if (!std::isfinite(time) || time < 0)
    throw std::invalid_argument("invalid timestamp");
  if (!history_.empty() && (frame <= history_.back().frame || time < history_.back().media_ms))
    throw std::invalid_argument("out-of-order observation; reset before a new source");
  if (ds.size() > 256)
    throw std::invalid_argument("at most 256 detections per observation");
  for (const auto &d : ds) {
    auto b = d.box;
    if (!std::isfinite(b.x + b.y + b.w + b.h + d.confidence) || b.x < 0 || b.y < 0 || b.w <= 0 ||
        b.h <= 0 || b.x + b.w > 1.001f || b.y + b.h > 1.001f || d.confidence < 0 ||
        d.confidence > 1)
      throw std::invalid_argument("invalid detection");
  }
  std::vector<bool> matched(tracks_.size(), false);
  auto old_size = tracks_.size();
  // Greedy same-class IoU is intentionally conservative; it is not re-identification.
  for (const auto &d : ds) {
    std::size_t best = old_size;
    float overlap = .25f;
    for (std::size_t i = 0; i < old_size; ++i) {
      if (matched[i] || tracks_[i].detection.class_id != d.class_id ||
          time - tracks_[i].seen_ms > 1500)
        continue;
      auto score = iou(tracks_[i].detection.box, d.box);
      if (score > overlap) {
        overlap = score;
        best = i;
      }
    }
    if (best == old_size)
      tracks_.push_back({next_id_++, 1, d, time, true});
    else {
      auto &t = tracks_[best];
      auto a = t.detection.box;
      auto b = d.box;
      // A change in visibility or a material box change invalidates current-location answers.
      if (!t.visible || std::max({std::abs(a.x - b.x), std::abs(a.y - b.y), std::abs(a.w - b.w),
                                  std::abs(a.h - b.h)}) > .025f)
        ++t.revision;
      t.detection = d;
      t.seen_ms = time;
      t.visible = true;
      matched[best] = true;
    }
  }
  for (std::size_t i = 0; i < old_size; ++i)
    if (!matched[i] && tracks_[i].visible) {
      tracks_[i].visible = false;
      ++tracks_[i].revision;
    }
  std::erase_if(tracks_, [&](const Track &t) { return time - t.seen_ms > 1500; });
  while (tracks_.size() > 512) {
    auto oldest = std::min_element(tracks_.begin(), tracks_.end(),
                                   [](auto &a, auto &b) { return a.seen_ms < b.seen_ms; });
    tracks_.erase(oldest);
  }
  Snapshot s{epoch_, frame, time, tracks_};
  history_.push_back(s);
  while (history_.size() > limit_)
    history_.pop_front();
  return s;
}
Ticket EvidenceStore::query(const std::string &text) {
  std::lock_guard lock(mu_);
  Ticket t;
  t.epoch = epoch_;
  t.turn = ++turn_;
  if (history_.empty()) {
    t.reason = "no_observation";
    return t;
  }
  std::istringstream in(text);
  std::string selector, label;
  in >> selector;
  const Snapshot *s = &history_.back();
  float px = 0, py = 0;
  if (selector == "previous") {
    t.historical = true;
    std::getline(in, label);
    if (!label.empty())
      label.erase(0, 1);
    // Previous means the newest older retained snapshot where that label was visible.
    s = nullptr;
    for (auto it = std::next(history_.rbegin()); it != history_.rend(); ++it) {
      if (std::any_of(it->tracks.begin(), it->tracks.end(),
                      [&](auto &v) { return v.visible && v.detection.label == label; })) {
        s = &*it;
        break;
      }
    }
    if (!s) {
      t.reason = "no_retained_history";
      return t;
    }
  } else if (selector == "left" || selector == "right") {
    std::getline(in, label);
    if (!label.empty())
      label.erase(0, 1);
  } else if (selector == "at") {
    std::string extra;
    if (!(in >> px >> py) || (in >> extra) || !std::isfinite(px + py) || px < 0 || px > 1 ||
        py < 0 || py > 1) {
      t.reason = "invalid_point";
      return t;
    }
  } else
    label = text;
  std::vector<Track> candidates;
  for (const auto &v : s->tracks) {
    if (!v.visible)
      continue;
    auto b = v.detection.box;
    if (selector == "at" ? (px >= b.x && px <= b.x + b.w && py >= b.y && py <= b.y + b.h)
                         : v.detection.label == label)
      candidates.push_back(v);
  }
  if (candidates.empty()) {
    t.reason = "not_observed";
    return t;
  }
  if (selector == "left" || selector == "right") {
    std::sort(candidates.begin(), candidates.end(), [](auto &a, auto &b) {
      return a.detection.box.x + a.detection.box.w / 2 < b.detection.box.x + b.detection.box.w / 2;
    });
    if (selector == "right")
      std::reverse(candidates.begin(), candidates.end());
    if (candidates.size() > 1 &&
        std::abs(candidates[0].detection.box.x + candidates[0].detection.box.w / 2 -
                 candidates[1].detection.box.x - candidates[1].detection.box.w / 2) < .03f) {
      t.reason = "ambiguous_reference";
      return t;
    }
  } else if (candidates.size() != 1) {
    t.reason = "ambiguous_reference";
    return t;
  }
  const auto &chosen = candidates.front();
  t.target = chosen.id;
  t.revision = chosen.revision;
  t.evidence_ms = s->media_ms;
  t.label = chosen.detection.label;
  t.status = "ready";
  t.reason = "evidence_selected";
  t.evidence_box = chosen.detection.box;
  return t;
}
Decision EvidenceStore::commit(const Ticket &t, double now) const {
  std::lock_guard lock(mu_);
  if (t.epoch != epoch_)
    return {"suppressed", "source_reset", ""};
  if (t.turn != turn_)
    return {"suppressed", "superseded_turn", ""};
  if (t.status != "ready")
    return {"clarify", t.reason, "Please specify a visible object or a less ambiguous reference."};
  if (t.historical)
    return {"published", "historical_evidence",
            "In the retained observation at " + std::to_string(static_cast<int>(t.evidence_ms)) +
                " ms, I observed a " + t.label + "."};
  if (!std::isfinite(now) || now < t.evidence_ms || now - t.evidence_ms > freshness_)
    return {"suppressed", "evidence_too_old", ""};
  auto it = std::find_if(tracks_.begin(), tracks_.end(), [&](auto &v) { return v.id == t.target; });
  if (it == tracks_.end() || !it->visible)
    return {"suppressed", "target_unobserved", ""};
  if (it->revision != t.revision)
    return {"suppressed", "target_changed", ""};
  auto a = t.evidence_box, b = it->detection.box;
  if (std::max({std::abs(a.x - b.x), std::abs(a.y - b.y), std::abs(a.w - b.w),
                std::abs(a.h - b.h)}) > .025f)
    return {"suppressed", "target_changed", ""};
  return {"published", "current_evidence", "The selected object is a " + t.label + "."};
}
Snapshot EvidenceStore::latest() const {
  std::lock_guard lock(mu_);
  return history_.empty() ? Snapshot{epoch_, 0, 0, {}} : history_.back();
}
void EvidenceStore::reset() {
  std::lock_guard lock(mu_);
  ++epoch_;
  ++turn_;
  tracks_.clear();
  history_.clear();
}
std::size_t EvidenceStore::history_size() const {
  std::lock_guard lock(mu_);
  return history_.size();
}
std::string json_escape(const std::string &s) {
  std::ostringstream o;
  for (unsigned char c : s) {
    switch (c) {
    case '"':
      o << "\\\"";
      break;
    case '\\':
      o << "\\\\";
      break;
    case '\n':
      o << "\\n";
      break;
    case '\r':
      o << "\\r";
      break;
    case '\t':
      o << "\\t";
      break;
    default:
      if (c < 32)
        o << "\\u" << std::hex << std::setw(4) << std::setfill('0') << int(c);
      else
        o << c;
    }
  }
  return o.str();
}
} // namespace cueframe
