#pragma once
#include <algorithm>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace cueframe {
enum class Overflow { drop_oldest, drop_newest };
// A single ownership transfer per item. Closing wakes all waiters and drains survivors.
template <class T> class BoundedQueue {
public:
  explicit BoundedQueue(std::size_t capacity, Overflow policy = Overflow::drop_oldest)
      : capacity_(capacity), policy_(policy) {
    if (!capacity)
      throw std::invalid_argument("queue capacity must be positive");
  }
  bool push(T item) {
    std::lock_guard lock(mu_);
    if (closed_)
      return false;
    if (items_.size() == capacity_) {
      ++dropped_;
      if (policy_ == Overflow::drop_newest)
        return false;
      items_.pop_front();
    }
    items_.push_back(std::move(item));
    high_water_ = std::max(high_water_, items_.size());
    cv_.notify_one();
    return true;
  }
  std::optional<T> pop() {
    std::unique_lock lock(mu_);
    cv_.wait(lock, [&] { return closed_ || !items_.empty(); });
    if (items_.empty())
      return std::nullopt;
    T value = std::move(items_.front());
    items_.pop_front();
    return value;
  }
  void close() {
    std::lock_guard lock(mu_);
    closed_ = true;
    cv_.notify_all();
  }
  std::size_t dropped() const {
    std::lock_guard lock(mu_);
    return dropped_;
  }
  std::size_t high_water() const {
    std::lock_guard lock(mu_);
    return high_water_;
  }

private:
  const std::size_t capacity_;
  const Overflow policy_;
  mutable std::mutex mu_;
  std::condition_variable cv_;
  std::deque<T> items_;
  bool closed_ = false;
  std::size_t dropped_ = 0, high_water_ = 0;
};

struct Box {
  float x = 0, y = 0, w = 0, h = 0;
}; // Normalized image coordinates.
float iou(const Box &, const Box &);
struct Detection {
  int class_id = 0;
  std::string label;
  float confidence = 0;
  Box box;
};
struct Track {
  std::uint64_t id = 0, revision = 0;
  Detection detection;
  double seen_ms = 0;
  bool visible = false;
};
struct Snapshot {
  std::uint64_t epoch = 0, frame = 0;
  double media_ms = 0;
  std::vector<Track> tracks;
};
struct Ticket {
  std::uint64_t epoch = 0, turn = 0, target = 0, revision = 0;
  double evidence_ms = 0;
  bool historical = false;
  std::string status = "clarify", reason, label;
  Box evidence_box;
};
struct Decision {
  std::string status, reason, text;
};

// All reads and mutations are serialized. Snapshot/ticket values own their evidence.
class EvidenceStore {
public:
  explicit EvidenceStore(std::size_t history_limit = 96, double freshness_ms = 750);
  Snapshot observe(std::uint64_t frame, double media_ms, const std::vector<Detection> &);
  // Deliberately finite grammar: left LABEL | right LABEL | previous LABEL |
  // LABEL | at X Y. Corrections are new turns, not free-form NLP.
  Ticket query(const std::string &text);
  Decision commit(const Ticket &, double current_media_ms) const;
  Snapshot latest() const;
  void reset();
  std::size_t history_size() const;

private:
  mutable std::mutex mu_;
  std::size_t limit_;
  double freshness_;
  std::uint64_t epoch_ = 1, next_id_ = 1, turn_ = 0;
  std::deque<Snapshot> history_;
  std::vector<Track> tracks_;
};
std::string json_escape(const std::string &);
} // namespace cueframe
