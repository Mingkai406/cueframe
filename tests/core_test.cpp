#include "cueframe/core.hpp"
#include <atomic>
#include <functional>
#include <iostream>
#include <thread>
using namespace cueframe;
void require(bool v) {
  if (!v)
    throw std::runtime_error("contract failed");
}
Detection cup(float x = .1f) { return {41, "cup", .9f, {x, .1f, .2f, .2f}}; }
template <class F> void rejects(F f) {
  bool hit = false;
  try {
    f();
  } catch (const std::invalid_argument &) {
    hit = true;
  }
  require(hit);
}
int main() {
  int count = 0;
  auto test = [&](const char *name, std::function<void()> f) {
    try {
      f();
      ++count;
      std::cout << "PASS " << name << '\n';
    } catch (const std::exception &e) {
      std::cerr << "FAIL " << name << ": " << e.what() << '\n';
      std::exit(1);
    }
  };
  test("latest queue drops oldest and drains after close", [] {
    BoundedQueue<int> q(2);
    q.push(1);
    q.push(2);
    q.push(3);
    q.close();
    require(*q.pop() == 2);
    require(*q.pop() == 3);
    require(!q.pop());
    require(!q.push(4));
    require(q.dropped() == 1 && q.high_water() == 2);
  });
  test("FIFO bounded queue rejects newest", [] {
    BoundedQueue<int> q(1, Overflow::drop_newest);
    q.push(1);
    require(!q.push(2));
    q.close();
    require(*q.pop() == 1 && q.dropped() == 1);
  });
  test("close wakes empty consumer", [] {
    BoundedQueue<int> q(2);
    std::thread t([&] { require(!q.pop()); });
    q.close();
    t.join();
  });
  test("queue concurrency preserves order with bounded capacity", [] {
    BoundedQueue<int> q(4);
    std::atomic<int> consumed = 0;
    std::thread t([&] {
      int last = -1;
      while (auto x = q.pop()) {
        require(*x > last);
        last = *x;
        ++consumed;
      }
    });
    for (int i = 0; i < 10000; ++i)
      q.push(i);
    q.close();
    t.join();
    require(consumed + q.dropped() == 10000 && q.high_water() <= 4);
  });
  test("current evidence publishes", [] {
    EvidenceStore s;
    s.observe(1, 0, {cup()});
    auto t = s.query("cup");
    require(s.commit(t, 10).status == "published");
  });
  test("movement suppresses delayed response", [] {
    EvidenceStore s;
    s.observe(1, 0, {cup()});
    auto t = s.query("cup");
    s.observe(2, 100, {cup(.17f)});
    require(s.commit(t, 100).reason == "target_changed");
  });
  test("small cumulative movements also invalidate", [] {
    EvidenceStore s;
    s.observe(1, 0, {cup()});
    auto t = s.query("cup");
    s.observe(2, 20, {cup(.12f)});
    s.observe(3, 40, {cup(.14f)});
    require(s.commit(t, 40).reason == "target_changed");
  });
  test("unrelated object movement does not invalidate target", [] {
    EvidenceStore s;
    auto b = Detection{73, "book", .8, {.7, .6, .1, .1}};
    s.observe(1, 0, {cup(), b});
    auto t = s.query("cup");
    b.box.x = .8;
    s.observe(2, 100, {cup(), b});
    require(s.commit(t, 100).status == "published");
  });
  test("miss means unobserved, not a semantic removal claim", [] {
    EvidenceStore s;
    s.observe(1, 0, {cup()});
    auto t = s.query("cup");
    s.observe(2, 100, {});
    require(s.commit(t, 100).reason == "target_unobserved");
    require(s.query("cup").reason == "not_observed");
  });
  test("retained history survives current missing evidence", [] {
    EvidenceStore s;
    s.observe(1, 0, {cup()});
    s.observe(2, 100, {});
    auto t = s.query("previous cup");
    require(t.historical && t.evidence_ms == 0);
    require(s.commit(t, 9000).reason == "historical_evidence");
  });
  test("new turn suppresses older response", [] {
    EvidenceStore s;
    s.observe(1, 0, {cup()});
    auto a = s.query("cup");
    auto b = s.query("left cup");
    require(s.commit(a, 10).reason == "superseded_turn");
    require(s.commit(b, 10).status == "published");
  });
  test("source epoch invalidates historical tickets", [] {
    EvidenceStore s;
    s.observe(1, 0, {cup()});
    s.observe(2, 10, {});
    auto a = s.query("previous cup");
    s.reset();
    require(s.commit(a, 20).reason == "source_reset");
    require(s.history_size() == 0);
    s.observe(1, 0, {cup()});
  });
  test("ambiguity requests clarification", [] {
    EvidenceStore s;
    s.observe(1, 0, {cup(), cup(.6)});
    require(s.query("cup").reason == "ambiguous_reference");
    require(s.query("left cup").target != s.query("right cup").target);
  });
  test("point reference selects bounding box", [] {
    EvidenceStore s;
    s.observe(1, 0, {cup(), cup(.6)});
    require(s.query("at 0.15 0.15").status == "ready");
    require(s.query("at 2 3").reason == "invalid_point");
    require(s.query("at .1 .1 extra").reason == "invalid_point");
  });
  test("freshness expires even if inference stops", [] {
    EvidenceStore s;
    s.observe(1, 0, {cup()});
    auto t = s.query("cup");
    require(s.commit(t, 751).reason == "evidence_too_old");
  });
  test("history eviction is bounded", [] {
    EvidenceStore s(3);
    for (int i = 0; i < 30; ++i)
      s.observe(i, i * 10, i == 0 ? std::vector<Detection>{cup()} : std::vector<Detection>{});
    require(s.history_size() == 3);
    require(s.query("previous cup").reason == "no_retained_history");
  });
  test("reject invalid timestamps, boxes and capacities", [] {
    rejects([] { BoundedQueue<int> q(0); });
    EvidenceStore s;
    rejects([&] { s.observe(1, -1, {}); });
    s.observe(1, 10, {});
    rejects([&] { s.observe(1, 20, {}); });
    rejects([&] { s.observe(2, 9, {}); });
    auto d = cup();
    d.box.w = -1;
    rejects([&] { s.observe(2, 20, {d}); });
  });
  test("JSON escaping", [] { require(json_escape("a\n\"\\") == "a\\n\\\"\\\\"); });
  std::cout << count << " contracts passed\n";
}
