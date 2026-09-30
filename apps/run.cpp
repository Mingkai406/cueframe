#include "cueframe/detector.hpp"
#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <sstream>
#include <sys/resource.h>
#include <thread>
namespace fs = std::filesystem;
using namespace cueframe;
using Clock = std::chrono::steady_clock;
double elapsed(Clock::time_point t) {
  return std::chrono::duration<double, std::milli>(Clock::now() - t).count();
}
struct Frame {
  std::uint64_t id;
  double media_ms;
  Clock::time_point acquired;
  cv::Mat image;
};
struct Query {
  double at_ms;
  std::string text;
};
struct Work {
  Ticket ticket;
  std::string query;
};
std::string quote(const std::string &s) { return "\"" + json_escape(s) + "\""; }
int main(int argc, char **argv) {
  try {
    std::string input, model = "models/yolox_s.onnx", output = "runs/demo", queryfile,
                       policy = "latest";
    int camera = -1, frames = 120, capacity = 2, delay = 400, threads = 2;
    double fps = 15;
    bool realtime = true, interactive = false;
    for (int i = 1; i < argc; ++i) {
      std::string a = argv[i];
      auto value = [&]() {
        if (i + 1 == argc)
          throw std::invalid_argument("missing value for " + a);
        return std::string(argv[++i]);
      };
      if (a == "--input")
        input = value();
      else if (a == "--model")
        model = value();
      else if (a == "--out")
        output = value();
      else if (a == "--queries")
        queryfile = value();
      else if (a == "--policy")
        policy = value();
      else if (a == "--camera")
        camera = std::stoi(value());
      else if (a == "--frames")
        frames = std::stoi(value());
      else if (a == "--capacity")
        capacity = std::stoi(value());
      else if (a == "--response-delay-ms")
        delay = std::stoi(value());
      else if (a == "--threads")
        threads = std::stoi(value());
      else if (a == "--fps")
        fps = std::stod(value());
      else if (a == "--offline")
        realtime = false;
      else if (a == "--interactive")
        interactive = true;
      else if (a == "--help") {
        std::cout
            << "CueFrame: local visual evidence runtime\n--input IMAGE|VIDEO|IMAGE_DIRECTORY or "
               "--camera INDEX\n--model ONNX --out DIRECTORY --frames 120 --fps 15 --capacity "
               "2\n--policy latest|fifo --queries FILE.tsv --response-delay-ms 400 --threads "
               "2\n--offline (unpaced throughput run) --interactive (query stdin; /quit to stop)\n";
        return 0;
      } else
        throw std::invalid_argument("unknown option: " + a);
    }
    if ((input.empty()) == (camera < 0))
      throw std::invalid_argument("choose exactly one --input or --camera");
    if (frames < 1 || capacity < 1 || capacity > 128 || delay < 0 || delay > 10000 || threads < 1 ||
        fps <= 0 || fps > 240 || !std::isfinite(fps))
      throw std::invalid_argument("invalid runtime bounds");
    if (policy != "latest" && policy != "fifo")
      throw std::invalid_argument("policy must be latest or fifo");
    std::vector<Query> queries;
    if (!queryfile.empty()) {
      std::ifstream f(queryfile);
      if (!f)
        throw std::runtime_error("cannot open queries");
      std::string line;
      while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#')
          continue;
        auto tab = line.find('\t');
        if (tab == std::string::npos)
          throw std::invalid_argument("queries must be time_ms TAB query");
        double t = std::stod(line.substr(0, tab));
        if (t < 0 || !std::isfinite(t) || (!queries.empty() && t < queries.back().at_ms))
          throw std::invalid_argument("queries must have ordered nonnegative times");
        queries.push_back({t, line.substr(tab + 1)});
      }
    }
    fs::create_directories(output);
    fs::create_directories(fs::path(output) / "frames");
    std::ofstream events(fs::path(output) / "observations.jsonl"),
        responses(fs::path(output) / "responses.jsonl"), timings(fs::path(output) / "timings.csv");
    if (!events || !responses || !timings)
      throw std::runtime_error("cannot create output files");
    timings << "frame,media_ms,queue_ms,preprocess_ms,inference_ms,postprocess_ms,observation_"
               "latency_ms,visible_tracks\n";
    cv::setNumThreads(threads);
    Detector detector(model);
    // Warm-up is excluded from steady-state measurements and recorded in the summary.
    auto warm = Clock::now();
    detector.run(cv::Mat(480, 640, CV_8UC3, cv::Scalar(114, 114, 114)));
    double warm_ms = elapsed(warm);
    std::vector<fs::path> files;
    cv::VideoCapture capture;
    cv::Mat still;
    if (camera >= 0) {
      if (!capture.open(camera))
        throw std::runtime_error("cannot open camera; check OS permission");
      capture.set(cv::CAP_PROP_FRAME_WIDTH, 640);
      capture.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
    } else if (fs::is_directory(input)) {
      for (auto &e : fs::directory_iterator(input)) {
        auto ext = e.path().extension();
        if (ext == ".png" || ext == ".jpg" || ext == ".jpeg")
          files.push_back(e.path());
      }
      std::sort(files.begin(), files.end());
      if (files.empty())
        throw std::runtime_error("empty image directory");
    } else {
      auto ext = fs::path(input).extension();
      if (ext == ".png" || ext == ".jpg" || ext == ".jpeg") {
        still = cv::imread(input);
        if (still.empty())
          throw std::runtime_error("cannot decode image");
      } else if (!capture.open(input))
        throw std::runtime_error("cannot open video");
    }
    BoundedQueue<Frame> queue(capacity,
                              policy == "latest" ? Overflow::drop_oldest : Overflow::drop_newest);
    BoundedQueue<Work> work(8);
    EvidenceStore evidence(96, 750);
    std::atomic<bool> stop = false, finished = false;
    std::atomic<int> produced = 0, processed = 0, published = 0, suppressed = 0, clarified = 0;
    std::atomic<double> source_ms = 0;
    std::exception_ptr failure;
    std::mutex fail_mu;
    auto started = Clock::now();
    auto current_time = [&]() { return realtime ? elapsed(started) : source_ms.load(); };
    auto fail = [&]() {
      std::lock_guard l(fail_mu);
      if (!failure)
        failure = std::current_exception();
      stop = true;
      queue.close();
    };
    std::thread responder([&] {
      try {
        while (auto w = work.pop()) {
          auto wait_start = Clock::now();
          while (elapsed(wait_start) < delay) {
            std::this_thread::sleep_for(std::chrono::milliseconds(std::min(20, delay)));
            auto check = evidence.commit(w->ticket, current_time());
            if (check.reason == "superseded_turn" || check.reason == "source_reset")
              break;
          }
          auto d = evidence.commit(w->ticket, current_time());
          if (d.status == "published")
            ++published;
          else if (d.status == "suppressed")
            ++suppressed;
          else
            ++clarified;
          responses << "{\"turn\":" << w->ticket.turn << ",\"query\":" << quote(w->query)
                    << ",\"evidence_ms\":" << w->ticket.evidence_ms
                    << ",\"finished_ms\":" << current_time()
                    << ",\"historical\":" << (w->ticket.historical ? "true" : "false")
                    << ",\"status\":" << quote(d.status) << ",\"reason\":" << quote(d.reason)
                    << ",\"text\":" << quote(d.text) << "}\n";
          responses.flush();
          std::cout << "[" << d.status << "] " << w->query << " -> " << d.reason;
          if (!d.text.empty())
            std::cout << ": " << d.text;
          std::cout << '\n';
        }
      } catch (...) {
        fail();
      }
    });
    std::thread consumer([&] {
      try {
        std::size_t qi = 0;
        while (auto frame = queue.pop()) {
          auto inference_start = Clock::now();
          double queue_ms =
              std::chrono::duration<double, std::milli>(inference_start - frame->acquired).count();
          auto result = detector.run(frame->image);
          auto s = evidence.observe(frame->id, frame->media_ms, result.detections);
          int visible = 0;
          for (const auto &t : s.tracks)
            if (t.visible)
              ++visible;
          double observation_latency = elapsed(frame->acquired);
          timings << frame->id << ',' << frame->media_ms << ',' << queue_ms << ','
                  << result.preprocess_ms << ',' << result.inference_ms << ','
                  << result.postprocess_ms << ',' << observation_latency << ',' << visible << '\n';
          cv::Mat preview = frame->image.clone();
          events << "{\"frame\":" << s.frame << ",\"media_ms\":" << s.media_ms << ",\"tracks\":[";
          bool first = true;
          for (const auto &t : s.tracks) {
            if (!first)
              events << ',';
            first = false;
            auto b = t.detection.box;
            events << "{\"id\":" << t.id << ",\"revision\":" << t.revision
                   << ",\"label\":" << quote(t.detection.label)
                   << ",\"confidence\":" << t.detection.confidence
                   << ",\"visible\":" << (t.visible ? "true" : "false") << ",\"box\":[" << b.x
                   << ',' << b.y << ',' << b.w << ',' << b.h << "]}";
            if (t.visible) {
              cv::Rect r(int(b.x * preview.cols), int(b.y * preview.rows), int(b.w * preview.cols),
                         int(b.h * preview.rows));
              cv::rectangle(preview, r, cv::Scalar(150, 135, 15), 2);
              cv::putText(preview, t.detection.label + " #" + std::to_string(t.id),
                          cv::Point(r.x, std::max(16, r.y - 6)), cv::FONT_HERSHEY_SIMPLEX, .55,
                          cv::Scalar(150, 135, 15), 2);
            }
          }
          events << "]}\n";
          auto filename = "frames/" + std::to_string(s.frame) + ".jpg";
          cv::imwrite((fs::path(output) / filename).string(), preview);
          ++processed;
          while (qi < queries.size() && queries[qi].at_ms <= s.media_ms) {
            auto q = queries[qi++];
            work.push({evidence.query(q.text), q.text});
          }
        }
        if (qi < queries.size())
          std::cerr << "Warning: " << queries.size() - qi
                    << " scheduled queries are beyond the observed input.\n";
      } catch (...) {
        fail();
      }
      finished = true;
    });
    std::thread producer([&] {
      try {
        for (int n = 0; n < frames && !stop; ++n) {
          double media = n * 1000.0 / fps;
          if (realtime)
            std::this_thread::sleep_until(started +
                                          std::chrono::duration_cast<Clock::duration>(
                                              std::chrono::duration<double, std::milli>(media)));
          auto acquired = Clock::now();
          cv::Mat image;
          if (!files.empty()) {
            if (std::size_t(n) >= files.size())
              break;
            image = cv::imread(files[n].string());
          } else if (!still.empty())
            image = still.clone();
          else if (!capture.read(image))
            break;
          if (image.empty())
            throw std::runtime_error("frame decode failed");
          if (image.cols > 1280 || image.rows > 1280) {
            double scale = 1280.0 / std::max(image.cols, image.rows);
            cv::resize(image, image, cv::Size(), scale, scale);
          }
          // Camera observations use the acquisition clock. Files use explicit constant-rate replay.
          if (camera >= 0)
            media = elapsed(started);
          source_ms = media;
          queue.push({static_cast<std::uint64_t>(n), media, acquired, std::move(image)});
          ++produced;
        }
      } catch (...) {
        fail();
      }
      queue.close();
    });
    if (interactive) {
      std::cout << "Commands: cup | left cup | right cup | previous cup | at 0.5 0.5 | /quit\n";
      std::string line;
      while (!finished && std::getline(std::cin, line)) {
        if (line == "/quit")
          break;
        if (!line.empty())
          work.push({evidence.query(line), line});
      }
      stop = true;
      queue.close();
    }
    producer.join();
    consumer.join();
    work.close();
    responder.join();
    capture.release();
    if (failure)
      std::rethrow_exception(failure);
    if (processed == 0)
      throw std::runtime_error("no frames processed");
    rusage usage{};
    getrusage(RUSAGE_SELF, &usage);
#ifdef __APPLE__
    double peak_mb = usage.ru_maxrss / 1024.0 / 1024.0;
#else
    double peak_mb = usage.ru_maxrss / 1024.0;
#endif
    std::ofstream summary(fs::path(output) / "summary.json");
    summary << "{\"version\":\"0.1.0\",\"opencv\":" << quote(CV_VERSION)
            << ",\"model\":\"YOLOX-s FP32 / OpenCV CPU\",\"policy\":" << quote(policy)
            << ",\"capacity\":" << capacity << ",\"source_fps\":" << fps
            << ",\"paced\":" << (realtime ? "true" : "false") << ",\"threads\":" << threads
            << ",\"produced\":" << produced << ",\"processed\":" << processed
            << ",\"dropped\":" << queue.dropped() << ",\"queue_high_water\":" << queue.high_water()
            << ",\"warmup_ms\":" << warm_ms << ",\"peak_rss_mb\":" << peak_mb
            << ",\"wall_ms\":" << elapsed(started) << ",\"response_delay_ms\":" << delay
            << ",\"published\":" << published << ",\"suppressed\":" << suppressed
            << ",\"clarified\":" << clarified << ",\"dropped_queries\":" << work.dropped() << "}\n";
    std::cout << "Processed " << processed << " / " << produced << " frames; dropped "
              << queue.dropped() << "; peak RSS " << peak_mb << " MiB. Output: " << output << '\n';
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "CueFrame: " << e.what() << '\n';
    return 1;
  }
}
