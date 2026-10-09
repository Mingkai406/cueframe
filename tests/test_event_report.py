import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from event_benchmark import evaluate


class EventTests(unittest.TestCase):
    def test_late_detection_is_not_timely_and_dropped_event_is_not_detector_failure(
        self,
    ):
        manifest = {
            "frames": 6,
            "fps": 10,
            "events": [
                {"start_frame": 0, "end_frame": 1},
                {"start_frame": 2, "end_frame": 3},
                {"start_frame": 4, "end_frame": 5},
            ],
        }
        summary = {
            "produced": 6,
            "processed": 2,
            "dropped": 4,
            "paced": True,
            "source_fps": 10,
        }
        times = [
            {
                "frame": 0,
                "media_ms": 0,
                "capture_lag_ms": 20,
                "observation_latency_ms": 100,
            },
            {
                "frame": 4,
                "media_ms": 400,
                "capture_lag_ms": 0,
                "observation_latency_ms": 10,
            },
        ]
        obs = [
            {"frame": 0, "tracks": [{"label": "cup", "visible": True}]},
            {"frame": 4, "tracks": []},
        ]
        r = evaluate(manifest, summary, times, obs)
        self.assertAlmostEqual(r["event_coverage"], 1 / 3)
        self.assertEqual(r["timely_event_coverage"], 0)
        self.assertEqual(r["events"][1]["miss_reason"], "queue_dropped_all")
        self.assertEqual(
            r["events"][2]["miss_reason"], "detector_miss_on_retained_frames"
        )
        with self.assertRaises(ValueError):
            evaluate(manifest, summary, times[:-1], obs)


if __name__ == "__main__":
    unittest.main()
