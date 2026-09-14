"""Recording timing, annotations, layout, and interruption without MuJoCo or API calls."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import imageio.v2 as imageio
import numpy as np

from core.record.mujoco_recorder import CAMERAS, MujocoRecorder


class Scene:
    resolution = 128

    def __init__(self):
        self.data = SimpleNamespace(time=1.0)

    def render(self, camera):
        frame = np.zeros((self.resolution, self.resolution, 3), dtype=np.uint8)
        frame[:, :, CAMERAS.index(camera)] = round(160 + 20 * self.data.time)
        return frame

    def get_gripper_position(self):
        return np.array([0.04])


class RecordingTests(unittest.TestCase):
    def test_synchronized_motion_decisions_and_complete_log(self):
        with TemporaryDirectory() as directory:
            scene = Scene()
            recorder = MujocoRecorder(scene, directory, fps=10, decision_hold_s=0.2)
            output = json.dumps({"decision": "MV_UP", "reasoning": "Lift the cube. " * 100})
            recorder.begin_step(step_idx=0, stage="1/2 LIFT", token="MV_UP",
                                annotation="Lift the cube. " * 100, output=output)
            self.assertEqual(scene.data.time, 1.0)
            for time in (1.03, 1.1, 1.15, 1.2, 1.3):
                scene.data.time = time
                recorder.capture()
            recorder.end_step({"act": "MV_UP", "blocked": "z_floor"})
            recorder.begin_step(step_idx=1, stage="2/2 RELEASE", token="RELEASE",
                                annotation="Recovery opened the fingers.", source="recovery")
            recorder.gripper_command(False)
            recorder.end_step({"act": "RELEASE", "recover": True})
            recorder.finish(False, "max_steps_exceeded")
            recorder.close()  # Closing twice must preserve every file.
            self.assertEqual(scene.data.time, 1.3, "video pauses must not advance physics")

            events = [json.loads(line) for line in Path(directory, "annotations.jsonl").read_text().splitlines()]
            decision = next(e for e in events if e["event"] == "decision")
            self.assertEqual(events[0]["cameras"], ["side", "wrist", "front"])
            complete = next(e for e in events if e["event"] == "step_complete")
            self.assertEqual(decision["vlm_output"], output)
            self.assertEqual(complete["frame_start"], decision["frame"])
            self.assertEqual(complete["frame_end"], 7)
            self.assertIn("z-floor", complete["annotation"])
            self.assertEqual(complete["sim_time_s"], 0.3)
            self.assertEqual(events[-1]["frames"], recorder.frame_count)
            self.assertEqual(events[-2]["end_reason"], "max_steps_exceeded")
            self.assertTrue(any(e["source"] == "recovery" and e["event"] == "decision" for e in events))

            counts = []
            for name in (*CAMERAS, "combined"):
                with imageio.get_reader(recorder.paths[name]) as reader:
                    counts.append(reader.count_frames())
                    self.assertEqual(reader.get_meta_data()["fps"], 10)
                    first, last = reader.get_data(0), reader.get_data(recorder.frame_count - 1)
                    self.assertFalse(np.array_equal(first, last))
                    if name == "combined":
                        self.assertEqual(first.shape, (320, 256, 3))
                        # Camera quadrants retain their identity in the combined canvas.
                        for pixel, channel in ((first[96, 64], 0), (first[96, 192], 1), (first[256, 64], 2)):
                            self.assertEqual(int(np.argmax(pixel)), channel)
            self.assertEqual(counts, [recorder.frame_count] * 4)

    def test_interruption_keeps_frames_and_annotations(self):
        with TemporaryDirectory() as directory:
            recorder = MujocoRecorder(Scene(), directory, fps=10, decision_hold_s=0)
            recorder.begin_step(step_idx=3, stage="GRASP", token="GRASP", annotation="Close around the cube.")
            recorder.close()
            entries = [json.loads(line) for line in recorder.paths["annotations"].read_text().splitlines()]
            interrupted = next(e for e in entries if e["event"] == "step_interrupted")
            self.assertEqual(interrupted["decision"], "GRASP")
            self.assertEqual(interrupted["frame_end"], 2)
            for name in (*CAMERAS, "combined"):
                with imageio.get_reader(recorder.paths[name]) as reader:
                    self.assertEqual(reader.count_frames(), 2)

    def test_invalid_recording_settings_fail_before_opening_files(self):
        with TemporaryDirectory() as directory:
            for settings in ({"fps": 0}, {"fps": float("nan")}, {"decision_hold_s": -1}):
                with self.assertRaises(ValueError):
                    MujocoRecorder(Scene(), directory, **settings)
            self.assertEqual(list(Path(directory).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
