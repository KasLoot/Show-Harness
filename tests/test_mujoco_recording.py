"""Recording timing, annotations, layout, and interruption without MuJoCo or API calls."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import imageio.v2 as imageio
import numpy as np

from core.record.mujoco_recorder import CAMERAS, GLOBAL_CAMERA, MujocoRecorder


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


class PlugScene(Scene):
    cameras = ("side", "wrist", "front", "wrist_insert")
    colors = {
        "side": (220, 20, 20), "wrist": (20, 220, 20),
        "front": (20, 20, 220), "wrist_insert": (220, 220, 20),
    }

    def render(self, camera):
        return np.broadcast_to(np.array(self.colors[camera], np.uint8),
                               (self.resolution, self.resolution, 3)).copy()


class DepthPlugScene(PlugScene):
    cameras = (*PlugScene.cameras, "wrist_depth")
    colors = {**PlugScene.colors, "wrist_depth": (150, 150, 150)}


def recording_model():
    return SimpleNamespace(
        camera=Mock(return_value=SimpleNamespace(id=0)),
        vis=SimpleNamespace(global_=SimpleNamespace(offwidth=256, offheight=256)),
    )


class GlobalRenderer:
    """A separate render context whose pixels identify the captured simulation time."""

    def __init__(self, model, *, height, width):
        self.shape = (height, width, 3)
        self.model = model
        settings = model.vis.global_
        self.initial_buffer = (settings.offwidth, settings.offheight)
        self.captures = []
        self.close = Mock()

    def update_scene(self, data, *, camera):
        settings = self.model.vis.global_
        self.captures.append((data.time, camera, settings.offwidth, settings.offheight))

    @staticmethod
    def color(time):
        return np.array((round(time * 80), 45, 210), dtype=np.uint8)

    def render(self):
        return np.broadcast_to(self.color(self.captures[-1][0]), self.shape).copy()


class RecordingTests(unittest.TestCase):
    def test_global_recording_appends_canvas_and_stays_synchronized(self):
        for scene_type in (Scene, DepthPlugScene):
            with self.subTest(scene=scene_type.__name__), TemporaryDirectory() as directory:
                scene = scene_type()
                scene.model = recording_model()
                scene.cameras = tuple(getattr(scene, "cameras", CAMERAS))
                observation_cameras = scene.cameras
                original_render = scene.render
                camera_captures = []

                def render(camera):
                    camera_captures.append((camera, scene.data.time))
                    return original_render(camera)

                scene.render = render
                factory = Mock(side_effect=GlobalRenderer)
                with patch.dict("sys.modules", {"mujoco": SimpleNamespace(Renderer=factory)}):
                    recorder = MujocoRecorder(scene, directory, fps=10, decision_hold_s=0.2)
                    self.addCleanup(recorder.close)
                    renderer = recorder._global_renderer
                    base = recorder._canvas({name: original_render(name) for name in scene.cameras}, 0)
                    height, width = base.shape[:2]
                    overview = np.broadcast_to(np.array((75, 55, 180), np.uint8),
                                               (height - 32, height - 32, 3)).copy()
                    expanded = recorder._append_global(base, overview)
                    self.assertEqual(expanded.shape, (height, width + height - 32, 3))
                    np.testing.assert_array_equal(expanded[:, :width], base)
                    np.testing.assert_array_equal(expanded[32:, width:], overview)
                    self.assertTrue(np.any(expanded[:32, width:] != (19, 25, 35)),
                                    "The recording-only view should have its own title.")

                    recorder.begin_step(step_idx=0, stage="LIFT", token="MV_UP",
                                        annotation="Lift above the table.")
                    scene.data.time = 1.1
                    recorder.capture()
                    recorder.end_step({"act": "MV_UP"})
                    recorder.finish(False, "max_steps_exceeded")
                    recorder.close()

                # Paused video frames reuse one capture, without advancing physics
                # or rendering another view at a later simulation timestamp.
                self.assertEqual(scene.data.time, 1.1)
                capture_times = [1.0, 1.0, 1.1, 1.1, 1.1]
                self.assertEqual(renderer.captures,
                                 [(time, GLOBAL_CAMERA, 256, 256) for time in capture_times])
                self.assertEqual(camera_captures,
                                 [(name, time) for time in capture_times for name in observation_cameras])
                factory.assert_called_once_with(scene.model, height=height - 32, width=height - 32)
                self.assertEqual(renderer.initial_buffer, (height - 32, height - 32))
                self.assertEqual((scene.model.vis.global_.offwidth, scene.model.vis.global_.offheight),
                                 (256, 256))
                renderer.close.assert_called_once_with()
                self.assertEqual(scene.cameras, observation_cameras)
                self.assertEqual(recorder.cameras, observation_cameras)
                self.assertEqual(set(recorder.paths), {*observation_cameras, "combined", "annotations"})
                self.assertEqual(recorder.frame_count, 7)
                events = [json.loads(line) for line in recorder.paths["annotations"].read_text().splitlines()]
                self.assertEqual(events[0]["cameras"], list(observation_cameras))
                self.assertEqual(events[0]["combined_only_cameras"], [GLOBAL_CAMERA])
                self.assertEqual(events[-1]["frames"], recorder.frame_count)
                for name in (*observation_cameras, "combined"):
                    with imageio.get_reader(recorder.paths[name]) as reader:
                        self.assertEqual(reader.count_frames(), recorder.frame_count)
                        self.assertEqual(reader.get_meta_data()["fps"], 10)
                        if name == "combined":
                            for index, time in enumerate((1.0, 1.0, 1.1, 1.1, 1.1, 1.1, 1.1)):
                                frame = reader.get_data(index)
                                self.assertEqual(frame.shape, expanded.shape)
                                np.testing.assert_allclose(frame[height // 2, width + 64],
                                                           renderer.color(time), atol=8)

    def test_global_renderer_construction_failure_restores_framebuffer(self):
        with TemporaryDirectory() as directory:
            scene = DepthPlugScene()
            scene.model = recording_model()

            def fail(model, *, height, width):
                self.assertEqual((height, width), (448, 448))
                self.assertEqual((model.vis.global_.offwidth, model.vis.global_.offheight), (448, 448))
                raise RuntimeError("No recording render context")

            with patch.dict("sys.modules", {"mujoco": SimpleNamespace(Renderer=fail)}):
                with self.assertRaisesRegex(RuntimeError, "No recording render context"):
                    MujocoRecorder(scene, directory)
            self.assertEqual((scene.model.vis.global_.offwidth, scene.model.vis.global_.offheight), (256, 256))
            self.assertFalse(list(Path(directory).glob("*.mp4")))

    def test_global_capture_failure_closes_its_renderer(self):
        with TemporaryDirectory() as directory:
            scene = DepthPlugScene()
            scene.model = recording_model()
            renderer = Mock()
            renderer.render.side_effect = RuntimeError("Global capture failed")
            with patch.dict("sys.modules", {"mujoco": SimpleNamespace(Renderer=Mock(return_value=renderer))}):
                with self.assertRaisesRegex(RuntimeError, "Global capture failed"):
                    MujocoRecorder(scene, directory)
            renderer.close.assert_called_once_with()
            self.assertEqual((scene.model.vis.global_.offwidth, scene.model.vis.global_.offheight), (256, 256))

    def test_five_views_keep_depth_bottom_left_and_decision_bottom_right(self):
        with TemporaryDirectory() as directory:
            scene = DepthPlugScene()
            recorder = MujocoRecorder(scene, directory, fps=10, decision_hold_s=0)
            recorder.begin_step(step_idx=0, stage="GRASP", token="MV_DOWN_SMALL",
                                annotation="The depth image still shows a gap to the handle.")
            scene.data.time += 0.1
            recorder.capture()
            recorder.end_step({"act": "MV_DOWN_SMALL"})
            recorder.finish(False, "max_steps_exceeded")
            self.assertEqual(set(recorder.paths), {*scene.cameras, "combined", "annotations"})
            events = [json.loads(line) for line in recorder.paths["annotations"].read_text().splitlines()]
            self.assertEqual(events[0]["cameras"], list(scene.cameras))
            self.assertEqual(events[-1]["frames"], recorder.frame_count)
            for name in (*scene.cameras, "combined"):
                with imageio.get_reader(recorder.paths[name]) as reader:
                    self.assertEqual(reader.count_frames(), recorder.frame_count)
                    self.assertEqual(reader.get_meta_data()["fps"], 10)
                    frame = reader.get_data(1)
                    if name == "combined":
                        self.assertEqual(frame.shape, (480, 256, 3))
                        for camera, (row, col) in {
                            "wrist": (96, 64), "front": (96, 192),
                            "side": (256, 64), "wrist_insert": (256, 192),
                            "wrist_depth": (416, 64),
                        }.items():
                            np.testing.assert_allclose(frame[row, col], scene.colors[camera], atol=12)
                        # The decision cell contains its teal action text and does
                        # not cover the grayscale E image to its left.
                        panel = frame[352:450, 128:].astype(int)
                        self.assertTrue(np.any((panel[..., 1] > panel[..., 0] + 50)
                                               & (panel[..., 2] > panel[..., 0] + 50)))
                        self.assertLess(int(frame[470, 240].max()), 80)
                    else:
                        self.assertEqual(frame.shape, (128, 128, 3))
                        np.testing.assert_allclose(frame[64, 64], scene.colors[name], atol=12)

    def test_four_camera_plug_recording_preserves_views_and_full_width_decision_row(self):
        with TemporaryDirectory() as directory:
            scene = PlugScene()
            recorder = MujocoRecorder(scene, directory, fps=10, decision_hold_s=0)
            recorder.begin_step(step_idx=0, stage="INSERT", token="MV_DOWN_SMALL",
                                annotation="The pin axis enters the black bore.")
            scene.data.time += 0.1
            recorder.capture()
            recorder.end_step({"act": "MV_DOWN_SMALL"})
            recorder.finish(False, "max_steps_exceeded")
            recorder.close()
            self.assertEqual(set(recorder.paths), {*scene.cameras, "combined", "annotations"})
            events = [json.loads(line) for line in recorder.paths["annotations"].read_text().splitlines()]
            self.assertEqual(set(events[0]["cameras"]), set(scene.cameras))
            self.assertEqual(events[-1]["frames"], recorder.frame_count)
            for name in (*scene.cameras, "combined"):
                with imageio.get_reader(recorder.paths[name]) as reader:
                    self.assertEqual(reader.count_frames(), recorder.frame_count)
                    self.assertEqual(reader.get_meta_data()["fps"], 10)
                    frame = reader.get_data(0)
                    if name == "combined":
                        self.assertEqual(frame.shape, (480, 256, 3))
                        for camera, (row, col) in {
                            "wrist": (96, 64), "front": (96, 192),
                            "side": (256, 64), "wrist_insert": (256, 192),
                        }.items():
                            np.testing.assert_allclose(frame[row, col], scene.colors[camera], atol=12)
                        # The decision row extends below both columns without replacing D.
                        self.assertLess(int(frame[470, 64].max()), 80)
                        self.assertLess(int(frame[470, 192].max()), 80)
                    else:
                        self.assertEqual(frame.shape, (128, 128, 3))
                        np.testing.assert_allclose(frame[64, 64], scene.colors[name], atol=12)

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
