"""Metric wrist depth encoding and render-mode isolation, without graphics or API calls."""
import importlib.util
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from core.sim.mujoco_depth import depth_prompt, depth_to_grayscale, depth_to_text


class DepthEncodingTests(unittest.TestCase):
    def test_fixed_metric_scale_is_shared_by_all_three_channels(self):
        distances = np.array([[0.06, 0.12, 0.18], [0.24, 0.30, 0.90]], dtype=np.float32)
        image = depth_to_grayscale(distances)
        self.assertEqual(image.dtype, np.uint8)
        self.assertEqual(image.shape, (2, 3, 3))
        self.assertTrue(image.flags.c_contiguous)
        expected = np.array([[204, 153, 102], [51, 0, 0]], dtype=np.uint8)
        for channel in range(3):
            np.testing.assert_array_equal(image[..., channel], expected)
        np.testing.assert_array_equal(distances,
                                      np.array([[0.06, 0.12, 0.18], [0.24, 0.30, 0.90]], np.float32))

    def test_background_and_invalid_distances_are_black(self):
        distances = np.array([[np.nan, np.inf, -np.inf], [0.0, -0.1, 100.0]])
        np.testing.assert_array_equal(depth_to_grayscale(distances), np.zeros((2, 3, 3), np.uint8))

    def test_custom_scale_saturates_without_frame_normalization(self):
        distances = np.array([[0.02, 0.04, 0.14, 0.24, 0.40]])
        image = depth_to_grayscale(distances, near_m=0.04, far_m=0.24)
        np.testing.assert_allclose(image[0, :, 0], [255, 255, 128, 0, 0], atol=1)
        # The same object depth must retain its brightness when nearer/farther
        # objects enter the frame; otherwise grayscale cannot express distance.
        near_scene = depth_to_grayscale(np.array([[0.12, 0.06]]))
        far_scene = depth_to_grayscale(np.array([[0.12, 0.28]]))
        uniform_scene = depth_to_grayscale(np.full((3, 4), 0.12))
        np.testing.assert_array_equal(near_scene[0, 0], far_scene[0, 0])
        np.testing.assert_array_equal(uniform_scene, np.full((3, 4, 3), 153, np.uint8))

    def test_invalid_calibration_fails_before_encoding(self):
        for near, far in ((-0.01, 0.30), (0.30, 0.30), (0.31, 0.30),
                          (np.nan, 0.30), (0.0, np.inf)):
            with self.subTest(near=near, far=far), self.assertRaises(ValueError):
                depth_to_grayscale(np.ones((2, 2)), near_m=near, far_m=far)
        with self.assertRaises(ValueError):
            depth_to_grayscale(np.ones((2, 2, 3)))

    def test_prompt_reports_configured_scale_and_measured_tcp_plane(self):
        prompt = depth_prompt({"near_m": 0.02, "far_m": 0.42, "tcp_depth_m": 0.061})
        for phrase in ("Wrist Depth (E)", "Wrist RGB (A)", "20 mm", "420 mm",
                       "61.0 mm", "optical axis", "never auto-adjusts", "Occluded"):
            self.assertIn(phrase, prompt)
        self.assertNotIn("48.0 mm", prompt)


class NumericDepthTests(unittest.TestCase):
    calibration = {"near_m": 0.0, "far_m": 0.30, "tcp_depth_m": 0.048,
                   "representation": "text"}

    def test_numeric_grid_samples_exact_pixel_centers_without_boundary_averaging(self):
        depth = (np.arange(48).reshape(6, 8) + 20) / 1000
        payload = json.loads(depth_to_text(depth, self.calibration, grid_rows=3, grid_cols=4))
        self.assertEqual(payload["name"], "WRIST_DEPTH_MM")
        self.assertEqual((payload["source_width_px"], payload["source_height_px"]), (8, 6))
        self.assertEqual(payload["sample_x_px"], [1, 3, 5, 7])
        self.assertEqual(payload["sample_y_px"], [1, 3, 5])
        self.assertEqual(payload["depth_mm"], [[29, 31, 33, 35], [45, 47, 49, 51], [61, 63, 65, 67]])
        self.assertEqual(payload["tcp_depth_mm"], 48)
        self.assertIn("zero-based", payload["pixel_indices"])
        self.assertIn("top-to-bottom", payload["pixel_indices"])
        self.assertIn("left-to-right", payload["pixel_indices"])
        self.assertIn("thin features", payload["coverage"])
        # Changing a neighboring pixel on another surface must not contaminate
        # the sampled value through interpolation or averaging.
        depth[0, 0] = .299
        again = json.loads(depth_to_text(depth, self.calibration, grid_rows=3, grid_cols=4))
        self.assertEqual(again["depth_mm"], payload["depth_mm"])

    def test_grid_clamps_to_source_size_without_duplicate_or_invented_pixels(self):
        payload = json.loads(depth_to_text(np.full((2, 3), .052), self.calibration))
        self.assertEqual((payload["requested_grid_rows"], payload["requested_grid_cols"]), (32, 32))
        self.assertEqual((payload["grid_rows"], payload["grid_cols"]), (2, 3))
        self.assertEqual(payload["sample_x_px"], [0, 1, 2])
        self.assertEqual(payload["sample_y_px"], [0, 1])
        self.assertEqual(payload["depth_mm"], [[52, 52, 52], [52, 52, 52]])

    def test_invalid_and_out_of_range_samples_are_json_null_not_zero_or_saturated(self):
        depth = np.array([[np.nan, np.inf, -np.inf, 0, -.1, .301],
                          [.04, .0501, .0528, .10, .299, .30]])
        calibration = {**self.calibration, "near_m": .05}
        payload = json.loads(depth_to_text(depth, calibration, grid_rows=2, grid_cols=6))
        self.assertEqual(payload["depth_mm"], [[None] * 6, [None, 50, 53, 100, 299, 300]])
        self.assertEqual(payload["valid_range_mm"], [50, 300])
        self.assertIn("unknown", payload["null"])

    def test_grid_parameters_and_depth_shape_are_validated(self):
        for value in (0, -1, 2.0, 1.5, True, False, "32", np.nan, np.inf):
            for key in ("grid_rows", "grid_cols"):
                with self.subTest(key=key, value=value), self.assertRaisesRegex(ValueError, key):
                    depth_to_text(np.ones((2, 2)), self.calibration, **{key: value})
        for shape in ((0, 2), (2, 0), (2,), (2, 2, 3)):
            with self.subTest(shape=shape), self.assertRaisesRegex(ValueError, "two-dimensional"):
                depth_to_text(np.zeros(shape), self.calibration)

    def test_approaching_surfaces_changes_numeric_metres_without_frame_normalization(self):
        before = np.array([[.062, .190], [.055, .240]], dtype=np.float32)
        after = before - .010
        a = json.loads(depth_to_text(before, self.calibration))
        b = json.loads(depth_to_text(after, self.calibration))
        self.assertEqual(a["depth_mm"], [[62, 190], [55, 240]])
        self.assertEqual(b["depth_mm"], [[52, 180], [45, 230]])
        self.assertEqual(a["sample_x_px"], b["sample_x_px"])
        self.assertEqual(a["sample_y_px"], b["sample_y_px"])
        self.assertEqual(a["valid_range_mm"], b["valid_range_mm"])

    def test_numeric_prompt_uses_calibration_without_a_fifth_image_or_gray_scale(self):
        prompt = depth_prompt(self.calibration)
        for phrase in ("Numeric Wrist Depth", "Wrist RGB (A)", "millimetres", "48.0 mm",
                       "Null", "sparse grid", "occluded", "not world-Z"):
            self.assertIn(phrase, prompt)
        for phrase in ("(E)", "grayscale", "Gray-value", "black", "white"):
            self.assertNotIn(phrase, prompt)


class _Renderer:
    def __init__(self):
        self.depth_enabled = False
        self.fail_at = None
        self.depth = np.array([[0.048, 0.12], [0.24, 0.30]], dtype=np.float32)
        self.rgb = np.full((2, 2, 3), [80, 160, 240], dtype=np.uint8)
        self.cameras = []
        self.depth_reads = 0

    def enable_depth_rendering(self):
        self.depth_enabled = True

    def disable_depth_rendering(self):
        self.depth_enabled = False

    def update_scene(self, data, *, camera):
        self.cameras.append(camera)
        if self.fail_at == "update_scene":
            raise RuntimeError("scene update failed")

    def render(self):
        if self.fail_at == "render":
            raise RuntimeError("render failed")
        self.depth_reads += int(self.depth_enabled)
        return self.depth if self.depth_enabled else self.rgb


@unittest.skipUnless(importlib.util.find_spec("mujoco"), "MuJoCo session requires its optional dependency")
class DepthRenderModeTests(unittest.TestCase):
    def setUp(self):
        from core.sim.mujoco_session import MujocoSession

        self.session = object.__new__(MujocoSession)
        self.session.model = object()
        self.session.data = object()
        self.session._renderer = _Renderer()
        self.session.depth_near_m = 0.0
        self.session.depth_far_m = 0.30
        for name in ("mj_kinematics", "mj_camlight"):
            active = patch(f"core.sim.mujoco_session.mujoco.{name}")
            active.start()
            self.addCleanup(active.stop)

    def test_depth_is_metric_and_next_render_is_rgb(self):
        renderer = self.session._renderer
        depth = self.session.render_depth("wrist")
        np.testing.assert_array_equal(depth, renderer.depth)
        self.assertFalse(np.shares_memory(depth, renderer.depth), "Renderer owns a reusable frame buffer")
        self.assertFalse(renderer.depth_enabled)
        rgb = self.session.render("wrist")
        np.testing.assert_array_equal(rgb, renderer.rgb)
        self.assertEqual(renderer.cameras, ["wrist", "wrist"])

    def test_depth_failure_always_restores_rgb_mode(self):
        renderer = self.session._renderer
        for operation in ("update_scene", "render"):
            with self.subTest(operation=operation):
                renderer.fail_at = operation
                with self.assertRaises(RuntimeError):
                    self.session.render_depth("wrist")
                self.assertFalse(renderer.depth_enabled)
                renderer.fail_at = None
                np.testing.assert_array_equal(self.session.render("wrist"), renderer.rgb)

    def test_virtual_depth_view_reuses_original_wrist_camera_and_scale(self):
        renderer = self.session._renderer
        self.session.depth_far_m = 0.60
        image = self.session.render("wrist_depth")
        np.testing.assert_array_equal(image, depth_to_grayscale(renderer.depth, far_m=0.60))
        self.assertEqual(renderer.cameras, ["wrist"])
        self.assertFalse(renderer.depth_enabled)

    def _observation_session(self, representation):
        self.session.cameras = ("side", "wrist", "front", "wrist_insert", "wrist_depth")
        self.session.depth_representation = representation
        self.session.depth_grid_rows = self.session.depth_grid_cols = 32
        self.session._tcp = 0
        self.session.data = SimpleNamespace(
            camera=lambda _: SimpleNamespace(xmat=np.eye(3).reshape(-1), xpos=np.array([0., 0., .1])),
            site=lambda _: SimpleNamespace(xpos=np.array([0., 0., .052])),
        )
        for name, value in (("get_ee_pose", np.array([0., 0., .052, 0., 0., 0., 1.])),
                            ("get_gripper_position", np.array([.08]))):
            active = patch.object(self.session, name, return_value=value)
            active.start()
            self.addCleanup(active.stop)

    def test_observation_captures_depth_once_for_text_raw_data_and_diagnostic_image(self):
        self._observation_session("text")
        renderer = self.session._renderer
        with patch.object(self.session, "render_depth", wraps=self.session.render_depth) as render_depth:
            obs = self.session.get_observation()
        render_depth.assert_called_once_with("wrist")
        self.assertEqual(renderer.depth_reads, 1)
        self.assertEqual(renderer.cameras, ["side", "wrist", "front", "wrist_insert", "wrist"])
        self.assertEqual(obs["wrist_depth_calibration"]["representation"], "text")
        self.assertAlmostEqual(obs["wrist_depth_calibration"]["tcp_depth_m"], .048)
        raw = obs["wrist_depth_m"]
        np.testing.assert_array_equal(raw, renderer.depth)
        self.assertFalse(np.shares_memory(raw, renderer.depth))
        self.assertFalse(raw.flags.writeable)
        np.testing.assert_array_equal(obs["wrist_depth"], depth_to_grayscale(raw))
        self.assertIs(obs["extra_views"]["wrist_depth"], obs["wrist_depth"])
        self.assertEqual(json.loads(obs["wrist_depth_text"])["depth_mm"][0], [48, 120])

        original_text = obs["wrist_depth_text"]
        # RGB changes leave the numeric sensor reading unchanged. Moving surfaces
        # closer in the independent raw depth buffer changes the millimetre values.
        renderer.rgb[:] = [230, 15, 60]
        color_changed = self.session.get_observation()
        self.assertEqual(color_changed["wrist_depth_text"], original_text)
        renderer.depth[:] -= .010
        closer = self.session.get_observation()
        self.assertEqual(json.loads(closer["wrist_depth_text"])["depth_mm"][0], [38, 110])
        self.assertEqual(obs["wrist_depth_text"], original_text)
        self.assertAlmostEqual(float(raw[0, 0]), .048, places=6)

    def test_grayscale_observation_remains_backward_compatible_and_exposes_raw_depth(self):
        self._observation_session("grayscale")
        obs = self.session.get_observation()
        self.assertNotIn("wrist_depth_text", obs)
        self.assertNotIn("representation", obs["wrist_depth_calibration"])
        self.assertIn("Wrist Depth (E)", depth_prompt(obs["wrist_depth_calibration"]))
        self.assertEqual(self.session._renderer.depth_reads, 1)
        np.testing.assert_array_equal(obs["wrist_depth_m"], self.session._renderer.depth)


if __name__ == "__main__":
    unittest.main()
