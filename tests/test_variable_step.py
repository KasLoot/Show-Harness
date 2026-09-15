"""Regression for coarse alignment oscillations at the MuJoCo starting height."""
import json
import unittest
from unittest.mock import Mock

from core.vlm.roles import ControllerAgent
from core.vlm.vlm_client import VLMResponse
from plugins.variable_step import VariableStepPlugin
from plugins.proprioception import ProprioceptionPlugin


class VariableStepTests(unittest.TestCase):
    def test_three_sizes_use_clearance_without_coarsening_visible_alignment(self):
        plugin = VariableStepPlugin(enabled=True, large_step_m=0.10)
        for height, descent, travel in ((0.075, 0.02, 0.05), (0.175, 0.05, 0.05), (0.275, 0.10, 0.10)):
            self.assertEqual(plugin.step_m_for("MV_DOWN", 0.02, height, 0.025, True), descent)
            self.assertEqual(plugin.step_m_for("MV_UP", 0.02, height, 0.025, True), travel)
            self.assertEqual(plugin.step_m_for("MV_FWD", 0.02, height, 0.025, False), travel)
            for visible in (True, None):
                self.assertEqual(plugin.step_m_for("MV_FWD", 0.02, height, 0.025, visible), 0.02)
        self.assertEqual(plugin.step_m_for("MV_FWD", 0.02, None, None, False), 0.05)
        plugin.enabled = False
        self.assertEqual(plugin.step_m_for("MV_UP", 0.02, 0.275, 0.025, False), 0.02)
        for kwargs in ({"large_step_m": 0.04}, {"large_step_m": float("nan")},
                       {"large_step_m": 0.10, "large_above_table_m": 0.05}):
            with self.assertRaises(ValueError):
                VariableStepPlugin(**kwargs)
        text = ProprioceptionPlugin(coarse_step_m=0.05, large_step_m=0.10).render(
            {"eef_pos": [0.48, 0.16, 0.24]}, 0.025, stage="GRASP",
        )
        for label in ("fine steps move ~2 cm", "coarse steps ~5 cm", "large steps ~10 cm"):
            self.assertIn(label, text)

    def test_height_hint_does_not_undo_release_or_retreat(self):
        plugin = ProprioceptionPlugin(enabled=True)
        proprio = {"eef_pos": [0.48, 0.16, 0.175]}
        for stage in ("RELEASE", "RETREAT"):
            for holding in (True, False):
                text = plugin.render(proprio, 0.025, holding=holding, stage=stage)
                self.assertIn("15.0 cm above the table", text)
                self.assertNotIn("descend with MV_DOWN", text)
                self.assertNotIn("lift until", text)
        grasp = plugin.render(proprio, 0.025, stage="GRASP")
        self.assertIn("only after rough horizontal alignment", grasp)
        self.assertNotIn("MV_DOWN first", grasp)

    def test_height_does_not_override_horizontal_alignment(self):
        plugin = VariableStepPlugin(enabled=True)
        for token in ("MV_LEFT", "MV_RIGHT", "MV_FWD", "MV_BACK"):
            for visible in (True, None, False):
                self.assertEqual(plugin.step_m_for(token, 0.02, 0.24, 0.025, visible),
                                 0.05 if visible is False else 0.02)
        self.assertEqual(plugin.step_m_for("MV_DOWN", 0.02, 0.24, 0.025, True), 0.05)
        self.assertEqual(plugin.step_m_for("MV_DOWN", 0.02, 0.08, 0.025, True), 0.02)
        self.assertEqual(plugin.step_m_for("MV_UP", 0.02, 0.08, 0.025, True), 0.05)
        plugin.enabled = False
        self.assertEqual(plugin.step_m_for("MV_UP", 0.02, 0.24, 0.025, False), 0.02)

    def test_typed_visibility_wins_and_legacy_marker_still_works(self):
        client = Mock()
        agent = ControllerAgent(client, "{variable_step}\n{output_contract}", "",
                                variable_step_plugin=VariableStepPlugin(enabled=True))
        for fields, reasoning, expected in (
            ({"target_in_wrist": True}, "WRIST: NO. Small correction.", True),
            ({"target_in_wrist": False}, "Move toward the target.", False),
            ({}, "WRIST: YES. Small correction.", True),
            ({}, "The red cube is slightly right of center.", None),
            ({"target_in_wrist": "false"}, "Small correction.", None),
        ):
            answer = {"decision": "MV_RIGHT", "reasoning": reasoning, **fields}
            client.reset_mock()
            client.complete_json.return_value = VLMResponse("", json.dumps(answer), {"json": answer})
            response = agent.decide("pick cube", {}, "none", "NONE", "OPEN", None)
            self.assertEqual(response.payload["target_in_wrist"], expected)
            self.assertEqual(response.token, "MV_RIGHT")
            self.assertEqual(client.complete_json.call_count, 1)
            client.complete_token.assert_not_called()
            schema = client.complete_json.call_args.kwargs["schema"]
            self.assertEqual(schema["properties"]["target_in_wrist"]["type"], "boolean")


if __name__ == "__main__":
    unittest.main()
