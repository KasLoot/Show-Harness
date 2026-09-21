"""Ordered current/before-action images in the actual offline provider payloads."""
import base64
import io
import json
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import requests

from core.cartesian_actions import CartesianActions
from core.record.images import frame_fingerprint
from core.visual_history import VisualHistoryEntry, compose_visual_history
from core.vlm.roles import ControllerAgent
from core.vlm.vlm_client import VLMClient


CAMERAS = ("wrist", "front", "side", "wrist_insert", "wrist_depth")
LABELS = ("Wrist", "Front", "Right Side", "Angled Wrist", "Wrist Depth")


def views(offset):
    return [(name, np.full((8, 8, 3), offset + index, np.uint8))
            for index, name in enumerate(CAMERAS)]


def response(text, provider):
    reply = requests.Response()
    reply.status_code = 200
    body = ({"message": {"content": text}, "done": True} if provider == "ollama"
            else {"choices": [{"message": {"content": text}}]})
    reply._content = json.dumps(body).encode()
    return reply


class VisualHistoryPayloadTests(unittest.TestCase):
    def make_agent(self, provider, *, cot=False):
        client = VLMClient("https://example.test/v1", "test-model", "test-key",
                           30, 8192, 0.0, provider=provider, max_retries=0)
        self.addCleanup(client.session.close)
        return ControllerAgent(client, "CURRENT stage: {stage}\n{output_contract}", "",
                               cot_mode=cot, cartesian_actions=CartesianActions())

    def decide(self, agent, current, history, *, current_depth_text=""):
        return agent.decide(
            "Insert the plug", {"motion": "INSERT"}, "MV_FWD_SMALL", "MV_FWD_SMALL", "CLOSED",
            current[0][1], wrist_image=[image for _, image in current[1:]],
            current_view_names=LABELS[:len(current)], current_step_idx=10, visual_history=history,
            current_depth_text=current_depth_text,
        )

    def inspect(self, payload, provider):
        self.assertEqual(len(payload["messages"]), 1)
        message = payload["messages"][0]
        self.assertEqual(message["role"], "user")
        if provider == "ollama":
            encoded, prompt = message["images"], message["content"]
        else:
            parts = message["content"]
            self.assertEqual(parts[-1]["type"], "text")
            self.assertTrue(all(part["type"] == "image_url" for part in parts[:-1]))
            encoded = [part["image_url"]["url"].split(",", 1)[1] for part in parts[:-1]]
            prompt = parts[-1]["text"]
        images = []
        for data in encoded:
            with Image.open(io.BytesIO(base64.b64decode(data))) as image:
                images.append(np.asarray(image).copy())
        return images, prompt

    def test_current_then_oldest_before_frames_and_effects_across_providers(self):
        current, older, latest = views(200), views(20), views(100)
        history = [
            VisualHistoryEntry(9, "ALIGN", latest, "MV_FWD_SMALL", "Requested +2 mm X; moved +1.9 mm X."),
            VisualHistoryEntry(3, "LIFT", older, "MV_UP_LARGE", "Requested +50 mm Z; moved +49.8 mm Z."),
        ]
        expected = current + older + latest
        for provider in ("openai", "gemini", "ollama"):
            with self.subTest(provider=provider):
                agent = self.make_agent(provider)
                reply = response('{"decision":"STOP","reasoning":"Inspect alignment."}', provider)
                with patch.object(agent.client.session, "post", return_value=reply) as post:
                    self.assertEqual(self.decide(agent, current, history).token, "STOP")
                post.assert_called_once()
                images, prompt = self.inspect(post.call_args.kwargs["json"], provider)
                self.assertEqual(len(images), 15)
                for actual, (_, wanted) in zip(images, expected):
                    np.testing.assert_array_equal(actual, wanted)
                for phrase in ("CURRENT step 10: image 1 = Wrist", "image 5 = Wrist Depth",
                               "HISTORY step 3, stage LIFT, BEFORE action MV_UP_LARGE: image 6 = Wrist",
                               "HISTORY step 9, stage ALIGN, BEFORE action MV_FWD_SMALL: image 11 = Wrist",
                               "moved +49.8 mm Z", "moved +1.9 mm X", "not extra current cameras",
                               "CURRENT stage", "next past set, or CURRENT"):
                    self.assertIn(phrase, prompt)
                self.assertLess(prompt.index("HISTORY step 3"), prompt.index("HISTORY step 9"))
                self.assertEqual([entry["t_offset"] for entry in agent.last_prompt_media],
                                 [0] * 5 + [-7] * 5 + [-1] * 5)
                self.assertEqual([entry["camera"] for entry in agent.last_prompt_media], list(CAMERAS) * 3)
                for slot, (media, (_, frame)) in enumerate(zip(agent.last_prompt_media, expected)):
                    self.assertEqual(media["slot"], slot)
                    self.assertEqual(media["sha1"], frame_fingerprint(frame)["sha1"])
                self.assertEqual([entry.step_idx for entry in history], [9, 3], "Do not mutate the queue")

    def test_json_and_internal_strict_token_retries_keep_all_images_and_pairing(self):
        current, previous = views(200)[:4], views(30)[:4]
        current_depth = "Metric optical-axis depth, m.\n[[0.048,0.055],[0.102,null]]"
        previous_depth = "Metric optical-axis depth, m.\n[[0.067,0.074],[0.122,null]]"
        history = [VisualHistoryEntry(9, "GRASP", previous, "GRASP",
                                      "Empty grasp; recovery reopened fingers.", depth_text=previous_depth)]
        agent = self.make_agent("openai")
        replies = [response(text, "openai") for text in ("unparseable answer", "no valid command", "STOP")]
        with patch.object(agent.client.session, "post", side_effect=replies) as post:
            result = self.decide(agent, current, history, current_depth_text=current_depth)
        self.assertEqual(result.token, "STOP")
        self.assertEqual(post.call_count, 3)
        for call in post.call_args_list:
            images, prompt = self.inspect(call.kwargs["json"], "openai")
            self.assertEqual(len(images), 8)
            for actual, (_, wanted) in zip(images, current + previous):
                np.testing.assert_array_equal(actual, wanted)
            self.assertIn("BEFORE action GRASP: image 5 = Wrist", prompt)
            self.assertIn("recovery reopened fingers", prompt)
            self.assertEqual(prompt.count(current_depth), 1)
            self.assertEqual(prompt.count(previous_depth), 1)

    def test_cot_and_token_retry_keep_the_same_history(self):
        current, previous = views(200)[:4], views(30)[:4]
        agent = self.make_agent("gemini", cot=True)
        current_depth = "Metric optical-axis depth, m.\n[[0.038,0.056],[null,0.07]]"
        previous_depth = "Metric optical-axis depth, m.\n[[0.041,0.060],[null,0.06]]"
        history = [VisualHistoryEntry(9, "ALIGN", previous, "ROT_Y_POS_SMALL", "Rotated +1.9 degrees.",
                                      depth_text=previous_depth)]
        with patch.object(agent.client.session, "post", side_effect=[
            response("I am uncertain.", "gemini"), response("STOP", "gemini"),
        ]) as post:
            self.assertEqual(self.decide(agent, current, history, current_depth_text=current_depth).token, "STOP")
        self.assertEqual(post.call_count, 2)
        for call in post.call_args_list:
            images, prompt = self.inspect(call.kwargs["json"], "gemini")
            self.assertEqual(len(images), 8)
            self.assertIn("BEFORE action ROT_Y_POS_SMALL", prompt)
            self.assertIn("Rotated +1.9 degrees", prompt)
            self.assertEqual(prompt.count(current_depth), 1)
            self.assertEqual(prompt.count(previous_depth), 1)

    def test_four_rgb_views_and_numeric_depth_remain_distinct_across_providers(self):
        current, older, latest = views(200)[:4], views(20)[:4], views(100)[:4]
        current_depth = "Optical-axis metres, TCP=0.048 m, rows top-to-bottom.\n[[0.031,0.045],[null,0.106]]"
        old_depth = "Optical-axis metres, TCP=0.048 m, rows top-to-bottom.\n[[0.081,0.095],[null,0.156]]"
        last_depth = "Optical-axis metres, TCP=0.048 m, rows top-to-bottom.\n[[0.033,0.047],[null,0.108]]"
        history = [
            VisualHistoryEntry(9, "ALIGN", latest, "MV_DOWN_SMALL", "Moved -2 mm Z.", depth_text=last_depth),
            VisualHistoryEntry(3, "APPROACH", older, "MV_DOWN_LARGE", "Moved -48 mm Z.", depth_text=old_depth),
        ]
        for provider in ("openai", "gemini", "ollama"):
            with self.subTest(provider=provider):
                agent = self.make_agent(provider)
                reply = response('{"decision":"STOP","reasoning":"Inspect depth."}', provider)
                with patch.object(agent.client.session, "post", return_value=reply) as post:
                    self.assertEqual(self.decide(agent, current, history, current_depth_text=current_depth).token,
                                     "STOP")
                images, prompt = self.inspect(post.call_args.kwargs["json"], provider)
                self.assertEqual(len(images), 12)
                for actual, (_, wanted) in zip(images, current + older + latest):
                    np.testing.assert_array_equal(actual, wanted)
                for depth in (current_depth, old_depth, last_depth):
                    self.assertEqual(prompt.count(depth), 1, "Preserve each serialized array exactly once")
                self.assertIn("CURRENT WRIST DEPTH (step 10):\n" + current_depth, prompt)
                self.assertIn("HISTORY WRIST DEPTH (step 3, BEFORE action MV_DOWN_LARGE):\n" + old_depth, prompt)
                self.assertIn("HISTORY WRIST DEPTH (step 9, BEFORE action MV_DOWN_SMALL):\n" + last_depth, prompt)
                self.assertLess(prompt.index(old_depth), prompt.index(last_depth))
                self.assertIn("BEFORE action MV_DOWN_LARGE: image 5 = Wrist", prompt)
                self.assertIn("BEFORE action MV_DOWN_SMALL: image 9 = Wrist", prompt)
                self.assertNotIn("grayscale", prompt)
                self.assertNotIn("Wrist Depth (E)", prompt)
                self.assertEqual([entry["camera"] for entry in agent.last_prompt_media], list(CAMERAS[:4]) * 3)
                self.assertEqual([entry["t_offset"] for entry in agent.last_prompt_media],
                                 [0] * 4 + [-7] * 4 + [-1] * 4)

    def test_first_request_includes_current_depth_without_history(self):
        current = views(200)[:4]
        depth = "Depth sensor values in metres:\n[[0.01,null],[0.04,0.056789]]"
        agent = self.make_agent("openai")
        with patch.object(agent.client.session, "post", return_value=response(
            '{"decision":"STOP","reasoning":"Inspect."}', "openai",
        )) as post:
            self.decide(agent, current, (), current_depth_text=depth)
        images, prompt = self.inspect(post.call_args.kwargs["json"], "openai")
        self.assertEqual(len(images), 4)
        self.assertEqual(prompt.count("CURRENT WRIST DEPTH"), 1)
        self.assertEqual(prompt.count(depth), 1)
        self.assertNotIn("HISTORY WRIST DEPTH", prompt)
        self.assertNotIn("VISUAL ACTION HISTORY", prompt)

    def test_empty_history_preserves_only_current_payload_and_manifest(self):
        current = views(200)
        agent = self.make_agent("openai")
        reply = response('{"decision":"STOP","reasoning":"Inspect."}', "openai")
        with patch.object(agent.client.session, "post", return_value=reply) as post:
            self.decide(agent, current, [])
        images, prompt = self.inspect(post.call_args.kwargs["json"], "openai")
        self.assertEqual(len(images), 5)
        self.assertNotIn("VISUAL ACTION HISTORY", prompt)
        self.assertEqual([entry["t_offset"] for entry in agent.last_prompt_media], [0] * 5)

    def test_history_requires_a_real_current_step_and_current_views(self):
        history = [VisualHistoryEntry(9, "ALIGN", views(30), "MV_DOWN_SMALL", "Moved 2 mm down.")]
        for current, step in ((views(200), None), (views(200), 9), ([], 10)):
            with self.subTest(step=step), self.assertRaises(ValueError):
                compose_visual_history(current, history, current_step_idx=step)


if __name__ == "__main__":
    unittest.main()
