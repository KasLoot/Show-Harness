"""Native Ollama transport, offline; no SDK or live API key required."""
import base64
import io
import json
import unittest
from unittest.mock import Mock, patch

import numpy as np
import requests

from core.vlm.vlm_client import VLMClient, _chat_completion_data


def response(body, status=200):
    reply = requests.Response()
    reply.status_code = status
    reply._content = json.dumps(body).encode()
    return reply


class OllamaTests(unittest.TestCase):
    def test_five_camera_images_preserve_order_in_native_request(self):
        from PIL import Image

        client = VLMClient("https://ollama.com/api/", "test-model:cloud", "test-key",
                           30, 8192, 0.0, provider="ollama")
        self.addCleanup(client.session.close)
        frames = [np.full((8, 8, 3), value, np.uint8) for value in (30, 90, 160, 230, 120)]
        reply = response({"message": {"content": '{"decision":"STOP"}'}, "done": True})
        with patch.object(client.session, "post", return_value=reply) as post:
            client.complete_json("Wrist, Front, Right Side, Angled Wrist, Wrist Depth", frames[0],
                                 wrist_image=frames[1:], schema={"type": "object"})
        images = post.call_args.kwargs["json"]["messages"][0]["images"]
        self.assertEqual(len(images), 5)
        for encoded, expected in zip(images, frames):
            with Image.open(io.BytesIO(base64.b64decode(encoded))) as image:
                np.testing.assert_array_equal(np.asarray(image), expected)

    def test_images_json_and_thinking_use_native_cloud_contract(self):
        client = VLMClient("https://ollama.com/api/", "glm-5.3-flash:cloud", "test-key",
                           30, 8192, 0.0, provider="ollama", reasoning_effort="low")
        self.addCleanup(client.session.close)
        reply = response({"message": {"content": '{"decision":"MV_UP"}',
                                       "thinking": "GRASP"}, "done": True})
        with patch.object(client.session, "post", return_value=reply) as post:
            result = client.complete_json("Choose an action", np.zeros((8, 8, 3), np.uint8),
                                          wrist_image=[np.ones((8, 8, 3), np.uint8),
                                                       np.full((8, 8, 3), 2, np.uint8)],
                                          schema={"type": "object"}, debug=True)
        self.assertEqual(result.payload["json"], {"decision": "MV_UP"})
        self.assertEqual(post.call_args.args[0], "https://ollama.com/api/chat")
        payload = post.call_args.kwargs["json"]
        self.assertEqual(set(payload), {"model", "messages", "stream", "options", "think"})
        self.assertEqual(payload["options"], {"num_predict": 8192, "temperature": 0.0})
        self.assertEqual(payload["think"], "low")
        self.assertFalse(payload["stream"])
        message = payload["messages"][0]
        self.assertEqual(len(message["images"]), 3)
        for encoded in message["images"]:
            self.assertTrue(base64.b64decode(encoded, validate=True))
        self.assertIn('"type": "object"', message["content"])
        self.assertEqual(client.session.headers["Authorization"], "Bearer test-key")

    def test_reasoning_without_a_final_answer_is_rejected(self):
        for body in ({"done": True, "message": {"content": "", "thinking": "GRASP"}},
                     {"done": False, "message": {"content": "GRASP"}},
                     {"error": "unavailable"}, {"done": True, "message": []}):
            data, problem = _chat_completion_data(response(body), "ollama")
            self.assertIsNone(data)
            self.assertTrue(problem)

    def test_token_budget_retries_and_health_check(self):
        client = VLMClient("https://ollama.com", "glm-5.3-flash:cloud", "test-key",
                           30, 8192, 0.0, provider="ollama", max_retries=1)
        self.addCleanup(client.session.close)
        responses = [response({"error": "busy"}, 429),
                     response({"message": {"content": "MV_LEFT"}, "done": True})]
        with patch.object(client.session, "post", side_effect=responses) as post, \
                patch("core.vlm.vlm_client.time.sleep"):
            result = client.complete_token("Reply MV_LEFT", ["MV_LEFT"], None)
        self.assertEqual(result.token, "MV_LEFT")
        self.assertEqual(post.call_count, 2)
        self.assertEqual(post.call_args.kwargs["json"]["options"]["num_predict"], 8192)
        with patch.object(client.session, "get", return_value=Mock(status_code=200)) as get:
            client.health_check()
        self.assertEqual(get.call_args.args[0], "https://ollama.com/api/tags")
        chained = client._finalize_payload({
            "model": "model", "max_tokens": 8192, "temperature": 0,
            "messages": [{"role": "assistant", "content": "MV_LEFT"},
                         {"role": "user", "content": "And the other arm?"}],
        })
        self.assertEqual(chained["messages"][1]["content"], "And the other arm?")


if __name__ == "__main__":
    unittest.main()
