"""Gemini transport with the three simulation views; no network or live key."""
import json
import unittest
from unittest.mock import patch

import numpy as np
import requests

from core.vlm.vlm_client import VLMClient


class GeminiTests(unittest.TestCase):
    def test_gemini3_keeps_configured_temperature_and_three_images(self):
        client = VLMClient("https://generativelanguage.googleapis.com/v1beta/openai",
                           "gemini-3.8-flash", "test-key", 30, 8192, 1.0,
                           provider="gemini", reasoning_effort="low")
        self.addCleanup(client.session.close)
        answer = {"decision": "MV_DOWN", "target_in_wrist": True, "reasoning": "Descend to the cube."}
        reply = requests.Response()
        reply.status_code = 200
        reply._content = json.dumps({"choices": [{"message": {"content": json.dumps(answer)}}]}).encode()
        frames = [np.full((8, 8, 3), i, np.uint8) for i in range(3)]
        with patch.object(client.session, "post", return_value=reply) as post:
            result = client.complete_json("Choose an action", frames[0], wrist_image=frames[1:],
                                          schema={"type": "object"}, temperature=0.0)
        self.assertEqual(result.payload["json"], answer)
        self.assertEqual(post.call_args.args[0], f"{client.base_url}/chat/completions")
        payload = post.call_args.kwargs["json"]
        self.assertEqual(payload["model"], "gemini-3.8-flash")
        self.assertEqual(payload["temperature"], 1.0)
        self.assertEqual(payload["reasoning_effort"], "low")
        self.assertEqual(payload["response_format"], {"type": "json_object"})
        self.assertEqual([p["type"] for p in payload["messages"][0]["content"]],
                         ["image_url", "image_url", "image_url", "text"])
        self.assertNotIn("chat_template_kwargs", payload)
        self.assertNotIn("options", payload)
        self.assertEqual(client.session.headers["Authorization"], "Bearer test-key")


if __name__ == "__main__":
    unittest.main()
