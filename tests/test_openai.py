"""OpenAI model selection and Chat Completions transport, without live API calls."""
import argparse
import base64
import importlib.util
import io
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image
import requests

from core.cartesian_actions import CartesianActions
from core.config import load_yaml, resolve_vlm_config
from core.sim.launch import make_vlm_client
from core.v0_types import EpisodeResult
from core.vlm.roles import ControllerAgent
from core.vlm.vlm_client import VLMClient


ROOT = Path(__file__).resolve().parents[1]


def _reply(content):
    response = requests.Response()
    response.status_code = 200
    response._content = json.dumps({
        "choices": [{"message": {"content": content, "reasoning_content": "MV_DOWN_LARGE"},
                     "finish_reason": "stop"}],
        "usage": {"completion_tokens": 120, "completion_tokens_details": {"reasoning_tokens": 100}},
    }).encode()
    return response


class OpenAIConfigTests(unittest.TestCase):
    def test_both_mujoco_tasks_default_to_sol_with_medium_native_reasoning(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "openai-test-key"}, clear=True):
            for filename in ("robot_mujoco.yaml", "robot_mujoco_plug.yaml"):
                with self.subTest(config=filename):
                    cfg = load_yaml(ROOT / "configs" / filename)
                    self.assertEqual(cfg["vlm_backend"], "openai")
                    resolved = resolve_vlm_config(cfg)
                    self.assertEqual(resolved["backend"], "openai")
                    self.assertEqual(resolved["provider"], "openai")
                    self.assertEqual(resolved["model"], "gpt-5.6-sol")
                    self.assertEqual(resolved["base_url"], "https://api.openai.com/v1")
                    self.assertEqual(resolved["reasoning_effort"], "medium")
                    self.assertEqual(resolved["max_tokens"], 8192)
                    self.assertFalse(resolved["reasoning_cot"], "Native reasoning keeps the JSON action contract")
                    self.assertEqual(resolved["api_key"], "openai-test-key")
                    client = make_vlm_client(argparse.Namespace(), {"vlm": resolved})
                    self.addCleanup(client.session.close)
                    self.assertTrue(client.reasoning_enabled)
                    self.assertEqual(client.api_dialect, "openai")

    def test_openai_environment_overrides_do_not_change_other_backend_profiles(self):
        cfg = load_yaml(ROOT / "configs/robot_mujoco_plug.yaml")
        environment = {
            "OPENAI_API_KEY": "openai-test-key", "OPENAI_MODEL": "openai-env-model",
            "OPENAI_REASONING_EFFORT": "high", "GEMINI_API_KEY": "gemini-test-key",
            "GEMINI_MODEL": "gemini-env-model", "OLLAMA_API_KEY": "ollama-test-key",
            "OLLAMA_MODEL": "ollama-env-model:cloud",
        }
        with patch.dict(os.environ, environment, clear=True):
            for backend, model, effort in (("openai", "openai-env-model", "high"),
                                           ("gemini", "gemini-env-model", "low"),
                                           ("ollama", "ollama-env-model:cloud", "low")):
                with self.subTest(backend=backend):
                    resolved = resolve_vlm_config(cfg, backend=backend)
                    self.assertEqual(resolved["backend"], backend)
                    self.assertEqual(resolved["provider"], backend)
                    self.assertEqual(resolved["model"], model)
                    self.assertEqual(resolved["reasoning_effort"], effort)
                    self.assertEqual(resolved["api_key"], f"{backend}-test-key")
        with patch.dict(os.environ, {"GEMINI_API_KEY": "other-key", "OLLAMA_API_KEY": "other-key"}, clear=True):
            self.assertEqual(resolve_vlm_config(cfg)["api_key"], "EMPTY")

    @unittest.skipUnless(importlib.util.find_spec("mujoco"), "MuJoCo entrypoint requires its optional dependency")
    def test_cli_model_and_reasoning_overrides_win_over_environment(self):
        from scripts.run_mujoco import main

        environment = {"OPENAI_API_KEY": "test-key", "OPENAI_MODEL": "env-model",
                       "OPENAI_REASONING_EFFORT": "high"}
        for flags, expected_model, expected_effort in (
            ([], "env-model", "high"),
            (["--model", "cli-model", "--reasoning-effort", "medium"], "cli-model", "medium"),
        ):
            with self.subTest(flags=flags), TemporaryDirectory() as directory, \
                    patch.dict(os.environ, environment, clear=True), \
                    patch("scripts.run_mujoco.load_secrets_env"), \
                    patch("scripts.run_mujoco.MujocoSession") as session_factory, \
                    patch("scripts.run_mujoco.make_controller", return_value=Mock(cartesian_actions=None)), \
                    patch("scripts.run_mujoco.make_vlm_client") as client_factory, \
                    patch("scripts.run_mujoco.EpisodeLogger") as logger_factory, \
                    patch("scripts.run_mujoco.make_runner") as runner_factory:
                session_factory.return_value.recorder = None
                logger_factory.return_value.run_dir = Path(directory)
                runner_factory.return_value.run.return_value = EpisodeResult(
                    False, 0, "test_finished", "", directory,
                )
                self.assertEqual(main(["--max-steps", "1", "--log-dir", directory, *flags]), 1)
                cfg = client_factory.call_args.args[1]
                self.assertEqual(cfg["vlm_backend"], "openai")
                self.assertEqual(cfg["vlm"]["provider"], "openai")
                self.assertEqual(cfg["vlm"]["model"], expected_model)
                self.assertEqual(cfg["vlm"]["reasoning_effort"], expected_effort)
                self.assertEqual(cfg["vlm"]["api_key"], "test-key")
                client_factory.return_value.health_check.assert_called_once()
                runner_factory.return_value.run.assert_called_once()


class OpenAITransportTests(unittest.TestCase):
    def _client(self, *, temperature=0.0, reasoning_effort="medium"):
        client = VLMClient("https://api.openai.com/v1", "gpt-5.6-sol", "test-key",
                           30, 8192, temperature, provider="openai",
                           reasoning_effort=reasoning_effort,
                           chat_template_kwargs={"enable_thinking": True})
        self.addCleanup(client.session.close)
        return client

    def test_five_views_medium_reasoning_and_json_final_action_use_openai_payload(self):
        client = self._client(temperature=1.0)
        frames = [np.full((8, 8, 3), value, np.uint8) for value in (30, 90, 160, 230, 120)]
        answer = {"decision": "MV_FWD_SMALL", "reasoning": "Align the pin with the black bore."}
        agent = ControllerAgent(client, "{output_contract}", "", cartesian_actions=CartesianActions())
        with patch.object(client.session, "post", return_value=_reply(json.dumps(answer))) as post:
            result = agent.decide("Insert the plug", {}, "none", "NONE", "OPEN",
                                  frames[0], wrist_image=frames[1:], debug=True)
        self.assertEqual(result.token, "MV_FWD_SMALL")
        self.assertEqual(result.raw_text, json.dumps(answer, ensure_ascii=False, sort_keys=True))
        post.assert_called_once()
        self.assertEqual(post.call_args.args[0], "https://api.openai.com/v1/chat/completions")
        payload = post.call_args.kwargs["json"]
        self.assertEqual(set(payload), {"model", "messages", "max_completion_tokens", "reasoning_effort", "response_format"})
        self.assertEqual(payload["model"], "gpt-5.6-sol")
        self.assertEqual(payload["reasoning_effort"], "medium")
        self.assertEqual(payload["max_completion_tokens"], 8192)
        self.assertEqual(payload["response_format"], {"type": "json_object"})
        self.assertEqual(client.session.headers["Authorization"], "Bearer test-key")
        images = [part for part in payload["messages"][0]["content"] if part["type"] == "image_url"]
        self.assertEqual(len(images), 5)
        for part, expected in zip(images, frames):
            data = base64.b64decode(part["image_url"]["url"].split(",", 1)[1])
            with Image.open(io.BytesIO(data)) as image:
                np.testing.assert_array_equal(np.asarray(image), expected)

    def test_token_only_response_uses_final_content_and_full_reasoning_budget(self):
        client = self._client()
        with patch.object(client.session, "post", return_value=_reply("STOP")) as post:
            result = client.complete_token("Return STOP or MV_DOWN_LARGE", ["STOP", "MV_DOWN_LARGE"],
                                           None, debug=True)
        self.assertEqual(result.token, "STOP")
        post.assert_called_once()
        payload = post.call_args.kwargs["json"]
        self.assertEqual(set(payload), {"model", "messages", "max_completion_tokens", "reasoning_effort"})
        self.assertEqual(payload["max_completion_tokens"], 8192)
        self.assertEqual(payload["reasoning_effort"], "medium")

    def test_explicit_none_disables_native_reasoning_despite_local_template_flag(self):
        client = self._client(reasoning_effort="none")
        self.assertFalse(client.reasoning_enabled)
        with patch.object(client.session, "post", return_value=_reply("STOP")) as post:
            client.complete_token("Return STOP", ["STOP"], None)
        self.assertEqual(post.call_args.kwargs["json"]["reasoning_effort"], "none")
        self.assertNotIn("chat_template_kwargs", post.call_args.kwargs["json"])

    def test_reasoning_requests_omit_temperature_across_completion_methods(self):
        for temperature in (0.0, 1.0):
            for method in ("complete_json", "complete_text", "complete_token"):
                with self.subTest(temperature=temperature, method=method):
                    client = self._client(temperature=temperature)
                    content = '{"decision":"STOP"}' if method == "complete_json" else "STOP"
                    with patch.object(client.session, "post", return_value=_reply(content)) as post:
                        if method == "complete_token":
                            client.complete_token("Reply STOP", ["STOP"], None, debug=True)
                        else:
                            getattr(client, method)("Return a JSON action or STOP", None, debug=True)
                    payload = post.call_args.kwargs["json"]
                    self.assertNotIn("temperature", payload)
                    for field in ("max_tokens", "chat_template_kwargs", "guided_json", "guided_choice", "logprobs", "options", "think"):
                        self.assertNotIn(field, payload)
                    self.assertTrue(client.reasoning_enabled)


if __name__ == "__main__":
    unittest.main()
