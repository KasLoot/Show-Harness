"""Common, passive instrumentation for one-factor MuJoCo/VLM experiments.

Each treatment changes one input factor: an image, a camera description, or the
visual GRASP condition. Raw physics states are recorded to a separate file for
analysis after the episode, never to the VLM.
"""
from __future__ import annotations

import base64
import copy
from contextlib import nullcontext
from datetime import datetime, timezone
import hashlib
import io
import json
import re
import time

import numpy as np
from PIL import Image

from core.record.images import image_to_data_url, save_png


SIDE_CAMERA_DESCRIPTION = (
    "CAMERA INPUTS: Image 1 is the front (AgentView) camera; image 2 is the wrist camera; "
    "image 3 is a fixed side camera. In image 3, MV_FWD projects right, MV_BACK left, "
    "MV_UP up, and MV_DOWN down; this view shows forward/backward alignment and height. "
    "All three images are simultaneous."
)
GRASP_RULE_PATTERN = (
    r"^- GRASP when BOTH AgentView and Wrist view confirm the (.+) "
    r"is clearly between the center of two grippers$"
)
SIDE_GRASP_RULE_TEMPLATE = (
    "- GRASP only when AgentView and Wrist show the {affordance} between the fingers, "
    "AND the fixed side view (image 3) confirms the finger pads are horizontally centered "
    "on that body and vertically overlap its middle. If that side-view check fails, "
    "correct alignment or height before closing."
)


def side_camera_spec():
    position = np.array([0.45, -0.65, 0.25])
    look_at = np.array([0.45, 0.0, 0.22])  # fixed workspace point, not an object tracker
    forward = look_at - position
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0, 0, 1])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    return {"name": "ablation_side", "position": position.tolist(),
            "xyaxes": np.r_[right, up].tolist(), "fovy": 45.0,
            "look_at": look_at.tolist(), "image_size": [256, 256]}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def add_side_image(payload, image, enabled):
    """The complete experimental manipulation: append one raw image, no text."""
    if not enabled:
        return payload
    if image is None:
        raise RuntimeError("Side-camera input was not captured before the VLM request")
    result = copy.deepcopy(payload)
    candidates = [m for m in result["messages"] if isinstance(m.get("content"), list)
                  and any(p.get("type") == "image_url" for p in m["content"])]
    if len(candidates) != 1:
        raise ValueError("Expected one multimodal user message")
    content = candidates[0]["content"]
    slots = [i for i, part in enumerate(content) if part.get("type") == "image_url"]
    if len(slots) != 2:
        raise ValueError(f"Expected the unchanged front/wrist pair, got {len(slots)} images")
    content.insert(slots[-1] + 1, {"type": "image_url", "image_url": {"url": image_to_data_url(image)}})
    return result


def add_camera_description(payload, enabled):
    """A separate factor: one fixed camera-description paragraph, no policy edits."""
    if not enabled:
        return payload
    result = copy.deepcopy(payload)
    candidates = [m for m in result["messages"] if isinstance(m.get("content"), list)
                  and any(p.get("type") == "image_url" for p in m["content"])]
    if len(candidates) != 1:
        raise ValueError("Expected one multimodal user message")
    content = candidates[0]["content"]
    if sum(p.get("type") == "image_url" for p in content) != 3:
        raise ValueError("The camera description requires the three-camera input")
    content.append({"type": "text", "text": SIDE_CAMERA_DESCRIPTION})
    return result


def add_side_grasp_check(payload, enabled):
    """Replace one visual GRASP condition; the VLM still decides every action."""
    if not enabled:
        return payload
    matches = []
    for mi, message in enumerate(payload["messages"]):
        if not isinstance(message.get("content"), list):
            continue
        for pi, part in enumerate(message["content"]):
            if part.get("type") == "text" and re.search(GRASP_RULE_PATTERN, part["text"], re.M):
                matches.append((mi, pi))
    if not matches:
        return payload  # planner requests have no controller GRASP condition
    if len(matches) != 1:
        raise ValueError("Expected one controller GRASP condition")
    result = copy.deepcopy(payload)
    mi, pi = matches[0]
    part = result["messages"][mi]["content"][pi]
    part["text"], count = re.subn(
        GRASP_RULE_PATTERN,
        lambda match: SIDE_GRASP_RULE_TEMPLATE.format(affordance=match.group(1)),
        part["text"], flags=re.M)
    if count != 1:
        raise ValueError("Expected exactly one controller GRASP condition")
    return result


class AblationInstrumentation:
    def __init__(self, send_side: bool, describe_side: bool = False, check_side_grasp: bool = False):
        self.send_side = bool(send_side)
        self.describe_side = bool(describe_side)
        self.check_side_grasp = bool(check_side_grasp)
        if self.describe_side and not self.send_side:
            raise ValueError("The side-camera description requires the side image")
        if self.check_side_grasp and not (self.send_side and self.describe_side):
            raise ValueError("The side-view grasp check requires the side image and camera description")
        self.side = None
        self.run_dir = None
        self.request_count = 0
        self.events = None
        self.last_request = None
        self.action_count = 0

    def install(self, task, session, controller, logger, client):
        self.run_dir = logger.run_dir
        self.request_dir = self.run_dir / "requests"
        self.request_dir.mkdir()
        self.events = (self.run_dir / "physics_states.jsonl").open("w")
        self.secret = str(client._api_key)
        has_side = any(c["name"] == "ablation_side" for c in task.cfg.get("observer_cameras", []))
        self.side = task.render_observer("ablation_side") if has_side else None
        if self.side is not None:
            save_png(self.run_dir / "initial_side.png", self.side)
        # A fixed model snapshot plus state vectors makes post-run diagnostics
        # reproducible; no geometric metric is evaluated in the control loop.
        from core.sim.mujoco_task import build_model_xml
        xml, _ = build_model_xml(task.cfg)
        (self.run_dir / "model_snapshot.xml").write_text(xml)
        original_post = client.session.post

        def post(url, *args, **kwargs):
            payload = add_side_image(kwargs["json"], self.side, self.send_side)
            payload = add_camera_description(payload, self.describe_side)
            payload = add_side_grasp_check(payload, self.check_side_grasp)
            kwargs["json"] = payload
            self.request_count += 1
            self.last_request = self.request_count
            directory = self.request_dir / f"{self.request_count:04d}"
            directory.mkdir()
            image_parts, text_parts = [], []
            for message in payload["messages"]:
                content = message.get("content", [])
                if isinstance(content, str):
                    text_parts.append(content)
                    continue
                for part in content:
                    if part.get("type") == "image_url":
                        image_parts.append(part)
                    elif part.get("type") == "text":
                        text_parts.append(part["text"])
            expected = 3 if self.send_side else 2
            if len(image_parts) != expected:
                raise ValueError(f"Ablation request has {len(image_parts)} images, expected {expected}")
            inputs = []
            for index, part in enumerate(image_parts):
                raw = base64.b64decode(part["image_url"]["url"].split(",", 1)[1])
                path = directory / f"image_{index}.png"
                path.write_bytes(raw)
                pixels = np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"))
                inputs.append({"slot": index, "camera": ["front", "wrist", "side"][index],
                               "file": path.name, "shape": list(pixels.shape),
                               "pixels_sha256": hashlib.sha256(pixels.tobytes()).hexdigest()})
            prompt = "\n\n".join(text_parts)
            (directory / "prompt.txt").write_text(prompt)
            stage = re.search(r"^STAGE:\s*(.*)$", prompt, flags=re.M)
            record = {"request": self.request_count, "send_side": self.send_side,
                      "describe_side": self.describe_side,
                      "check_side_grasp": self.check_side_grasp,
                      "role": "planner" if "ROLE: SubgoalPlanner" in prompt else "controller",
                      "stage": stage.group(1) if stage else None, "images": inputs,
                      "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                      "payload_sha256": digest(payload),
                      "parameters": {k: v for k, v in payload.items() if k != "messages"},
                      "started_utc": datetime.now(timezone.utc).isoformat(), "status": "sent"}
            (directory / "request.json").write_text(json.dumps(record, indent=2))
            started = time.monotonic()
            try:
                response = original_post(url, *args, **kwargs)
                record.update(status="received", http_status=response.status_code)
                try:
                    data = response.json()
                    text = json.dumps(data, indent=2)
                    record["usage"] = data.get("usage")
                    record["response_model"] = data.get("model")
                except (ValueError, TypeError):
                    text = response.text
                (directory / "response.txt").write_text(self.redact(text))
                return response
            except Exception as exc:
                record.update(status="request_error", error=self.redact(str(exc)))
                raise
            finally:
                record["elapsed_s"] = time.monotonic() - started
                (directory / "request.json").write_text(self.redact(json.dumps(record, indent=2)))

        client.session.post = post
        original_step = controller.step

        def state():
            return {"time": float(task.data.time), "qpos": task.data.qpos.tolist(),
                    "qvel": task.data.qvel.tolist(), "ctrl": task.data.ctrl.tolist()}

        def step(token, *args, **kwargs):
            event = {"action": self.action_count, "token": token, "request": self.last_request,
                     "before": state()}
            self.action_count += 1
            try:
                result = original_step(token, *args, **kwargs)
                event.update(grasp_empty=result.grasp_empty, note=result.note)
                return result
            except Exception as exc:
                event["error"] = self.redact(str(exc))
                raise
            finally:
                event["after"] = state()
                self.events.write(json.dumps(event) + "\n")
                self.events.flush()

        controller.step = step
        instrumentation = self

        class Session:
            robot = session.robot
            config = session.config

            def get_observation(self):
                with getattr(task, "lock", nullcontext()):
                    observation = session.get_observation()
                    if has_side:
                        instrumentation.side = task.render_observer("ablation_side")
                    return observation

        return Session()

    def redact(self, value):
        return value.replace(self.secret, "[REDACTED]") if self.secret else value

    def close(self):
        if self.events is not None and not self.events.closed:
            self.events.close()
