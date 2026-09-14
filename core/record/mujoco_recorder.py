"""Synchronized simulation camera videos, a four-panel canvas, and annotations."""
from contextlib import ExitStack
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import textwrap

import numpy as np
from PIL import Image, ImageDraw

from core.record.episode_logger import _font, _jsonable, status_flags
from core.record.images import StreamingVideoWriter


CAMERAS = ("side", "wrist", "front")


class MujocoRecorder:
    def __init__(self, session, directory, *, fps=30, decision_hold_s=1.0, model="", task=""):
        self.fps = float(fps)
        hold = float(decision_hold_s)
        if not math.isfinite(self.fps) or not 1 <= self.fps <= 60:
            raise ValueError("recording.fps must be between 1 and 60")
        if not math.isfinite(hold) or not 0 <= hold <= 10:
            raise ValueError("recording.decision_hold_s must be between 0 and 10")
        self.session = session
        self.hold_frames = max(1, round(hold * self.fps))
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.paths = {name: self.directory / f"{name}.mp4" for name in (*CAMERAS, "combined")}
        self.paths["annotations"] = self.directory / "annotations.jsonl"
        self._resources = ExitStack()
        self._log = self._resources.enter_context(self.paths["annotations"].open("w", encoding="utf-8"))
        self.writers = {name: StreamingVideoWriter(self.paths[name], self.fps)
                        for name in (*CAMERAS, "combined")}
        for writer in self.writers.values():
            self._resources.callback(writer.close)
        self.frame_count = 0
        self._start_time = float(session.data.time)
        self._next_time = self._start_time + 1 / self.fps
        self._step_start = None
        self._closed = False
        self.context = {"stage": "Preparing", "decision": "WAIT", "source": "system",
                        "annotation": "Waiting for the first decision.", "vlm_output": ""}
        self.status = ""
        size = session.resolution
        self.font = _font(max(9, round(size * 18 / 512)))
        self.title_font = _font(max(9, round(size * 16 / 512)), bold=True)
        self.action_font = _font(max(12, round(size * 36 / 512)), bold=True)
        try:
            self.event("recording_started", fps=self.fps, model=model, task=task, cameras=CAMERAS,
                       video_files={key: str(value) for key, value in self.paths.items()})
            self.capture(force=True)
        except BaseException:
            self._resources.close()
            raise

    def event(self, name, **fields):
        entry = {"event": name, "utc": datetime.now(timezone.utc).isoformat(),
                 "sim_time_s": round(float(self.session.data.time) - self._start_time, 6),
                 "video_time_s": round(self.frame_count / self.fps, 6),
                 "frame": self.frame_count, **self.context, **fields}
        self._log.write(json.dumps(_jsonable(entry), ensure_ascii=False) + "\n")
        self._log.flush()

    def begin_step(self, *, step_idx, stage, token, annotation, output="", source="vlm"):
        self.context = {"step": step_idx, "stage": stage, "decision": token,
                        "executing": token, "source": source,
                        "annotation": str(annotation), "vlm_output": str(output)}
        self.status = "Executing"
        self._step_start = self.frame_count
        self.event("decision", frame_start=self._step_start)
        self.capture(force=True)

    def gripper_command(self, close):
        self.context["executing"] = "GRASP" if close else "RELEASE"
        self.event("gripper_command", annotation="Physical gripper close" if close else "Physical gripper open")

    def end_step(self, record):
        flags = [label for label, _ in status_flags(record)]
        self.status = "; ".join(flags) or "Action completed"
        self.capture(force=True, repeat=self.hold_frames)
        self.event("step_complete", frame_start=self._step_start, frame_end=self.frame_count,
                   annotation=self.status, result=record)
        self._step_start = None

    def capture(self, *, force=False, repeat=1):
        now = float(self.session.data.time)
        if self._closed or (not force and now + 1e-9 < self._next_time):
            return
        frames = {name: self.session.render(name) for name in CAMERAS}
        frames["combined"] = self._canvas(frames, now - self._start_time)
        for _ in range(repeat):
            for name, frame in frames.items():
                self.writers[name].append(frame)
            self.frame_count += 1
        while self._next_time <= now + 1e-9:
            self._next_time += 1 / self.fps

    def _canvas(self, frames, sim_time):
        size = self.session.resolution
        bar = 32
        canvas = Image.new("RGB", (2 * size, 2 * (size + bar)), (19, 25, 35))
        draw = ImageDraw.Draw(canvas)
        for i, (name, title) in enumerate(zip(CAMERAS, ("SIDE CAMERA", "WRIST CAMERA", "FRONT CAMERA"))):
            x, y = (i % 2) * size, (i // 2) * (size + bar)
            canvas.paste(Image.fromarray(frames[name]), (x, y + bar))
            draw.text((x + 12, y + 7), title, font=self.title_font, fill=(218, 230, 240))
        x, y = size, size + bar
        draw.text((x + 12, y + 7), "VLM OUTPUT / DECISION", font=self.title_font, fill=(45, 212, 191))
        panel = Image.new("RGB", (size, size), (19, 25, 35))
        text = ImageDraw.Draw(panel)
        margin = max(8, size // 24)
        line_h = self.font.size + 7
        text.text((margin, margin), f"STEP {self.context.get('step', '-')}  /  {self.context['source'].upper()}",
                  font=self.title_font, fill=(154, 172, 193))
        text.text((margin, margin + line_h), self.context["decision"],
                  font=self.action_font, fill=(45, 212, 191))
        cursor = margin + line_h + self.action_font.size + 15

        def paragraph(value, max_lines, color=(235, 241, 247)):
            nonlocal cursor
            width = max(6, int((size - 2 * margin) / max(1, self.font.getlength("W"))))
            lines = textwrap.wrap(" ".join(str(value).split()), width=width)
            if len(lines) > max_lines:
                lines = lines[:max_lines]
                lines[-1] = "... [full output in log]"
            for line in lines:
                text.text((margin, cursor), line, font=self.font, fill=color)
                cursor += line_h

        paragraph(self.context["stage"], 2, (154, 172, 193))
        cursor += 8
        available = max(1, (size - cursor - 4 * line_h) // line_h)
        paragraph(self.context["annotation"] or self.context["vlm_output"], available)
        footer = max(cursor + 8, size - 3 * line_h)
        text.text((margin, footer), f"SIM {sim_time:.2f}s  |  GRIP {self.session.get_gripper_position()[0] * 1000:.1f} mm",
                  font=self.title_font, fill=(154, 172, 193))
        text.text((margin, footer + line_h), f"Executing: {self.context.get('executing', '-')}",
                  font=self.title_font, fill=(154, 172, 193))
        text.text((margin, footer + 2 * line_h), self.status,
                  font=self.title_font, fill=(245, 190, 95))
        canvas.paste(panel, (x, y + bar))
        return np.asarray(canvas)

    def finish(self, success, end_reason):
        self.status = f"{'SUCCESS' if success else 'STOPPED'}: {end_reason}"
        try:
            self.capture(force=True, repeat=self.hold_frames)
            self.event("episode_complete", success=success, end_reason=end_reason, annotation=self.status)
        finally:
            self.close()

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            if self._step_start is not None:
                self.event("step_interrupted", frame_start=self._step_start, frame_end=self.frame_count,
                           annotation="Recording ended before this action completed.")
            self.event("recording_closed", frames=self.frame_count, duration_s=self.frame_count / self.fps)
        finally:
            self._resources.close()
