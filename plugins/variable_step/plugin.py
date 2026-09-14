"""Variable step-size tool: a controller *motion-magnitude* provider.

The controller normally moves a fixed ``step_m`` per atomic token. When this tool is
enabled it returns a COARSE step (e.g. 5 cm) instead of the fine ``step_m`` in these cases:

  VERTICAL MOTION:
    * the token is ``MV_UP`` -- lift clear of the table quickly; and
    * ``MV_DOWN`` while high above the table (gap > ``high_above_table_m``) -- a
      coarse descent from altitude.

  WRIST-VISIBILITY (the VLM's distance signal):
    * the TARGET is NOT yet visible in the wrist view (``target_in_wrist`` False) -- the
      gripper is still far, so close distance with a big step. ``target_in_wrist`` is the
      shared wrist-visibility judgment normalized by the controller agent from its JSON
      boolean or legacy ``WRIST: YES/NO`` marker and forwarded by the runner.

Height alone must never force a coarse horizontal alignment: that produces repeated
overshoot even while the target is visible. Horizontal moves are fine unless the target
is explicitly reported outside the wrist view. Missing visibility defaults to fine.

When ``large_step_m`` is configured, travel is promoted to that third size only above
``large_above_table_m`` clearance. Omitting it preserves the original two-size policy.
"""
from __future__ import annotations

import math
from typing import Optional


MV_UP = "MV_UP"


class VariableStepPlugin:
    """Pick the per-command translation magnitude from height, token, and wrist visibility."""

    def __init__(
        self,
        enabled: bool = False,
        coarse_step_m: float = 0.05,
        high_above_table_m: float = 0.10,
        large_step_m: Optional[float] = None,
        large_above_table_m: float = 0.20,
    ) -> None:
        self.enabled = bool(enabled)
        self.coarse_step_m = max(0.0, float(coarse_step_m))
        self.high_above_table_m = max(0.0, float(high_above_table_m))
        self.large_step_m = None if large_step_m is None else float(large_step_m)
        self.large_above_table_m = float(large_above_table_m)
        if self.large_step_m is not None and (
            not math.isfinite(self.large_step_m) or not math.isfinite(self.large_above_table_m)
            or self.large_step_m <= self.coarse_step_m
            or self.large_above_table_m < self.large_step_m
        ):
            raise ValueError("Require finite large_step_m > coarse_step_m and large_above_table_m >= large_step_m")

    def step_m_for(
        self,
        token: str,
        default_step_m: float,
        eef_height_m: Optional[float] = None,
        table_height_m: Optional[float] = None,
        target_in_wrist: Optional[bool] = None,
    ) -> float:
        """Return the translation magnitude (meters) to use for ``token``.

        Coarse for lifting, descent from altitude, or an explicitly distant target.
        Horizontal alignment with visible/unknown targets always uses the fine step.
        """
        default = float(default_step_m)
        if not self.enabled:
            return default
        token = str(token or "").strip().upper()
        gap = None
        if eef_height_m is not None and table_height_m is not None:
            try:
                gap = float(eef_height_m) - float(table_height_m)
            except (TypeError, ValueError):
                pass
        travel = self.coarse_step_m
        if self.large_step_m is not None and gap is not None and math.isfinite(gap) and gap > self.large_above_table_m:
            travel = self.large_step_m
        if token == MV_UP:
            return travel
        if token == "MV_DOWN" and gap is not None and gap > self.high_above_table_m:
            return travel
        # --- Wrist-visibility rule: far (TARGET not in the wrist view) -> big step ---
        if target_in_wrist is False:
            return travel
        return default
