"""Shared task requirements for MuJoCo success checks and visual task prompts."""
from collections.abc import Mapping
import math


PLUG_RETREAT_CLEARANCE_M = 0.05
PLUG_RETREAT_MARGIN_M = 0.005


def plug_retreat_clearance_m(cfg: Mapping | None = None) -> float:
    """Read the configured strict TCP-to-handle-top clearance in metres."""
    section = (cfg or {}).get("plug_success", {})
    if not isinstance(section, Mapping):
        raise ValueError("plug_success must be a mapping")
    raw = section.get("retreat_clearance_m", PLUG_RETREAT_CLEARANCE_M)
    try:
        clearance = float(raw)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("plug_success.retreat_clearance_m must be finite and positive") from exc
    if isinstance(raw, bool) or not math.isfinite(clearance) or clearance <= 0:
        raise ValueError("plug_success.retreat_clearance_m must be finite and positive")
    return clearance


def plug_success_prompt(cfg: Mapping | None = None) -> str:
    """Expose the completion requirement, never measured target/object state."""
    clearance = plug_retreat_clearance_m(cfg)
    return (
        "PLUG COMPLETION CLEARANCE: after full insertion and release, keep the fingers "
        "open and retreat until the TCP (the grasp-point midpoint) is strictly more than "
        f"{clearance * 1000:g} mm vertically above the top of the seated orange handle. "
        f"Aim for at least {(clearance + PLUG_RETREAT_MARGIN_M) * 1000:g} mm to leave a margin. "
        "This is the world-Z gap from the handle top to the TCP, not merely the distance "
        "moved since RELEASE or the presence of a small visible gap. Confirm the seated "
        "plug remains stationary and complete this retreat before the final DONE."
    )
