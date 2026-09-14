"""Variable step-size capability: coarse vertical travel or explicitly distant targets;
fine horizontal alignment when the target is visible or visibility is unknown.

Public API:
  * ``VariableStepPlugin`` -- maps (token, EEF height, wrist visibility) to the per-command
    step magnitude (``step_m_for``). The wrist-visibility signal it consumes is the shared
    ``target_in_wrist`` judgment from :mod:`core.prompting.wrist_marker` (rendered/parsed by the
    controller agent), not owned here.

See :mod:`plugins.variable_step.plugin` for the implementation.
"""
from .plugin import VariableStepPlugin

__all__ = ["VariableStepPlugin"]
