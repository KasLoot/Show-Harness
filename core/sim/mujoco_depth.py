"""Metric wrist depth for numeric observations and grayscale diagnostics."""
from collections.abc import Mapping
import json
from numbers import Integral

import numpy as np


def validate_depth_range(near_m: float, far_m: float) -> None:
    if not np.isfinite([near_m, far_m]).all() or not 0 <= near_m < far_m:
        raise ValueError("wrist_depth requires finite 0 <= near_m < far_m")


def validate_depth_grid(grid_rows: int, grid_cols: int) -> None:
    for name, value in (("grid_rows", grid_rows), ("grid_cols", grid_cols)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value <= 0:
            raise ValueError(f"wrist_depth.{name} must be a positive integer")


def _calibration_values(metadata: Mapping) -> tuple[float, float, float]:
    near, far, tcp = (float(metadata[key]) for key in ("near_m", "far_m", "tcp_depth_m"))
    validate_depth_range(near, far)
    if not np.isfinite(tcp) or tcp <= 0:
        raise ValueError("Wrist camera-to-TCP depth must be finite and positive")
    return near, far, tcp


def depth_to_text(depth_m, calibration: Mapping, *, grid_rows=32, grid_cols=32) -> str:
    """Sample raw metric pixels into an immutable, spatially registered JSON grid.

    Each value comes from one pixel near a uniform cell center, with no averaging
    across surface boundaries. Small source images clamp the grid to their size,
    so an oversized request never invents or duplicates spatial samples.
    """
    validate_depth_grid(grid_rows, grid_cols)
    near, far, tcp = _calibration_values(calibration)
    depth = np.asarray(depth_m, dtype=float)
    if depth.ndim != 2 or min(depth.shape) <= 0:
        raise ValueError("Depth must be a nonempty two-dimensional metric image")
    height, width = depth.shape
    rows, cols = min(int(grid_rows), height), min(int(grid_cols), width)
    ys = np.floor((np.arange(rows) + 0.5) * height / rows).astype(int)
    xs = np.floor((np.arange(cols) + 0.5) * width / cols).astype(int)
    samples = depth[np.ix_(ys, xs)]
    valid = np.isfinite(samples) & (samples > 0) & (samples >= near) & (samples <= far)
    values = [[int(np.rint(value * 1000)) if usable else None
               for value, usable in zip(row, mask)] for row, mask in zip(samples, valid)]
    payload = {
        "name": "WRIST_DEPTH_MM",
        "sensor": "original Wrist camera depth buffer",
        "units": "mm along camera optical axis; rounded to nearest 1 mm",
        "source_width_px": width, "source_height_px": height,
        "requested_grid_rows": int(grid_rows), "requested_grid_cols": int(grid_cols),
        "grid_rows": rows, "grid_cols": cols,
        "sampling": "one pixel per uniform cell center; no averaging; grid clamped to source size",
        "pixel_indices": "zero-based pixel centers; rows top-to-bottom, columns left-to-right",
        "sample_x_px": xs.tolist(), "sample_y_px": ys.tolist(),
        "valid_range_mm": [near * 1000, far * 1000], "tcp_depth_mm": tcp * 1000,
        "null": "invalid, nonpositive, or outside configured range; unknown, never free space",
        "coverage": "sparse point samples can miss thin features and edges; unsampled and occluded pixels are unknown",
    }
    # One row per line preserves the 2D layout without expanding each sample onto
    # its own line. The complete result is ordinary JSON, including JSON nulls.
    header = json.dumps(payload, ensure_ascii=True, allow_nan=False, indent=2)
    grid = ",\n".join("    " + json.dumps(row, separators=(",", ":")) for row in values)
    return header[:-2] + ',\n  "depth_mm": [\n' + grid + "\n  ]\n}"


def depth_to_grayscale(depth_m, *, near_m=0.0, far_m=0.30) -> np.ndarray:
    """Encode optical-axis metres as near=white, far=black, never frame-normalized.

    Equal RGB channels keep the image compatible with the existing PNG/VLM/video
    transport. Invalid readings are black, like values at/beyond the far limit.
    """
    validate_depth_range(near_m, far_m)
    depth = np.asarray(depth_m, dtype=float)
    if depth.ndim != 2:
        raise ValueError("Depth must be a two-dimensional metric image")
    valid = np.isfinite(depth) & (depth > 0)
    safe = np.where(valid, depth, far_m)
    gray = np.rint(255 * (1 - np.clip((safe - near_m) / (far_m - near_m), 0, 1)))
    gray = np.where(valid, gray, 0).astype(np.uint8)
    return np.repeat(gray[:, :, None], 3, axis=2)


def depth_prompt(metadata: Mapping) -> str:
    """Describe sensor/robot calibration only, never target coordinates or depth."""
    near, far, tcp = _calibration_values(metadata)
    if metadata.get("representation") == "text":
        return (
            "Numeric Wrist Depth text contains raw depth-buffer point samples registered to "
            "original Wrist RGB (A) by the listed zero-based source pixel indices. Rows run "
            "top-to-bottom and columns left-to-right. Values are visible-surface distances "
            "along the camera optical axis in millimetres, rounded to 1 mm; they are not "
            "radial range, world height, or fingertip clearance. "
            f"The fixed valid range is {near * 1000:g} to {far * 1000:g} mm. "
            "Null means invalid, nonpositive, or outside that range; it is unknown, not "
            "zero distance or free space. Sampling does not average across object boundaries, "
            "but its sparse grid can miss thin features or edges and does not describe "
            "unsampled or occluded surfaces. "
            f"The TCP plane is {tcp * 1000:.1f} mm from this camera. "
            "Use RGB to identify sampled surfaces. A top surface near the TCP plane does "
            "not prove a centered body grasp: pads must surround the body below its top "
            "edge, with contact height confirmed in Front/Side. Camera and TCP rotate "
            "together; with a tilted wrist, depth is not world-Z clearance. Use SMALL "
            "corrections near contact and inspect again."
        )
    levels = np.array([255, 213, 170, 128, 85, 43, 0])
    distances = near + (1 - levels / 255) * (far - near)
    legend = "; ".join(f"{level}={distance * 1000:.1f} mm"
                       for level, distance in zip(levels, distances))
    return (
        "Wrist Depth (E) is registered pixel-for-pixel with original Wrist RGB (A). "
        f"Fixed linear grayscale: white (255) is the near limit {near * 1000:g} mm; "
        f"black (0) is {far * 1000:g} mm or farther, or invalid. "
        f"Gray-value legend: {legend}. The scale never auto-adjusts between requests. "
        "Depth is visible-surface distance along the camera optical axis, not radial range, "
        "world height, or fingertip clearance. "
        f"The TCP plane is {tcp * 1000:.1f} mm from this camera. "
        "The RGB image identifies which depth pixels belong to the object or fingers. "
        "A visible top surface near the TCP plane does not mean a centered body grasp: "
        "the fingertip pads must surround the body below its top edge. Compare the "
        "visible surface and finger depths, then confirm body contact height in Front/Side. "
        "The camera and TCP rotate together; with a tilted wrist, camera depth is not "
        "world-Z clearance. Occluded surfaces have no depth evidence, and black is not "
        "proof of free space. Use SMALL corrections near contact and inspect again."
    )
