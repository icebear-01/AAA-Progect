"""Planner-side confidence maps for learned residual heuristics."""

from __future__ import annotations

import numpy as np


def _box_filter_2d(arr: np.ndarray, kernel_size: int) -> np.ndarray:
    if int(kernel_size) <= 1:
        return np.asarray(arr, dtype=np.float32).copy()
    if int(kernel_size) % 2 == 0:
        raise ValueError(f"kernel_size must be odd, got {kernel_size}")

    pad = int(kernel_size) // 2
    padded = np.pad(np.asarray(arr, dtype=np.float32), pad_width=pad, mode="edge")
    integral = np.pad(padded, ((1, 0), (1, 0)), mode="constant").cumsum(0).cumsum(1)
    window_sum = (
        integral[kernel_size:, kernel_size:]
        - integral[:-kernel_size, kernel_size:]
        - integral[kernel_size:, :-kernel_size]
        + integral[:-kernel_size, :-kernel_size]
    )
    denom = float(int(kernel_size) * int(kernel_size))
    return (window_sum / max(denom, 1.0)).astype(np.float32)


def _max_filter_bool_2d(arr: np.ndarray, kernel_size: int) -> np.ndarray:
    if int(kernel_size) <= 1:
        return np.asarray(arr, dtype=bool).copy()
    if int(kernel_size) % 2 == 0:
        raise ValueError(f"kernel_size must be odd, got {kernel_size}")

    src = np.asarray(arr, dtype=bool)
    h, w = src.shape
    pad = int(kernel_size) // 2
    padded = np.pad(src, pad_width=pad, mode="edge")
    out = np.zeros_like(src, dtype=bool)
    for dy in range(int(kernel_size)):
        for dx in range(int(kernel_size)):
            out |= padded[dy : dy + h, dx : dx + w]
    return out


def build_residual_confidence_map(
    residual_map: np.ndarray,
    occ_map: np.ndarray,
    *,
    mode: str = "none",
    kernel_size: int = 5,
    strength: float = 0.75,
    min_confidence: float = 0.25,
) -> np.ndarray:
    """Build a planner-side confidence map from a predicted residual map.

    ``spike_suppression`` downweights locations where the predicted residual is
    much larger than its local neighborhood average, which helps suppress
    isolated overestimation spikes that can distort A* queue ordering.
    """
    residual = np.maximum(np.asarray(residual_map, dtype=np.float32), 0.0)
    occ = np.asarray(occ_map, dtype=np.float32)
    if residual.shape != occ.shape:
        raise ValueError(f"Shape mismatch: residual={residual.shape}, occ={occ.shape}")

    free_mask = occ < 0.5
    confidence = np.zeros_like(residual, dtype=np.float32)
    confidence[free_mask] = 1.0
    if mode == "none":
        return confidence

    if mode != "spike_suppression":
        raise ValueError(f"Unknown residual confidence mode: {mode}")
    if int(kernel_size) <= 0 or int(kernel_size) % 2 == 0:
        raise ValueError(f"kernel_size must be a positive odd integer, got {kernel_size}")

    smooth = _box_filter_2d(residual, kernel_size=int(kernel_size))
    smooth = np.maximum(smooth, 1e-4)
    spike_ratio = np.maximum(residual - smooth, 0.0) / smooth
    conf = 1.0 / (1.0 + float(strength) * spike_ratio)
    conf = np.clip(conf, float(min_confidence), 1.0).astype(np.float32)
    confidence[free_mask] = conf[free_mask]
    confidence[~free_mask] = 0.0
    return confidence.astype(np.float32)


def apply_confidence_safety_gate(
    confidence_map: np.ndarray | None,
    occ_map: np.ndarray,
    *,
    gate_threshold: float = 0.0,
    gate_kernel: int = 1,
    low_scale: float = 0.0,
    residual_map: np.ndarray | None = None,
    residual_min: float = 0.0,
) -> np.ndarray | None:
    """Hard-gate low-confidence residual regions back toward baseline A*.

    Cells with confidence below ``gate_threshold`` are downscaled by
    ``low_scale``. When ``gate_kernel`` > 1, the low-confidence mask is dilated
    to suppress risky residual neighborhoods instead of only isolated pixels.
    """
    if confidence_map is None:
        return None
    conf = np.clip(np.asarray(confidence_map, dtype=np.float32), 0.0, 1.0)
    occ = np.asarray(occ_map, dtype=np.float32)
    if conf.shape != occ.shape:
        raise ValueError(f"Shape mismatch: confidence={conf.shape}, occ={occ.shape}")
    if float(gate_threshold) <= 0.0:
        conf[occ > 0.5] = 0.0
        return conf.astype(np.float32)
    if int(gate_kernel) <= 0 or int(gate_kernel) % 2 == 0:
        raise ValueError(f"gate_kernel must be a positive odd integer, got {gate_kernel}")
    if not (0.0 <= float(low_scale) <= 1.0):
        raise ValueError(f"low_scale must be in [0, 1], got {low_scale}")

    low_conf = conf < float(gate_threshold)
    if float(residual_min) > 0.0:
        if residual_map is None:
            raise ValueError("residual_min > 0 requires residual_map")
        residual = np.asarray(residual_map, dtype=np.float32)
        if residual.shape != occ.shape:
            raise ValueError(f"Shape mismatch: residual={residual.shape}, occ={occ.shape}")
        low_conf &= residual >= float(residual_min)
    low_conf = _max_filter_bool_2d(low_conf, kernel_size=int(gate_kernel))
    gated = conf.copy()
    gated[low_conf] *= float(low_scale)
    gated[occ > 0.5] = 0.0
    return gated.astype(np.float32)


def resolve_residual_confidence_map(
    *,
    mode: str,
    occ_map: np.ndarray,
    residual_map: np.ndarray,
    learned_confidence_map: np.ndarray | None = None,
    kernel_size: int = 5,
    strength: float = 0.75,
    min_confidence: float = 0.25,
) -> np.ndarray | None:
    """Resolve planner-side residual confidence from learned and/or analytic sources."""
    if mode == "none":
        return None
    if mode == "learned":
        if learned_confidence_map is None:
            raise ValueError("mode=learned requires learned_confidence_map")
        return np.clip(np.asarray(learned_confidence_map, dtype=np.float32), 0.0, 1.0)
    if mode == "spike_suppression":
        return build_residual_confidence_map(
            residual_map=residual_map,
            occ_map=occ_map,
            mode="spike_suppression",
            kernel_size=kernel_size,
            strength=strength,
            min_confidence=min_confidence,
        )
    if mode == "learned_spike":
        if learned_confidence_map is None:
            raise ValueError("mode=learned_spike requires learned_confidence_map")
        spike = build_residual_confidence_map(
            residual_map=residual_map,
            occ_map=occ_map,
            mode="spike_suppression",
            kernel_size=kernel_size,
            strength=strength,
            min_confidence=min_confidence,
        )
        learned = np.clip(np.asarray(learned_confidence_map, dtype=np.float32), 0.0, 1.0)
        if learned.shape != spike.shape:
            raise ValueError(
                f"Shape mismatch: learned_confidence={learned.shape}, spike={spike.shape}"
            )
        return (learned * spike).astype(np.float32)
    raise ValueError(f"Unknown residual confidence mode: {mode}")
