"""API package for Route A guidance inference."""

from __future__ import annotations


def __getattr__(name: str):
    if name == "infer_cost_map":
        from .guidance_infer import infer_cost_map

        return infer_cost_map
    if name == "infer_cost_volume":
        from .guidance_infer import infer_cost_volume

        return infer_cost_volume
    raise AttributeError(name)


__all__ = ["infer_cost_map", "infer_cost_volume"]
