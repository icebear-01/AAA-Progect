#!/usr/bin/env python3
"""Convert a ROS map.yaml + image pair into guided-frontend request JSON.

Output JSON matches the schema consumed by:
`model_base_astar/neural-astar/scripts/run_transformer_guided_astar_frontend.py`
"""

from __future__ import annotations

import argparse
import ast
import json
from collections import deque
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw


GridXY = Tuple[int, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert ROS map to frontend request JSON.")
    parser.add_argument("--map-yaml", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--preview-png", type=Path, default=None)
    parser.add_argument("--start-world", type=float, nargs=2, default=None)
    parser.add_argument("--goal-world", type=float, nargs=2, default=None)
    parser.add_argument("--start-yaw", type=float, default=0.0)
    parser.add_argument("--goal-yaw", type=float, default=0.0)
    parser.add_argument("--unknown-as-obstacle", action="store_true", default=True)
    return parser.parse_args()


def _parse_simple_map_yaml(path: Path) -> Dict[str, object]:
    data: Dict[str, object] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if key == "image":
            data[key] = value
        elif key == "origin":
            data[key] = ast.literal_eval(value)
        elif key in {"resolution", "occupied_thresh", "free_thresh"}:
            data[key] = float(value)
        elif key == "negate":
            data[key] = int(value)
        else:
            data[key] = value
    required = {"image", "resolution", "origin", "negate", "occupied_thresh", "free_thresh"}
    missing = sorted(required.difference(data.keys()))
    if missing:
        raise ValueError(f"map yaml missing fields: {missing}")
    return data


def _load_ros_map_grayscale(image_path: Path) -> np.ndarray:
    with Image.open(image_path) as img:
        gray = img.convert("L")
        return np.asarray(gray, dtype=np.uint8)


def _ros_image_to_frontend_occupancy(
    image: np.ndarray,
    *,
    negate: int,
    occupied_thresh: float,
    free_thresh: float,
    unknown_as_obstacle: bool,
) -> np.ndarray:
    image_f = image.astype(np.float32) / 255.0
    if negate:
        occ_prob = image_f
    else:
        occ_prob = 1.0 - image_f

    occupied = occ_prob > float(occupied_thresh)
    free = occ_prob < float(free_thresh)
    unknown = ~(occupied | free)

    occ = np.zeros_like(image, dtype=np.uint8)
    occ[occupied] = 1
    if unknown_as_obstacle:
        occ[unknown] = 1

    # ROS image origin is top-left; frontend occupancy expects row-major y from map origin.
    return np.flipud(occ)


def _neighbors8(x: int, y: int, width: int, height: int) -> Iterable[GridXY]:
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            nx = x + dx
            ny = y + dy
            if 0 <= nx < width and 0 <= ny < height:
                yield nx, ny


def _largest_connected_component(free_mask: np.ndarray) -> List[GridXY]:
    height, width = free_mask.shape
    visited = np.zeros_like(free_mask, dtype=bool)
    best_component: List[GridXY] = []

    for y in range(height):
        for x in range(width):
            if not free_mask[y, x] or visited[y, x]:
                continue
            comp: List[GridXY] = []
            queue = deque([(x, y)])
            visited[y, x] = True
            while queue:
                cx, cy = queue.popleft()
                comp.append((cx, cy))
                for nx, ny in _neighbors8(cx, cy, width, height):
                    if visited[ny, nx] or not free_mask[ny, nx]:
                        continue
                    visited[ny, nx] = True
                    queue.append((nx, ny))
            if len(comp) > len(best_component):
                best_component = comp
    return best_component


def _farthest_cell(start: GridXY, allowed_mask: np.ndarray) -> Tuple[GridXY, int]:
    height, width = allowed_mask.shape
    dist = -np.ones((height, width), dtype=np.int32)
    sx, sy = start
    dist[sy, sx] = 0
    queue = deque([start])
    farthest = start
    farthest_dist = 0

    while queue:
        cx, cy = queue.popleft()
        base = int(dist[cy, cx])
        if base > farthest_dist:
            farthest = (cx, cy)
            farthest_dist = base
        for nx, ny in _neighbors8(cx, cy, width, height):
            if not allowed_mask[ny, nx] or dist[ny, nx] >= 0:
                continue
            dist[ny, nx] = base + 1
            queue.append((nx, ny))
    return farthest, farthest_dist


def _auto_pick_far_pair(occupancy: np.ndarray) -> Tuple[GridXY, GridXY, int]:
    free_mask = occupancy == 0
    component = _largest_connected_component(free_mask)
    if not component:
        raise RuntimeError("map has no reachable free component")

    comp_mask = np.zeros_like(free_mask, dtype=bool)
    for x, y in component:
        comp_mask[y, x] = True

    seed = component[0]
    a, _ = _farthest_cell(seed, comp_mask)
    b, dist = _farthest_cell(a, comp_mask)
    return a, b, dist


def _world_to_grid(
    x: float,
    y: float,
    origin_x: float,
    origin_y: float,
    resolution: float,
    width: int,
    height: int,
) -> GridXY:
    gx = int(np.floor((x - origin_x) / resolution))
    gy = int(np.floor((y - origin_y) / resolution))
    gx = max(0, min(width - 1, gx))
    gy = max(0, min(height - 1, gy))
    return gx, gy


def _grid_to_world(gx: int, gy: int, origin_x: float, origin_y: float, resolution: float) -> Tuple[float, float]:
    return (
        float(origin_x + (gx + 0.5) * resolution),
        float(origin_y + (gy + 0.5) * resolution),
    )


def _validate_free(label: str, grid_xy: GridXY, occupancy: np.ndarray) -> None:
    gx, gy = grid_xy
    if occupancy[gy, gx] != 0:
        raise ValueError(f"{label} lies on obstacle/unknown cell: grid={grid_xy}")


def _save_preview(path: Path, occupancy: np.ndarray, start_xy: GridXY, goal_xy: GridXY) -> None:
    h, w = occupancy.shape
    vis = np.where(np.flipud(occupancy) > 0, 30, 245).astype(np.uint8)
    rgb = np.stack([vis, vis, vis], axis=-1)
    img = Image.fromarray(rgb, mode="RGB")
    draw = ImageDraw.Draw(img)

    def draw_marker(xy: GridXY, color: Tuple[int, int, int]) -> None:
        gx, gy = xy
        px = gx
        py = h - 1 - gy
        r = max(2, min(w, h) // 60)
        draw.ellipse((px - r, py - r, px + r, py + r), fill=color, outline=(255, 255, 255))

    draw_marker(start_xy, (50, 200, 50))
    draw_marker(goal_xy, (220, 50, 50))
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def main() -> int:
    args = parse_args()
    map_cfg = _parse_simple_map_yaml(args.map_yaml)
    map_dir = args.map_yaml.parent
    image_path = (map_dir / str(map_cfg["image"])).resolve()
    if not image_path.exists():
        raise FileNotFoundError(f"map image not found: {image_path}")

    resolution = float(map_cfg["resolution"])
    origin = map_cfg["origin"]
    if not isinstance(origin, (list, tuple)) or len(origin) < 2:
        raise ValueError(f"invalid origin in yaml: {origin}")
    origin_x = float(origin[0])
    origin_y = float(origin[1])

    image = _load_ros_map_grayscale(image_path)
    occupancy = _ros_image_to_frontend_occupancy(
        image,
        negate=int(map_cfg["negate"]),
        occupied_thresh=float(map_cfg["occupied_thresh"]),
        free_thresh=float(map_cfg["free_thresh"]),
        unknown_as_obstacle=bool(args.unknown_as_obstacle),
    )
    height, width = occupancy.shape

    if args.start_world is not None and args.goal_world is not None:
        start_xy = _world_to_grid(args.start_world[0], args.start_world[1], origin_x, origin_y, resolution, width, height)
        goal_xy = _world_to_grid(args.goal_world[0], args.goal_world[1], origin_x, origin_y, resolution, width, height)
        auto_dist = -1
    else:
        start_xy, goal_xy, auto_dist = _auto_pick_far_pair(occupancy)

    _validate_free("start", start_xy, occupancy)
    _validate_free("goal", goal_xy, occupancy)

    start_world = _grid_to_world(start_xy[0], start_xy[1], origin_x, origin_y, resolution)
    goal_world = _grid_to_world(goal_xy[0], goal_xy[1], origin_x, origin_y, resolution)

    output_json = args.output_json or (map_dir / "map_frontend_request_auto.json")
    preview_png = args.preview_png or (map_dir / "map_frontend_request_auto_preview.png")
    output_json.parent.mkdir(parents=True, exist_ok=True)

    request = {
        "width": int(width),
        "height": int(height),
        "resolution": resolution,
        "origin_x": origin_x,
        "origin_y": origin_y,
        "start_world": [float(start_world[0]), float(start_world[1])],
        "goal_world": [float(goal_world[0]), float(goal_world[1])],
        "start_yaw": float(args.start_yaw),
        "goal_yaw": float(args.goal_yaw),
        "occupancy": occupancy.astype(int).tolist(),
    }
    output_json.write_text(json.dumps(request, ensure_ascii=False, indent=2), encoding="utf-8")
    _save_preview(preview_png, occupancy, start_xy, goal_xy)

    free_cells = int(np.sum(occupancy == 0))
    obstacle_cells = int(np.sum(occupancy > 0))
    print(f"saved_json={output_json}")
    print(f"saved_preview={preview_png}")
    print(f"map_size={width}x{height}")
    print(f"free_cells={free_cells}")
    print(f"obstacle_cells={obstacle_cells}")
    print(f"start_grid={start_xy}")
    print(f"goal_grid={goal_xy}")
    print(f"start_world=({start_world[0]:.3f}, {start_world[1]:.3f})")
    print(f"goal_world=({goal_world[0]:.3f}, {goal_world[1]:.3f})")
    if auto_dist >= 0:
        print(f"auto_pair_grid_distance={auto_dist}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
