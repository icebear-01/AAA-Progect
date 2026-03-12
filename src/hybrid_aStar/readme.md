# hybrid_aStar

## Overview

This package now supports two frontends and one shared backend:

- Classic Hybrid A* frontend in C++
- Transformer-guided A* frontend
  - Python bridge mode
  - ONNX Runtime C++ mode
- Shared backend smoothing in `HybridAStar::SmoothPath(...)`

Current main flow:

- frontend selection: [hybrid_a_star_flow.cpp](/home/wmd/elevator_car/P2P_fast_env_origin/src/planning/src/hybrid_aStar/src/hybrid_a_star_flow.cpp)
- backend smoothing: [hybrid_a_star.cpp](/home/wmd/elevator_car/P2P_fast_env_origin/src/planning/src/hybrid_aStar/src/hybrid_a_star.cpp)
- ONNX frontend inference: [guided_frontend_onnx.cpp](/home/wmd/elevator_car/P2P_fast_env_origin/src/planning/src/hybrid_aStar/src/guided_frontend_onnx.cpp)

## Build

```bash
cd /home/wmd/elevator_car/P2P_fast_env_origin
source /opt/ros/noetic/setup.bash
catkin_make --pkg hybrid_a_star
source devel/setup.bash
```

Notes:

- The ONNX frontend links against ONNX Runtime under `/usr/local/onnxruntime`.
- If you only use the classic frontend, the runtime path still stays the same package.

## Run Classic Hybrid A*

```bash
source /opt/ros/noetic/setup.bash
source /home/wmd/elevator_car/P2P_fast_env_origin/devel/setup.bash

roslaunch /home/wmd/elevator_car/P2P_fast_env_origin/src/planning/src/hybrid_aStar/launch/run_hybrid_a_star.launch
```

Default behavior:

- `planner/use_transformer_guided_frontend:=false`
- planner uses the original Hybrid A* search
- final path is smoothed by `HybridAStar::SmoothPath(...)`

## Run Transformer Frontend In Python

This mode keeps inference and 2D guided A* in Python, then sends the raw path
back to C++ for smoothing.

```bash
source /opt/ros/noetic/setup.bash
source /home/wmd/elevator_car/P2P_fast_env_origin/devel/setup.bash

roslaunch /home/wmd/elevator_car/P2P_fast_env_origin/src/planning/src/hybrid_aStar/launch/run_hybrid_a_star.launch \
  planner/use_transformer_guided_frontend:=true \
  planner/guided_frontend_backend:=python
```

## Run Transformer Frontend In ONNX Runtime C++

This is the recommended deployment path.

The flow is:

- export transformer guidance model to ONNX once
- run guidance inference in ONNX Runtime C++
- run guided 8-connected A* in C++
- smooth the result in `HybridAStar::SmoothPath(...)`

### 1. Export ONNX

Use the `neural-astar-gpu` conda environment:

```bash
eval "$(/home/wmd/anaconda3/bin/conda shell.bash hook)"
conda activate neural-astar-gpu
export PYTHONPATH=/home/wmd/elevator_car/P2P_fast_env_origin/src/planning/src/hybrid_aStar/model_base_astar/neural-astar/src:$PYTHONPATH

python /home/wmd/elevator_car/P2P_fast_env_origin/src/planning/src/hybrid_aStar/model_base_astar/neural-astar/scripts/export_guidance_encoder_onnx.py \
  --ckpt /home/wmd/elevator_car/P2P_fast_env_origin/src/planning/src/hybrid_aStar/model_base_astar/neural-astar/outputs/model_guidance_grid_mpd_unet_transformer_v3_gatedskip_formal_v1/best.pt \
  --out /home/wmd/elevator_car/P2P_fast_env_origin/src/planning/src/hybrid_aStar/model_base_astar/neural-astar/outputs/model_guidance_grid_mpd_unet_transformer_v3_gatedskip_formal_v1/best_cost_map.onnx \
  --device cpu \
  --height 64 \
  --width 64 \
  --opset 16
```

Current exported model path:

- [best_cost_map.onnx](/home/wmd/elevator_car/P2P_fast_env_origin/src/planning/src/hybrid_aStar/model_base_astar/neural-astar/outputs/model_guidance_grid_mpd_unet_transformer_v3_gatedskip_formal_v1/best_cost_map.onnx)

Important implementation note:

- ONNX export uses a fixed `64x64` graph for compatibility with the available PyTorch version.
- The C++ runtime automatically rescales occupancy, start, and goal maps to `64x64`, runs ONNX inference, then rescales the predicted guidance map back to the live planning grid.

### 2. Run ONNX Frontend

```bash
source /opt/ros/noetic/setup.bash
source /home/wmd/elevator_car/P2P_fast_env_origin/devel/setup.bash

roslaunch /home/wmd/elevator_car/P2P_fast_env_origin/src/planning/src/hybrid_aStar/launch/run_hybrid_a_star.launch \
  planner/use_transformer_guided_frontend:=true \
  planner/guided_frontend_backend:=onnx
```

## Run Offline Demo With C++ Frontend + C++ Backend

This is the new fully offline path for:

- street dataset loading in C++
- ONNX guidance inference in C++
- guided grid A* in C++
- XY seed generation in C++
- backend smoothing in C++
- Python only for final plotting

Binary:

- [offline_guided_astar_cpp_demo.cpp](/home/wmd/elevator_car/P2P_fast_env_origin/src/planning/src/hybrid_aStar/app/offline_guided_astar_cpp_demo.cpp)

Example:

```bash
cd /home/wmd/elevator_car/P2P_fast_env_origin
source /opt/ros/noetic/setup.bash
source devel/setup.bash
eval "$(/home/wmd/anaconda3/bin/conda shell.bash hook)"
conda activate neural-astar-gpu

/home/wmd/elevator_car/P2P_fast_env_origin/devel/lib/hybrid_a_star/offline_guided_astar_cpp_demo \
  --split train \
  --map-index -1 \
  --seed 2468 \
  --case-name street_demo_cpp \
  --python-exec /home/wmd/anaconda3/bin/python
```

Notes:

- `--map-index -1` means random street map.
- `--seed` controls random map index and random start/goal sampling.
- Python is only used to read generated CSV/YAML and save the final figures.
- If you do not want plotting, add `--skip-plot`.

Outputs are written under:

- [offline_results/street_guided_demo](/home/wmd/elevator_car/P2P_fast_env_origin/src/planning/src/hybrid_aStar/offline_results/street_guided_demo)

For example:

- `frontend_raw_path.csv`
- `frontend_seed_path.csv`
- `smoothed_path.csv`
- `occupancy.csv`
- `guidance_cost.csv`
- `meta.json`
- `offline_demo.png`
- `offline_demo_paper.png`
- `offline_demo_split_points.png`
- `curvature_compare.png`

## Run Random Map + RViz + Transformer-Guided A*

This is the direct entry for the workflow you asked for:

- generate one frontend-style random occupancy map
- publish that map to RViz and to the planner
- set start pose with `2D Pose Estimate`
- set goal with `2D Nav Goal`
- run transformer-guided A* frontend
- smooth with the existing backend optimizer

```bash
source /opt/ros/noetic/setup.bash
source /home/wmd/elevator_car/P2P_fast_env_origin/devel/setup.bash

roslaunch /home/wmd/elevator_car/P2P_fast_env_origin/src/planning/src/hybrid_aStar/launch/run_transformer_guided_astar_random_env.launch
```

Files:

- launch: [run_transformer_guided_astar_random_env.launch](/home/wmd/elevator_car/P2P_fast_env_origin/src/planning/src/hybrid_aStar/launch/run_transformer_guided_astar_random_env.launch)
- rviz: [guided_astar_random_env.rviz](/home/wmd/elevator_car/P2P_fast_env_origin/src/planning/src/hybrid_aStar/rviz/guided_astar_random_env.rviz)

Default behavior of this launch:

- frontend random occupancy map sampled once on startup
- `auto_refresh=false`
- transformer-guided frontend enabled
- backend set to `onnx`
- fallback to classic Hybrid A* disabled
- planner static map source set to `/guided_frontend_random_map`

If you want the random scene to refresh automatically:

```bash
roslaunch /home/wmd/elevator_car/P2P_fast_env_origin/src/planning/src/hybrid_aStar/launch/run_transformer_guided_astar_random_env.launch \
  auto_refresh:=true \
  refresh_period:=2.0
```

If you want to start it without RViz first:

```bash
roslaunch /home/wmd/elevator_car/P2P_fast_env_origin/src/planning/src/hybrid_aStar/launch/run_transformer_guided_astar_random_env.launch \
  launch_rviz:=false
```

If you want to allow fallback to classic Hybrid A*:

```bash
roslaunch /home/wmd/elevator_car/P2P_fast_env_origin/src/planning/src/hybrid_aStar/launch/run_transformer_guided_astar_random_env.launch \
  fallback_to_hybrid_astar:=true
```

## Run Street Map + RViz + Transformer-Guided A*

This entry uses the existing street benchmark dataset under neural-astar
planning-datasets, publishes one street map as `OccupancyGrid` from C++, and
lets you set start/goal in RViz:

```bash
source /opt/ros/noetic/setup.bash
source /home/wmd/elevator_car/P2P_fast_env_origin/devel/setup.bash

roslaunch /home/wmd/elevator_car/P2P_fast_env_origin/src/planning/src/hybrid_aStar/launch/run_transformer_guided_astar_street_env.launch
```

Files:

- launch: [run_transformer_guided_astar_street_env.launch](/home/wmd/elevator_car/P2P_fast_env_origin/src/planning/src/hybrid_aStar/launch/run_transformer_guided_astar_street_env.launch)
- rviz: [guided_astar_street_env.rviz](/home/wmd/elevator_car/P2P_fast_env_origin/src/planning/src/hybrid_aStar/rviz/guided_astar_street_env.rviz)
- map publisher: [frontend_street_map_env.cpp](/home/wmd/elevator_car/P2P_fast_env_origin/src/planning/src/hybrid_aStar/app/frontend_street_map_env.cpp)

Usage in RViz:

- click `2D Pose Estimate` to set the start pose
- click `2D Nav Goal` to set the goal pose
- planner uses transformer-guided A* frontend in ONNX Runtime C++ and the
  existing backend smoother in C++

Useful overrides:

```bash
roslaunch /home/wmd/elevator_car/P2P_fast_env_origin/src/planning/src/hybrid_aStar/launch/run_transformer_guided_astar_street_env.launch \
  street_split:=valid \
  street_map_index:=12
```

Random street sample on each startup:

```bash
roslaunch /home/wmd/elevator_car/P2P_fast_env_origin/src/planning/src/hybrid_aStar/launch/run_transformer_guided_astar_street_env.launch \
  random_index:=true
```

Street launch defaults to the street-specific guidance model:

- ckpt: [best.pt](/home/wmd/elevator_car/P2P_fast_env_origin/src/planning/src/hybrid_aStar/model_base_astar/neural-astar/outputs/model_guidance_street/best.pt)
- onnx: [best_cost_map.onnx](/home/wmd/elevator_car/P2P_fast_env_origin/src/planning/src/hybrid_aStar/model_base_astar/neural-astar/outputs/model_guidance_street/best_cost_map.onnx)

## Key Parameters

These parameters are defined in:

- [run_hybrid_a_star.launch](/home/wmd/elevator_car/P2P_fast_env_origin/src/planning/src/hybrid_aStar/launch/run_hybrid_a_star.launch)

Most important ones:

- `planner/use_transformer_guided_frontend`
  Turn transformer-guided frontend on or off.

- `planner/guided_frontend_backend`
  `python` or `onnx`.

- `planner/guided_frontend_ckpt`
  PyTorch checkpoint for Python frontend or ONNX export source.

- `planner/guided_frontend_onnx`
  ONNX model path for C++ runtime.

- `planner/fallback_to_hybrid_astar`
  If guided frontend fails, fall back to classic Hybrid A*.

- `planner/guided_frontend_lambda`
  Weight of transformer guidance in guided grid A*.

- `planner/guided_frontend_heuristic_mode`
  Heuristic mode for grid A* such as `octile`.

- `planner/guided_frontend_integration_mode`
  Guidance integration mode in guided A*, for example `g_cost`.

## Random Parking Environment

For RViz preview and dataset-style testing:

```bash
source /opt/ros/noetic/setup.bash
source /home/wmd/elevator_car/P2P_fast_env_origin/devel/setup.bash

roslaunch /home/wmd/elevator_car/P2P_fast_env_origin/src/planning/src/hybrid_aStar/launch/random_parking_env.launch
```

This helper publishes randomized parking-slot scenes and ego/goal topics for
planner testing.

## Files To Start From

If you want to continue developing this integration, read these first:

- [hybrid_a_star_flow.cpp](/home/wmd/elevator_car/P2P_fast_env_origin/src/planning/src/hybrid_aStar/src/hybrid_a_star_flow.cpp)
- [hybrid_a_star.cpp](/home/wmd/elevator_car/P2P_fast_env_origin/src/planning/src/hybrid_aStar/src/hybrid_a_star.cpp)
- [guided_frontend_onnx.cpp](/home/wmd/elevator_car/P2P_fast_env_origin/src/planning/src/hybrid_aStar/src/guided_frontend_onnx.cpp)
- [export_guidance_encoder_onnx.py](/home/wmd/elevator_car/P2P_fast_env_origin/src/planning/src/hybrid_aStar/model_base_astar/neural-astar/scripts/export_guidance_encoder_onnx.py)
- [README_TRANSFORMER_FRONTEND.md](/home/wmd/elevator_car/P2P_fast_env_origin/src/planning/src/hybrid_aStar/README_TRANSFORMER_FRONTEND.md)
