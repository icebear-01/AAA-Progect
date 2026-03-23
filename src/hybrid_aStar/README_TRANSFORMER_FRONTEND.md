# Transformer-Guided Frontend

## Current runtime modes

- `planner/guided_frontend_backend:=python`
  Uses the existing Python bridge:
  - infer transformer guidance cost map in Python
  - run guided 2D A* in Python
  - send raw path back to C++
  - smooth in `HybridAStar::SmoothPath(...)`

- `planner/guided_frontend_backend:=onnx`
  Uses C++ runtime:
  - infer transformer guidance cost map in ONNX Runtime C++
  - run guided 8-connected A* in C++
  - send raw path to `HybridAStar::SmoothPath(...)`

## Export ONNX

The C++ ONNX frontend expects a simplified model that outputs the final 2D
guidance cost map directly.

The current export path uses a fixed-shape ONNX graph by default for
compatibility with the available PyTorch version. The C++ runtime resizes
occupancy/start/goal tensors to the model input size automatically and resizes
the predicted cost map back to the live planning grid.

Example:

```bash
export HYBRID_ASTAR_ROOT=/home/wmd/elevetor_demo0317/AAA-Progect/src/hybrid_aStar

PYTHONPATH=$HYBRID_ASTAR_ROOT/model_base_astar/neural-astar/src:$PYTHONPATH \
/path/to/python_with_torch \
$HYBRID_ASTAR_ROOT/model_base_astar/neural-astar/scripts/export_guidance_encoder_onnx.py \
  --ckpt $HYBRID_ASTAR_ROOT/model_base_astar/neural-astar/outputs/model_guidance_grid_mpd_unet_transformer_v3_gatedskip_formal_v1/best.pt \
  --out $HYBRID_ASTAR_ROOT/model_base_astar/neural-astar/outputs/model_guidance_grid_mpd_unet_transformer_v3_gatedskip_formal_v1/best_cost_map.onnx \
  --device cpu \
  --height 64 \
  --width 64 \
  --opset 16
```

## Run with ONNX frontend

```bash
source /opt/ros/noetic/setup.bash
source /home/wmd/elevetor_demo0317/AAA-Progect/build_hybrid_astar/devel/setup.bash

roslaunch /home/wmd/elevetor_demo0317/AAA-Progect/src/hybrid_aStar/launch/run_hybrid_a_star.launch \
  planner/use_transformer_guided_frontend:=true \
  planner/guided_frontend_backend:=onnx
```

## Notes

- ONNX Runtime headers/libs are expected at `/usr/local/onnxruntime`.
- Default launch still keeps `planner/guided_frontend_backend:=python`, so
  existing behavior is unchanged unless you explicitly switch.
