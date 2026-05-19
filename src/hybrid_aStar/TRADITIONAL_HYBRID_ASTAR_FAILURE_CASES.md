# Traditional Hybrid A* HOPE Failure Cases

These cases are deterministic HOPE `Extrem` horizontal parallel-parking samples for showing that the traditional Hybrid A* frontend can spend a long time searching and still fail in difficult scenes.

Common setup:

```bash
source /opt/ros/noetic/setup.bash
source /home/wmd/elevetor_demo0317/AAA-Progect/build_hybrid_astar/devel/setup.bash

roslaunch /home/wmd/elevetor_demo0317/AAA-Progect/src/hybrid_aStar/launch/run_hope_extrem_hybrid_astar_rviz.launch \
  launch_rviz:=false exit_after_run:=true sample_attempts:=1 \
  seed:=<seed> warmup_cases:=<warmup_cases>
```

Planner settings are the defaults in `run_hope_extrem_hybrid_astar_rviz.launch`: `level=Extrem`, `case_id=1`, `horizontal_parallel_case=true`, `state_grid_resolution=0.6`, `map_grid_resolution=0.12`, `steering_angle=42.0`, `segment_length=0.75`, `phi_grid_size=72`.

## Case 1

```text
seed=1
warmup_cases=1
sample_attempts=1
start=(-5.04241, 4.16524, 2.95201)
goal=(2.82821, 1.32917, 3.10598)
obstacles=3
result=failed
observed_wall_time_ms=19527
```

Run:

```bash
roslaunch /home/wmd/elevetor_demo0317/AAA-Progect/src/hybrid_aStar/launch/run_hope_extrem_hybrid_astar_rviz.launch \
  seed:=1 warmup_cases:=1 sample_attempts:=1
```

Use this as a visual failure case. The RViz display shows the scene and the expanded search tree, but no successful path is produced.

## Case 2

```text
seed=1
warmup_cases=2
sample_attempts=1
start=(-4.35955, 4.39325, 2.96932)
goal=(2.82917, 1.42053, 3.11732)
obstacles=3
result=failed
observed_wall_time_ms=19361
```

Run:

```bash
roslaunch /home/wmd/elevetor_demo0317/AAA-Progect/src/hybrid_aStar/launch/run_hope_extrem_hybrid_astar_rviz.launch \
  seed:=1 warmup_cases:=2 sample_attempts:=1
```

This is another deterministic difficult scene with similar behavior: the traditional planner searches for about 19 seconds and returns failure.

## Notes

Do not use `seed=0` for documented failure cases. In this HOPE generator, `seed=0` means random initialization, so the scene is not reproducible across runs.
