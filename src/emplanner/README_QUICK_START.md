# EMPlanner Quick Start

## 1. 编译

```bash
cd /home/wmd/elevator_car/P2P_fast_env_origin
catkin_make --pkg emplanner
source /opt/ros/noetic/setup.bash
source devel/setup.bash
```

## 2. 一键跑离线实验并出图

默认论文场景：

```bash
/home/wmd/elevator_car/P2P_fast_env_origin/src/planning/src/emplanner/scripts/run_compare_benchmark.sh
```

当前默认场景是 `19` 个 `l` 方向采样的复杂多障碍物配置。

默认论文 launch 现在会执行 `1` 次真实 `Plan()`，所以除了路径对比图，还会同时导出 `ST` 速度规划图。

直线路径 + 横穿障碍物的 `ST` 专用场景：

```bash
cd /home/wmd/elevator_car/P2P_fast_env_origin
source /opt/ros/noetic/setup.bash
source devel/setup.bash
roslaunch emplanner compare_benchmark_st_crossing.launch
```

按弯道档位切换：

```bash
/home/wmd/elevator_car/P2P_fast_env_origin/src/planning/src/emplanner/scripts/run_compare_benchmark.sh \
  turn_angle_case:=0
```

可选档位：`0`、`30`、`60`、`90`

一次性批量跑四组弯道：

```bash
/home/wmd/elevator_car/P2P_fast_env_origin/src/planning/src/emplanner/scripts/run_turn_angle_experiments.sh all
```

改场景名：

```bash
/home/wmd/elevator_car/P2P_fast_env_origin/src/planning/src/emplanner/scripts/run_compare_benchmark.sh \
  scenario_name:=my_case
```

只跑 benchmark，不出论文图：

```bash
/home/wmd/elevator_car/P2P_fast_env_origin/src/planning/src/emplanner/scripts/run_compare_benchmark.sh \
  --no-paper
```

结果目录：

```text
/home/wmd/elevator_car/P2P_fast_env_origin/src/planning/src/emplanner/benchmark_results/<scenario_name>/
```

常见输出文件：

- `comparison_paper.png / pdf`：路径与曲率对比图
- `st_graph_paper.png / pdf`：`ST` 速度规划图
- `summary.txt`：规划耗时、曲率指标、场景参数
- `obstacles.csv`：障碍物世界坐标
- `speed_profile.csv`：`QP` 速度规划结果
- `st_lattice.csv`：`ST` 采样栅格
- `st_obstacles.csv`：障碍物投影到 `ST` 后的信息

## 3. 用 roslaunch 跑

默认论文场景：

```bash
cd /home/wmd/elevator_car/P2P_fast_env_origin
source /opt/ros/noetic/setup.bash
source devel/setup.bash
roslaunch emplanner compare_benchmark_paper.launch
```

这条默认也是 `row_node_num:=19`。


自定义场景：

```bash
roslaunch emplanner compare_benchmark.launch scenario_name:=my_case
```

## 4. 实验常改参数

### 环境/参考线

- `use_straight_trajectory`
- `straight_start_x`
- `straight_start_y`
- `straight_length`
- `straight_step`
- `straight_turn_x`
- `straight_turn_angle_deg`
- `straight_turn_arc_length`
- `turn_angle_case`
- `turn_shape_case`
- `second_turn_gap`
- `second_turn_angle_deg`
- `second_turn_arc_length`
- `trajectory_file`

直线参考线例子：

```bash
roslaunch emplanner compare_benchmark.launch \
  use_straight_trajectory:=true \
  straight_length:=10.0 \
  straight_step:=0.1
```

S 弯参考线例子：

```bash
/home/wmd/elevator_car/P2P_fast_env_origin/src/planning/src/emplanner/scripts/run_compare_benchmark.sh \
  scenario_name:=s_curve_demo \
  turn_shape_case:=s_curve \
  turn_angle_case:=30 \
  straight_turn_x:=2.8 \
  straight_turn_arc_length:=1.4 \
  second_turn_gap:=0.6 \
  second_turn_arc_length:=1.4
```

### 规划长度

- `sample_s`
- `col_node_num`
- `row_node_num`

规划长度近似：

```text
planning_s_horizon ~= sample_s * col_node_num
```

例子：`sample_s=0.8`，`col_node_num=10`，长度约 `8.0 m`

### QP 平滑参数

- `w_qp_l`
- `w_qp_dl`
- `w_qp_ddl`
- `w_qp_ref_dp`

更平滑通常先增大：

- `w_qp_dl`
- `w_qp_ddl`

例子：

```bash
roslaunch emplanner compare_benchmark.launch \
  scenario_name:=smooth_case \
  w_qp_l:=800 \
  w_qp_dl:=1200 \
  w_qp_ddl:=5000 \
  w_qp_ref_dp:=50
```

### RL-DP 开关

- `use_rl_dp`

如果本地没有匹配模型，直接关掉：

```bash
use_rl_dp:=false
```

## 5. 障碍物设置

### 多障碍物，推荐

用逗号分隔参数：

- `obstacle_centers_x`
- `obstacle_centers_y`
- `obstacle_lengths`
- `obstacle_widths`
- `obstacle_yaws`
- `obstacle_x_vels`
- `obstacle_y_vels`

例子：

```bash
roslaunch emplanner compare_benchmark.launch \
  scenario_name:=multi_obs_case \
  obstacle_centers_x:="3.0,4.8,6.3" \
  obstacle_centers_y:="0.0,2.0,-0.55" \
  obstacle_lengths:="0.8,0.8,0.8" \
  obstacle_widths:="0.8,0.7,0.7" \
  obstacle_yaws:="0.06,-0.09,0.04"
```

规则：

- `obstacle_centers_x` 和 `obstacle_centers_y` 数量必须一致
- 其余参数数量可以少于障碍物数，缺省会补默认值

### 复用历史障碍物分布

直接读取某次实验保存下来的 `obstacles.csv`：

```bash
/home/wmd/elevator_car/P2P_fast_env_origin/src/planning/src/emplanner/scripts/run_compare_benchmark.sh \
  scenario_name:=reuse_obs_case \
  obstacle_csv:=/home/wmd/elevator_car/P2P_fast_env_origin/src/planning/src/emplanner/benchmark_results/turn_30_deg_arc/obstacles.csv
```

也可以直接把 `obstacles.csv` 路径当成第一个参数传进去：

```bash
/home/wmd/elevator_car/P2P_fast_env_origin/src/planning/src/emplanner/scripts/run_compare_benchmark.sh \
  /home/wmd/elevator_car/P2P_fast_env_origin/src/planning/src/emplanner/benchmark_results/turn_30_deg_arc/obstacles.csv \
  scenario_name:=reuse_obs_case
```

或者直接按历史场景名复用：

```bash
/home/wmd/elevator_car/P2P_fast_env_origin/src/planning/src/emplanner/scripts/run_compare_benchmark.sh \
  scenario_name:=reuse_obs_case \
  obstacle_source_scenario:=turn_30_deg_arc
```

这两个参数一旦设置，会优先于手动传入的 `obstacle_centers_x / obstacle_centers_y`。

### 单障碍物

```bash
roslaunch emplanner compare_benchmark.launch \
  scenario_name:=single_obs_case \
  obstacle_center_x:=3.0 \
  obstacle_center_y:=0.0 \
  default_obstacle_length:=0.8 \
  default_obstacle_width:=0.8 \
  obstacle_yaw:=0.05
```

### 坐标含义

- `x` 越大，障碍物越靠前
- `y = 0` 在参考线中间
- `y > 0` 在参考线左侧
- `y < 0` 在参考线右侧
- `yaw` 单位是弧度

如果你想在 `ST` 图里看到障碍物占据区域，给对应障碍物的速度模长要大于当前动态阈值 `0.6 m/s`，这样它才会被当前速度规划逻辑当作动态障碍物。

## 6. 手动出图

普通图：

```bash
python3 /home/wmd/elevator_car/P2P_fast_env_origin/src/planning/src/emplanner/scripts/plot_compare.py \
  --input-dir /home/wmd/elevator_car/P2P_fast_env_origin/src/planning/src/emplanner/benchmark_results/my_case
```

论文图：

```bash
python3 /home/wmd/elevator_car/P2P_fast_env_origin/src/planning/src/emplanner/scripts/plot_compare.py \
  --input-dir /home/wmd/elevator_car/P2P_fast_env_origin/src/planning/src/emplanner/benchmark_results/my_case \
  --paper \
  --export-pdf
```

`ST` 速度规划图：

```bash
python3 /home/wmd/elevator_car/P2P_fast_env_origin/src/planning/src/emplanner/scripts/plot_st_graph.py \
  --input-dir /home/wmd/elevator_car/P2P_fast_env_origin/src/planning/src/emplanner/benchmark_results/my_case \
  --paper \
  --export-pdf
```

如果目录里没有 `speed_profile.csv`，说明这次实验没有执行真实速度规划；直接重跑并加上：

```bash
plan_iterations:=1
```

## 7. 常用模板

论文默认场景：

```bash
/home/wmd/elevator_car/P2P_fast_env_origin/src/planning/src/emplanner/scripts/run_compare_benchmark.sh \
  scenario_name:=paper_case \
  use_rl_dp:=false \
  sample_s:=0.8 \
  col_node_num:=10 \
  row_node_num:=19 \
  w_qp_l:=800 \
  w_qp_dl:=1200 \
  w_qp_ddl:=5000 \
  w_qp_ref_dp:=50 \
  obstacle_centers_x:="3.0,4.8,6.3" \
  obstacle_centers_y:="0.0,2.0,-0.55" \
  obstacle_lengths:="0.8,0.8,0.8" \
  obstacle_widths:="0.8,0.7,0.7" \
  obstacle_yaws:="0.06,-0.09,0.04"
```

## 8. 相关文件

- 详细说明：[README.md](/home/wmd/elevator_car/P2P_fast_env_origin/src/planning/src/emplanner/README.md)
- 通用 launch：[compare_benchmark.launch](/home/wmd/elevator_car/P2P_fast_env_origin/src/planning/src/emplanner/launch/compare_benchmark.launch)
- 论文 launch：[compare_benchmark_paper.launch](/home/wmd/elevator_car/P2P_fast_env_origin/src/planning/src/emplanner/launch/compare_benchmark_paper.launch)
- 一键脚本：[run_compare_benchmark.sh](/home/wmd/elevator_car/P2P_fast_env_origin/src/planning/src/emplanner/scripts/run_compare_benchmark.sh)
