# EMPlanner README

## 1. 说明

`emplanner` 包含两套使用方式：

- 在线规划节点：`emplanner_run`
  依赖真实/仿真的 ROS 话题输入，按原回调流程运行。
- 离线对比节点：`emplanner_compare_benchmark`
  直接注入车辆状态、参考线和障碍物，用于论文出图、参数对比和离线复现实验。

当前建议：

- 日常联调原系统，用 `emplanner_run`
- 做论文图、调 DP/QP 参数、改障碍物场景，用 `emplanner_compare_benchmark`

## 2. 目录

```text
emplanner/
├── include/                     # 头文件
├── launch/                      # benchmark 启动入口
├── scripts/
│   ├── plot_compare.py          # 对比图生成脚本
│   └── run_compare_benchmark.sh # 一键跑 benchmark + 出图
├── src/
│   ├── main.cpp                 # 在线节点入口
│   ├── emplanner.cpp            # 规划主逻辑
│   └── compare_benchmark.cpp    # 离线 benchmark
├── text/                        # 轨迹文件
├── benchmark_results/           # benchmark 输出目录
├── CMakeLists.txt
└── package.xml
```

## 3. 编译

在工作区根目录执行：

```bash
cd /home/wmd/elevator_car/P2P_fast_env_origin
catkin_make --pkg emplanner
source /opt/ros/noetic/setup.bash
source devel/setup.bash
```

## 4. 在线节点启动

在线节点入口在 [main.cpp](/home/wmd/elevator_car/P2P_fast_env_origin/src/planning/src/emplanner/src/main.cpp)。

启动方式：

```bash
cd /home/wmd/elevator_car/P2P_fast_env_origin
source /opt/ros/noetic/setup.bash
source devel/setup.bash
rosrun emplanner emplanner_run
```

在线模式默认订阅这些话题：

- `/car_pos`
- `/car_info`
- `/car_stop`
- `/obstacleList_lidar`
- `/object_detection_local`

在线模式默认发布这些规划相关输出：

- `em_Path`
- `line_Path`
- `local_Path`
- `local_qp_Path`

如果只是做论文图，不建议走这套在线回调，直接用下面的离线 benchmark。

## 5. 离线 Benchmark 启动

### 5.1 一键启动

最省事的方式：

```bash
/home/wmd/elevator_car/P2P_fast_env_origin/src/planning/src/emplanner/scripts/run_compare_benchmark.sh
```

默认行为：

- 使用 `compare_benchmark_paper.launch`
- 运行 `emplanner_compare_benchmark`
- 自动生成论文风格 `png + pdf`

改场景名：

```bash
/home/wmd/elevator_car/P2P_fast_env_origin/src/planning/src/emplanner/scripts/run_compare_benchmark.sh scenario_name:=my_case
```

只跑 benchmark，不导出论文图：

```bash
/home/wmd/elevator_car/P2P_fast_env_origin/src/planning/src/emplanner/scripts/run_compare_benchmark.sh --no-paper
```

### 5.2 roslaunch 启动

通用 launch：

```bash
cd /home/wmd/elevator_car/P2P_fast_env_origin
source /opt/ros/noetic/setup.bash
source devel/setup.bash
roslaunch emplanner compare_benchmark.launch
```

论文默认场景 launch：

```bash
roslaunch emplanner compare_benchmark_paper.launch
```

示例：改 QP 权重和障碍物位置

```bash
roslaunch emplanner compare_benchmark.launch \
  scenario_name:=my_case \
  use_rl_dp:=false \
  sample_s:=0.8 \
  col_node_num:=10 \
  w_qp_dl:=1200 \
  w_qp_ddl:=5000 \
  obstacle_centers_x:="3.0,4.8,6.3" \
  obstacle_centers_y:="0.0,2.0,-0.55" \
  obstacle_yaws:="0.06,-0.09,0.04"
```

### 5.3 直接运行可执行程序

如果你想最底层手动运行：

```bash
cd /home/wmd/elevator_car/P2P_fast_env_origin
source /opt/ros/noetic/setup.bash
source devel/setup.bash
roscore >/tmp/roscore_bench.log 2>&1 &
ROSCORE_PID=$!
sleep 2

./devel/lib/emplanner/emplanner_compare_benchmark _scenario_name:=manual_case

kill $ROSCORE_PID
wait $ROSCORE_PID 2>/dev/null
```

## 6. 结果输出

离线 benchmark 的结果目录是：

```text
emplanner/benchmark_results/<scenario_name>/
```

每次会生成这些文件：

- `dp_path.csv`
- `qp_path.csv`
- `reference_path.csv`
- `obstacles.csv`
- `summary.txt`
- `comparison.png`
- `comparison_paper.png`
- `comparison_paper.pdf`

## 7. 参考线设置

### 7.1 使用直线参考线

推荐论文对比时用直线：

- `use_straight_trajectory=true`
- `straight_start_x`
- `straight_start_y`
- `straight_length`
- `straight_step`

例如：

```bash
roslaunch emplanner compare_benchmark.launch \
  use_straight_trajectory:=true \
  straight_length:=10.0 \
  straight_step:=0.1
```

### 7.2 使用已有轨迹文件

如果不用直线，就给：

- `trajectory_file`

例如：

```bash
roslaunch emplanner compare_benchmark.launch \
  use_straight_trajectory:=false \
  trajectory_file:=/home/wmd/elevator_car/P2P_fast_env_origin/src/planning/src/emplanner/text/trajectory3f.txt
```

## 8. 规划参数设置

这些参数在 [emplanner.cpp](/home/wmd/elevator_car/P2P_fast_env_origin/src/planning/src/emplanner/src/emplanner.cpp) 中读取。

### 8.1 DP/采样相关

- `sample_s`
  每列节点的纵向步长，单位 m
- `sample_l`
  横向步长，单位 m
- `sample_s_num`
  DP 插值时每米采样点数
- `sample_s_per_meters`
  QP 和增密过程每米采样点数
- `col_node_num`
  纵向列数，决定规划长度
- `row_node_num`
  横向行数

规划长度大致由下面决定：

```text
planning_s_horizon ~= sample_s * col_node_num
```

例如：

- `sample_s=0.8`
- `col_node_num=10`

则规划长度约为：

```text
8.0 m
```

### 8.2 RL-DP 开关

- `use_rl_dp`

如果本地没有匹配模型，建议直接：

```bash
use_rl_dp:=false
```

这样前端就固定回退到传统 DP。

### 8.3 QP 平滑权重

- `w_qp_l`
- `w_qp_dl`
- `w_qp_ddl`
- `w_qp_ref_dp`

常用理解：

- `w_qp_l`
  约束横向偏移本身
- `w_qp_dl`
  抑制一阶导变化，路径更顺
- `w_qp_ddl`
  抑制二阶导变化，曲率更平滑
- `w_qp_ref_dp`
  让 QP 贴近 DP 前端路径

经验上：

- 想让 QP 看起来更平滑，优先增大 `w_qp_dl` 和 `w_qp_ddl`
- 如果 QP 太“死贴”DP，可以适当减小 `w_qp_ref_dp`

当前论文场景默认值：

```text
w_qp_l = 800
w_qp_dl = 1200
w_qp_ddl = 5000
w_qp_ref_dp = 50
```

## 9. 障碍物设置

障碍物读取逻辑在 [compare_benchmark.cpp](/home/wmd/elevator_car/P2P_fast_env_origin/src/planning/src/emplanner/src/compare_benchmark.cpp) 的 `LoadObstacleSpecs(...)` 中。

读取优先级：

1. `obstacles` 数组
2. `obstacle_centers_x / obstacle_centers_y` 多障碍物 CSV
3. `obstacle_center_x / obstacle_center_y` 单障碍物
4. 默认障碍物

### 9.1 多障碍物，推荐

使用这些参数：

- `obstacle_centers_x`
- `obstacle_centers_y`
- `obstacle_lengths`
- `obstacle_widths`
- `obstacle_yaws`
- `obstacle_x_vels`
- `obstacle_y_vels`

它们都是逗号分隔字符串。

示例：

```bash
roslaunch emplanner compare_benchmark.launch \
  scenario_name:=multi_obs_case \
  obstacle_centers_x:="3.0,4.8,6.3" \
  obstacle_centers_y:="0.0,2.0,-0.55" \
  obstacle_lengths:="0.8,0.8,0.8" \
  obstacle_widths:="0.8,0.7,0.7" \
  obstacle_yaws:="0.06,-0.09,0.04"
```

注意：

- `obstacle_centers_x` 和 `obstacle_centers_y` 数量必须一致
- `length/width/yaw/vel` 数量可以少于障碍物数，缺的会用默认值补

### 9.2 单障碍物

只测一个障碍物时：

- `obstacle_center_x`
- `obstacle_center_y`
- `default_obstacle_length`
- `default_obstacle_width`
- `obstacle_yaw`

示例：

```bash
roslaunch emplanner compare_benchmark.launch \
  scenario_name:=single_obs_case \
  obstacle_center_x:=3.0 \
  obstacle_center_y:=0.0 \
  default_obstacle_length:=0.8 \
  default_obstacle_width:=0.8 \
  obstacle_yaw:=0.05
```

### 9.3 `obstacles` 数组

最灵活，但写法更长。每个元素支持：

- `id`
- `center_x`
- `center_y`
- `length`
- `width`
- `yaw`
- `x_vel`
- `y_vel`

### 9.4 坐标含义

在直线参考线场景下，可以这样理解：

- `x` 越大，障碍物越靠前
- `y = 0` 表示压在参考线中心
- `y > 0` 表示在参考线左侧
- `y < 0` 表示在参考线右侧
- `yaw` 单位是弧度

## 10. 出图

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

## 11. 常用命令

### 11.1 默认论文场景，一键跑完

```bash
/home/wmd/elevator_car/P2P_fast_env_origin/src/planning/src/emplanner/scripts/run_compare_benchmark.sh
```

### 11.2 改场景名

```bash
/home/wmd/elevator_car/P2P_fast_env_origin/src/planning/src/emplanner/scripts/run_compare_benchmark.sh \
  scenario_name:=exp_case_01
```

### 11.3 改障碍物

```bash
/home/wmd/elevator_car/P2P_fast_env_origin/src/planning/src/emplanner/scripts/run_compare_benchmark.sh \
  scenario_name:=exp_case_02 \
  obstacle_centers_x:="2.8,4.5,6.0" \
  obstacle_centers_y:="0.0,1.5,-0.4" \
  obstacle_yaws:="0.03,-0.05,0.02"
```

### 11.4 改平滑权重

```bash
/home/wmd/elevator_car/P2P_fast_env_origin/src/planning/src/emplanner/scripts/run_compare_benchmark.sh \
  scenario_name:=exp_case_03 \
  w_qp_dl:=1500 \
  w_qp_ddl:=8000
```

## 12. 代码位置

核心文件：

- 在线节点入口：[src/main.cpp](/home/wmd/elevator_car/P2P_fast_env_origin/src/planning/src/emplanner/src/main.cpp)
- 规划主逻辑：[src/emplanner.cpp](/home/wmd/elevator_car/P2P_fast_env_origin/src/planning/src/emplanner/src/emplanner.cpp)
- 离线 benchmark：[src/compare_benchmark.cpp](/home/wmd/elevator_car/P2P_fast_env_origin/src/planning/src/emplanner/src/compare_benchmark.cpp)
- 通用 launch：[launch/compare_benchmark.launch](/home/wmd/elevator_car/P2P_fast_env_origin/src/planning/src/emplanner/launch/compare_benchmark.launch)
- 论文 launch：[launch/compare_benchmark_paper.launch](/home/wmd/elevator_car/P2P_fast_env_origin/src/planning/src/emplanner/launch/compare_benchmark_paper.launch)
- 一键脚本：[scripts/run_compare_benchmark.sh](/home/wmd/elevator_car/P2P_fast_env_origin/src/planning/src/emplanner/scripts/run_compare_benchmark.sh)
- 出图脚本：[scripts/plot_compare.py](/home/wmd/elevator_car/P2P_fast_env_origin/src/planning/src/emplanner/scripts/plot_compare.py)

## 13. 备注

- 如果 RL 模型不匹配，建议直接设置 `use_rl_dp=false`
- benchmark 输出的 `summary.txt` 会记录本次参数、障碍物数量和曲率统计
- 论文图默认横坐标已经对齐到全局参考线 `s`
