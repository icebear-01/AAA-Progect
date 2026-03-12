# ChatGPT Handoff for Paper Writing

这份文件用于交给网页版 ChatGPT 或其他大模型，帮助其快速理解当前项目，并协助撰写论文方法章节、实验章节和工程实现描述。

---

## 1. 一句话方法概述

本文方法是一个两阶段路径规划框架：

- 前端：**Transformer-guided A\*** 搜索
- 后端：**XY seed 生成 + 分段后端平滑优化**

注意：

- 前端不是 RL-DP
- 前端也不是 Hybrid A*
- 前端是 **Transformer 引导的栅格 A\*** 搜索
- 后端平滑复用了 `hybrid_aStar` 包中的 C++ 平滑器

可以概括为：

`Model Route -> Seed Route -> Smoothed / Final Route`

---

## 2. 当前工程里的真实链路

### 2.1 在线链路（RViz，手动点起点终点）

已经实现为 **全 C++ 运行时**：

- street 地图发布：C++
- ONNX 推理：C++
- guided A* 搜索：C++
- 后端平滑：C++
- RViz 中：
  - `2D Pose Estimate` 设置起点
  - `2D Nav Goal` 设置终点

在线入口：

- `planning/src/hybrid_aStar/launch/run_transformer_guided_astar_street_env.launch`

相关核心文件：

- 前端 ONNX 推理：
  - `planning/src/hybrid_aStar/src/guided_frontend_onnx.cpp`
  - `planning/src/hybrid_aStar/include/hybrid_a_star/guided_frontend_onnx.h`
- 主流程：
  - `planning/src/hybrid_aStar/src/hybrid_a_star_flow.cpp`
- street 地图发布器：
  - `planning/src/hybrid_aStar/app/frontend_street_map_env.cpp`
- RViz 配置：
  - `planning/src/hybrid_aStar/rviz/guided_astar_street_env.rviz`

### 2.2 离线链路（论文出图）

离线链路现在也是“规划全 C++，Python 只负责画图”：

- 读取 street 地图：C++
- ONNX guidance 推理：C++
- guided A* 搜索：C++
- XY seed 生成：C++
- 后端平滑：C++
- Python：只负责最终画图与导出 png/pdf

离线入口：

- `planning/src/hybrid_aStar/app/offline_guided_astar_cpp_demo.cpp`

后端平滑 CLI：

- `planning/src/hybrid_aStar/app/smooth_path_cli.cpp`

最终画图脚本：

- `planning/src/hybrid_aStar/scripts/offline_street_guided_astar_demo.py`

---

## 3. 三条路径各自代表什么

当前图里经常会看到三条路径：

### 3.1 Model Route

- 模型引导 A* 前端直接输出的原始路径
- 本质是栅格折线路径
- 往往存在明显的 45° / 90° 拐折

### 3.2 Seed Route

- 送进后端平滑器之前的中间路径
- 不是原始 Model Route
- 是在 C++ 中先做一层 XY QP seed 优化后得到的路径

### 3.3 Final Route / Smoothed Route

- 后端平滑器最终输出路径
- 如果后端判碰撞失败，则会触发回退

因此有时会出现：

- `Final Route == Seed Route`
或者
- `Final Route == Model Route`

这不代表后端没运行，而是触发了保护回退。

---

## 4. 当前 Seed Route 的真实生成方式

旧的 heuristic seed（`simplify + shortcut + Chaikin + resample`）已经不用了。

当前 seed 在这里生成：

- `planning/src/hybrid_aStar/app/smooth_path_cli.cpp`

它是一个 **XY QP seed 优化器**，当前特点：

- 以 raw route 为参考
- 只做 `x/y` 平面上的小范围优化
- 端点位置固定
- 端点 heading 不固定

当前关键参数：

- `x/y box = ±0.10 m`
- `w_smooth = 50000`
- `w_length = 20000`
- `w_ref = 0.1`

之后会再把 seed 做重采样。

当前 seed 重采样步长大约是：

- `0.10 m`

因此现在通常是：

- raw 路径点数较少
- seed 路径点数更密
- 更适合后端平滑器收敛

---

## 5. 当前后端平滑器

后端平滑器来自：

- `planning/src/hybrid_aStar/src/hybrid_a_star.cpp`

主要入口：

- `HybridAStar::SmoothPath(...)`

### 5.1 当前曲率约束

当前最大曲率约束为：

- `kappa_max = 0.5 1/m`

等价于最小转弯半径大约：

- `R = 2.0 m`

### 5.2 当前端点约束

目前后端平滑器对端点的处理是：

- 起点 `xy`：固定
- 终点 `xy`：固定
- 起点 heading：不硬约束
- 终点 heading：不硬约束

也就是说：

- 终点姿态不是强行对齐某个角度
- 最终终点 heading 主要由最后一段几何自然形成

### 5.3 当前碰撞检查

当前全局平滑采用的是简化碰撞模型，不是整车精细多边形碰撞。

当前设置：

- 原始地图分辨率：`0.25 m`
- 内部判碰分辨率：`0.125 m`
- 当前规则：**只查中心子格**
- 不再使用 `3x3` 邻域

另外：

- 起点和终点的选择比路径点更严格
- 起终点采用 `16` 点判碰

### 5.4 当前回退逻辑

当前保护逻辑为：

- 如果 `Seed Route` 碰撞，则回退到 `Model Route`
- 如果 `Smoothed Route` 碰撞，则回退到 `Seed Route`

所以如果最终图里三条线重合，不一定是平滑器没运行，而很可能是：

- seed 碰撞 -> 回退 raw
- smoothed 碰撞 -> 回退 seed

---

## 6. 为什么需要分段平滑

整条全局路径直接平滑，在以下场景很容易失败：

- 路径过长
- 窄通道多
- 局部拐弯多
- 原始前端路径是栅格折线而不是连续 kinodynamic seed

因此当前实现采用：

- **自动分段**
- **逐段平滑**
- **段间重叠与接缝融合**

---

## 7. 当前自动分段策略

实现位置：

- `planning/src/hybrid_aStar/src/hybrid_a_star.cpp`

当前启用的分段规则：

### 7.1 大航向变化点

- 相邻路径点局部航向变化超过 `30°`
- 触发一个分段点

### 7.2 相对上一个切点的累计航向变化

- 从上一个分段点开始累计转角
- 若累计转角超过 `45°`
- 再切一段

这条规则是为了抓“连续弯”，即使相邻点之间单步转角不大，但整体已经拐得很多。

### 7.3 窄通道入口 / 出口

- 根据 clearance 判断何时进入窄区域
- 以及何时离开窄区域

### 7.4 已关闭的规则

**路径拓扑切换点**已经关闭，因为原始路径平滑性较差时，这条规则会切得太碎。

### 7.5 段间连接

当前采用：

- 段间重叠
- seam blending

当前重叠点数大约：

- `4` 个点

这是为了尽量减少接缝处的折线感。

---

## 8. 当前方法的优点

1. 前端搜索速度更快
   - 使用 Transformer guidance 提供 cost map / bias
   - 比纯传统启发式搜索更快

2. 后端平滑可复用已有 C++ 工程
   - 不需要从零重写整套平滑框架

3. 在线链已经全 C++
   - 更适合部署

4. 离线实验自动化程度高
   - 随机地图
   - 自动生成 raw / seed / final 对比图
   - 自动生成曲率图
   - 自动保存 csv 与元信息

---

## 9. 当前局限性

1. 原始前端路径本质仍是栅格折线
   - 不如 Hybrid A* 原始轨迹那样天然连续

2. 后端可能回退
   - 某些贴边场景中，最终路径可能退回到 seed 或 raw

3. 分段接缝仍可能有轻微折感
   - 尤其在窄通道或复杂弯道处

4. 终点 heading 尚未显式优化
   - 当前终点姿态主要由末端几何自然形成

---

## 10. 当前论文建议怎么写

建议方法章节按下面结构写：

### 10.1 总体框架

- Transformer-guided A* 前端
- XY seed 生成
- 分段后端平滑优化

### 10.2 Transformer-guided A* 前端

- 输入：occupancy map、起点、终点
- 模型输出：guidance cost map
- 搜索：guided A* 使用模型 guidance 作为额外引导

### 10.3 Seed Route 生成

- 为什么原始栅格折线不能直接送进后端
- 为什么先做 XY seed
- 局部小范围偏移、长度项、参考项、平滑项

### 10.4 分段后端平滑

- 为什么整条平滑容易失败
- 当前分段规则：
  - 大航向变化点
  - 累计转角点
  - 窄通道入口/出口
- 段间重叠与 seam blending
- 曲率约束
- 占据图碰撞约束

### 10.5 工程实现

- 在线：全 C++ 前后端
- 离线：全 C++ 规划 + Python 只画图

### 10.6 实验

- Model / Seed / Final 路径对比
- 曲率对比
- 随机 street 地图实验
- 平滑失败与回退案例分析

---

## 11. 适合直接写进论文的一句话

本文提出一种“Transformer 引导搜索 + 分段后端优化”的两阶段路径规划框架：前端利用学习模型生成 guidance cost map 以加速栅格 A* 搜索，后端通过 XY seed 生成、自动分段和曲率约束平滑优化，对原始搜索路径进行几何质量提升与可行性修正。

---

## 12. 当前最关键文件

### 前端

- `planning/src/hybrid_aStar/src/guided_frontend_onnx.cpp`
- `planning/src/hybrid_aStar/include/hybrid_a_star/guided_frontend_onnx.h`
- `planning/src/hybrid_aStar/src/hybrid_a_star_flow.cpp`

### 后端平滑

- `planning/src/hybrid_aStar/src/hybrid_a_star.cpp`
- `planning/src/hybrid_aStar/app/smooth_path_cli.cpp`

### 在线 / RViz

- `planning/src/hybrid_aStar/launch/run_transformer_guided_astar_street_env.launch`
- `planning/src/hybrid_aStar/rviz/guided_astar_street_env.rviz`
- `planning/src/hybrid_aStar/app/frontend_street_map_env.cpp`

### 离线实验

- `planning/src/hybrid_aStar/app/offline_guided_astar_cpp_demo.cpp`
- `planning/src/hybrid_aStar/scripts/offline_street_guided_astar_demo.py`

---

## 13. 当前离线结果目录（示例）

当前常用离线结果目录：

- `planning/src/hybrid_aStar/offline_results/street_guided_demo/street_demo_random/`

常见输出包括：

- `offline_demo.png`
- `offline_demo_paper.png`
- `offline_demo_split_points.png`
- `curvature_compare.png`
- `frontend_raw_path.csv`
- `frontend_seed_path.csv`
- `smoothed_path.csv`
- `meta.json`

注意：

- 这个目录会被重复覆盖
- 它更适合作为“最近一次实验结果”
- 论文最终引用时建议固定某一次实验并另存

---

## 14. 可以继续让 ChatGPT 做什么

如果把这份文件交给网页版 ChatGPT，接下来可以让它继续做：

1. 写“方法章节”初稿
2. 写“工程实现章节”
3. 写“分段策略设计依据”
4. 写“实验设置与对比分析”
5. 写“失败案例与回退策略分析”
6. 帮忙整理成论文风格数学表述

建议直接问：

1. “根据这份 handoff，帮我写论文的方法章节初稿。”
2. “帮我把 Transformer-guided A* 前端和分段后端平滑写成论文风格。”
3. “帮我写实验章节，包括 raw/seed/final path 对比和曲率对比。”
4. “帮我把自动分段规则写成数学公式和算法描述。”

