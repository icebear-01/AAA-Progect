# RL-DP README

## 1. Overview

RL-DP is a hybrid local path planning method that combines the structured prior of traditional dynamic programming (DP) with the policy optimization ability of reinforcement learning (RL). The method works in a discrete Frenet `s-l` planning space. Instead of exhaustively searching the optimal node transition as in classical DP, RL-DP learns a policy that selects the lateral position on each longitudinal sampling column.

The key idea is not to replace DP completely, but to preserve the discrete planning structure and safety constraints from classical planning while using PPO to learn a more flexible sequential decision policy.

In short, RL-DP can be understood as:

- discrete `s-l` planning formulation
- PPO-based sequential lateral decision making
- explicit safety constraints through action masking and collision checking
- DP-screened scenario generation for stable training

---

## 2. Problem Formulation

The planning problem is defined in the Frenet coordinate system:

- `s`: longitudinal distance along the reference line
- `l`: lateral offset relative to the reference line

The local path is represented as a sequence of sampled points:

\[
\mathcal{P} = \{(s_0, l_0), (s_1, l_1), \dots, (s_T, l_T)\}
\]

where each `s_t` is fixed by the grid, and the planner only needs to choose `l_t`.

In the current implementation, the default grid is:

- `s ∈ [0, 8] m`
- `s_samples = 9`
- longitudinal step about `1.0 m`
- lateral spacing `0.35 m`
- `l ≈ [-3.85, 3.85] m`
- `l_samples = 23`

Therefore, the path planning task is transformed into selecting one lateral node on each longitudinal column.

---

## 3. Overall Framework

The RL-DP framework contains four main stages:

1. Build a discrete `s-l` sampling grid.
2. Randomly generate obstacle scenarios and use traditional DP to filter feasible and high-quality scenarios.
3. Train a PPO policy in the filtered planning environment.
4. During inference, use the learned policy to select lateral nodes column by column and generate the final path.

This gives a hybrid planning paradigm:

**discrete planning structure + DP prior + safe RL exploration + policy optimization**

---

## 4. MDP Formulation

RL-DP models local path planning as a Markov Decision Process (MDP).

### 4.1 State

The state is composed of:

1. **Occupancy grid**
   - a boolean grid in `s-l` space
   - indicates which sampled cells are occupied by obstacles

2. **Obstacle geometry**
   - each obstacle is represented by 4 corners
   - up to a fixed number of obstacles are kept
   - obstacle corners are normalized and flattened
   - nearest obstacles are prioritized when there are too many

3. **Action mask**
   - explicitly indicates which actions are currently valid
   - invalid actions are masked out before sampling or inference

4. **Scalar progress features**
   - normalized longitudinal progress
   - current lateral position
   - normalized start lateral position

This hybrid state representation combines spatial occupancy, obstacle geometry, feasibility prior, and planning progress.

### 4.2 Action

The action is discrete:

\[
a_t \in \{0, 1, \dots, l\_samples-1\}
\]

That is, at each step the policy selects the lateral index of the next longitudinal column.

In the default setup, the action dimension is `23`.

### 4.3 Transition

After taking action `a_t`, the environment:

- maps the selected action to a lateral coordinate on the current `s` column
- connects the previous point and the new point
- performs interpolated collision checking
- computes reward terms such as smoothness and reference deviation
- advances to the next longitudinal column

### 4.4 Terminal Conditions

An episode terminates when:

- the path reaches the last longitudinal column
- a collision occurs
- no valid action remains

---

## 5. Safety Mechanisms

One major characteristic of RL-DP is that safety is not only encouraged by reward shaping. It is also enforced by explicit feasibility constraints.

### 5.1 Action Mask

At every step, the environment computes a boolean action mask to remove obviously invalid actions.

An action is masked out if:

- it violates the lateral move limit
- the interpolated path segment collides with an obstacle

This significantly shrinks the exploration space and keeps policy optimization close to the feasible set.

### 5.2 Interpolated Collision Checking

Instead of checking only sampled nodes, RL-DP checks the whole transition segment between two adjacent path points.

The environment:

- checks the terminal point
- inserts interpolation points between adjacent nodes
- verifies collision on all interpolation points

This reduces collision leakage caused by coarse grid transitions.

### 5.3 Vehicle Footprint Constraint

If vehicle length and width are provided, collision checking is performed using oriented bounding box (OBB) intersection instead of point collision.

The implementation also supports collision inflation:

- coarse inflation for occupancy rasterization
- fine inflation for accurate collision checking

This gives additional safety margin and makes the planner closer to real vehicle planning.

### 5.4 No-valid-action Termination

If all actions are masked out, the episode terminates immediately with penalty. This avoids fake-feasible states where the policy still outputs an action although the problem is no longer recoverable.

---

## 6. Reward Function

The reward function is designed to balance:

- safety
- reachability
- path smoothness
- reference tracking

It can be written conceptually as:

\[
r_t =
r_{\text{terminal}}
 r_{\text{collision}}
 r_{\text{invalid}}
- \lambda_{\text{ref}} |l_t - l_{\text{ref}}|
- \lambda_{\text{smooth}} |l_t - l_{t-1}|
- \lambda_{\text{move}} \max(0, |a_t-a_{t-1}|-\Delta_{\max})
- \lambda_{\text{slope}} \max\left(0,\left|\frac{l_t-l_{t-1}}{s_t-s_{t-1}}\right|-\kappa_{\max}\right)
\]

### Reward components in the implementation

- **terminal reward**
  - positive reward when the path reaches the goal

- **collision penalty**
  - large negative reward when collision occurs

- **reference cost**
  - penalizes lateral deviation from the reference line

- **smoothness cost**
  - penalizes absolute lateral change between adjacent columns

- **move limit cost**
  - penalizes excessive lateral index jumps

- **slope cost**
  - penalizes overly steep local transitions

- **no valid action penalty**
  - penalizes unrecoverable dead-end states

### Design objective

The reward design follows three priorities:

1. avoid collision and finish the planning horizon
2. stay near the reference line
3. keep the path smooth and dynamically reasonable

---

## 7. Network Architecture

The policy model is a shared Actor-Critic network.

### 7.1 Convolutional occupancy encoder

The occupancy grid is processed by a CNN:

- `Conv2d(grid_channels -> 32, kernel=3, padding=1)`
- `ReLU`
- `Conv2d(32 -> 64, kernel=3, padding=1)`
- `ReLU`
- `AdaptiveAvgPool2d(1,1)`

This branch extracts spatial obstacle distribution features.

### 7.2 Feature fusion

The CNN output is concatenated with:

- obstacle corner features
- action mask
- normalized scalar progress features

### 7.3 Shared MLP trunk

The fused feature vector is passed through:

- `Linear -> ReLU`
- `Linear -> ReLU`

The default hidden dimension is `128`.

### 7.4 Dual heads

The network then branches into:

- **policy head**
  - outputs discrete action logits

- **value head**
  - estimates the state value

This architecture combines spatial representation learning and low-dimensional planning priors in a compact Actor-Critic model.

---

## 8. PPO Training

RL-DP uses PPO for policy optimization.

### 8.1 Rollout collection

Multiple environments are run in parallel. For each rollout step, the trainer stores:

- encoded states
- actions
- old log probabilities
- rewards
- done flags
- value estimates
- masked logits
- action masks

### 8.2 Advantage estimation

Advantages are computed by GAE:

\[
\hat{A}_t
\]

and returns are computed as:

\[
\hat{R}_t = \hat{A}_t + V(s_t)
\]

Advantages are normalized before PPO updates.

### 8.3 PPO objective

The policy is updated with clipped PPO objective:

\[
L^{clip} = \mathbb{E}\left[\min(r_t \hat{A}_t, \text{clip}(r_t,1-\epsilon,1+\epsilon)\hat{A}_t)\right]
\]

The total loss includes:

- policy loss
- value loss
- entropy regularization
- optional KL regularization

### 8.4 Action-mask consistency

A notable design choice is that action masking is applied both:

- during rollout sampling
- during PPO update

This keeps training and execution distributions consistent.

---

## 9. Role of Traditional DP

Traditional DP is not discarded in RL-DP. Instead, it provides planning prior in the data generation stage.

### 9.1 Scenario screening

For each random obstacle scene, classical DP is used to evaluate:

- whether a feasible path exists
- the total path cost
- the average transition cost

Only feasible and sufficiently good scenes are retained.

### 9.2 Why DP screening is useful

This mechanism has several benefits:

- removes meaningless fully blocked scenes
- focuses training on useful planning cases
- improves training stability
- provides a structured prior from classical planning

### 9.3 Offline scenario dataset

The implementation supports offline scenario datasets:

- first generate and screen scenarios with traditional DP
- then train PPO directly from the scenario dataset

This avoids expensive online scene screening during every environment reset and makes long training more efficient.

---

## 10. Complex Scenario Augmentation

To improve robustness in dense obstacle environments, RL-DP supports additional complex-scene generation.

These complex scenes are generated with the following principles:

- higher obstacle count, typically `5` or more
- obstacle layouts are denser
- coarse overlap rejection is applied for efficiency
- scenes are still filtered by traditional DP feasibility

This allows the training distribution to gradually shift from easy scenarios to more challenging planning situations.

---

## 11. Inference

During inference, the trained PPO policy is used to generate a local path sequentially:

1. initialize environment state
2. encode occupancy, obstacles, action mask, and scalar features
3. compute action logits
4. mask invalid actions
5. select the action with the highest score, or sample if needed
6. advance to the next column
7. repeat until the goal is reached or termination occurs

The output is a discrete `s-l` path, which can be visualized directly or used by downstream smoothing/optimization modules.

---

## 12. Main Contributions of RL-DP

The main characteristics worth emphasizing are:

1. **Structured planning formulation**
   - the method works directly on a discrete `s-l` planning grid
   - preserves interpretability and planning structure

2. **Explicit safe exploration**
   - invalid actions are masked before policy execution
   - safety is enforced not only by reward but also by hard feasibility constraints

3. **DP prior for data generation**
   - classical DP is used to screen high-quality feasible scenarios
   - improves data quality and training stability

4. **Hybrid state representation**
   - occupancy grid + obstacle geometry + action mask + progress features

5. **Complex-scene enhancement**
   - supports dense obstacle scenario generation and incremental dataset expansion

---

## 13. One-sentence Summary

RL-DP is a safe discrete local path planning framework that uses PPO to learn column-wise lateral decisions in `s-l` space, while relying on traditional DP to provide feasible structured planning prior and high-quality training scenarios.
