# RL Fine-Tuning Scaffold

This folder contains a scaffold for optional Route A RL fine-tuning.

Current status:
- `env.py`: environment wrapper running `infer_cost_map -> guided Hybrid A* -> reward`
- `train_ppo.py`: entrypoint placeholder with TODO

Reward template:
`reward = -path_length - 0.1*expanded_nodes - 10*collision - 5*failure`

This scaffold intentionally does **not** implement full PPO in this repository update.
