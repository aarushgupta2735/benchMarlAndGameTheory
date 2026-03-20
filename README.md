# Balance Scenario: Original vs Contact-Reward — MAPPO Comparison

## Overview

This project compares two variants of the VMAS **Balance** scenario trained with
**MAPPO** (Multi-Agent PPO):

| Variant | Description |
|---|---|
| **Original** | Agents carry a package on a rod to a goal. Reward = position shaping + fall penalty. |
| **Active (Contact)** | Same task, plus a small bonus when agents maintain physical contact with the rod. |

### Why a Contact Reward?

In the original scenario agents can sometimes learn lazy policies — drifting
away from the rod while a subset does the work.  The contact reward provides a
gentle *intrinsic* incentive to stay engaged:

```
contact_rew = Σ (contact_reward_coeff × 𝟙[dist(agent, rod) ≤ threshold])
```

**Calibration**: With the default `contact_reward_coeff = 0.5` and 3 agents, the
maximum contact bonus is **1.5 per step**, which is ~1.5 % of the position-shaping
reward scale (±100).  This is enough to break ties in favour of staying near the
rod, but never dominates the task objective.

---

## Project Structure

```
Code/
├── scenarios/
│   ├── balance_original.py       # Verbatim VMAS balance scenario
│   └── balance_active.py         # + rod-contact activity reward
├── training/
│   └── train_mappo.py            # TorchRL MAPPO training script
├── configs/
│   └── experiment_config.yaml    # Shared hyperparameters
├── compare_results.py            # Plot comparison curves
├── results/                      # Created at training time
│   ├── original/seed_0/...
│   └── active/seed_0/...
├── plots/                        # Created by compare_results.py
└── README.md                     # This file
```

---

## Setup

```bash
# Activate your existing environment
conda activate benchmarl

# Verify key packages
python -c "import vmas; import torchrl; print('OK')"

# (Optional) install plotting deps if missing
pip install matplotlib pandas
```

---

## Quick Start

### 1. Train a single scenario

```bash
cd Code

# Original balance
python train_mappo.py --scenario original --seed 0

# Contact-reward balance
python train_mappo.py --scenario active --seed 0
```

### 2. Run full comparison (3 seeds × 2 scenarios)

```bash
python train_mappo.py --run-all
```

### 3. Generate plots

```bash
python compare_results.py
# Outputs to ./plots/
```

---

## Key CLI Arguments

| Argument | Default | Description |
|---|---|---|
| `--scenario` | — | `original` or `active` |
| `--seed` | 0 | Random seed |
| `--run-all` | — | Run both scenarios with `--seeds` |
| `--seeds` | 0 1 2 | Seeds for `--run-all` |
| `--n-agents` | 3 | Number of agents |
| `--total-frames` | 300000 | Total training frames |
| `--num-envs` | 32 | Parallel envs |
| `--device` | cpu | `cpu` or `cuda` |
| `--contact-reward-coeff` | 0.5 | Contact reward weight (active only) |

---

## Reward Design Details

### Original

```
reward = ground_rew + pos_rew
```

- `pos_rew`: potential-based shaping (shaping_factor=100 × Δdist to goal)
- `ground_rew`: −10 when rod or package touches the floor

### Active (Contact)

```
reward = ground_rew + pos_rew + contact_rew
```

- `contact_rew`: for each agent within `contact_threshold` (default: 3.5× agent
  radius) of the rod, add `+contact_reward_coeff` (default: 0.5)
- Shared across all agents (same as pos_rew / ground_rew)

### Calibration Rationale

| Component | Typical magnitude per step |
|---|---|
| `pos_rew` | −100 to +100 |
| `ground_rew` | 0 or −10 |
| `contact_rew` | 0 to 1.5 (3 agents × 0.5) |

The contact reward is intentionally kept at **~1 %** of the task reward scale.

---

## Expected Outcomes

1. **Faster convergence**: the contact reward provides denser signal early on.
2. **Higher activity**: agents spend more time near the rod instead of idling.
3. **Similar asymptotic performance**: the contact reward doesn't distort the
   optimal policy since carrying the package to the goal still dominates.

---

## References

- [VMAS](https://github.com/proroklab/VectorizedMultiAgentSimulator)
- [BenchMARL](https://github.com/facebookresearch/BenchMARL)
- [TorchRL Multi-Agent PPO Tutorial](https://pytorch.org/rl/stable/tutorials/multiagent_ppo.html)
