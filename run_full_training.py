"""
Full-scale MAPPO training: Original vs Contact-Reward Balance Scenario.

Optimized for local CPU training (i7-1355U, 8GB RAM).
Runs both scenarios across 3 seeds, generates comparison plots.

Usage:
    conda activate benchMarl
    python run_full_training.py
"""

import argparse
import csv
import importlib.util
import math
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from torch import nn

# TorchRL imports
from torchrl.collectors import SyncDataCollector
from torchrl.data import TensorDictReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.envs import RewardSum, TransformedEnv
from torchrl.envs.libs.vmas import VmasEnv
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.modules import MultiAgentMLP, ProbabilisticActor, TanhNormal
from torchrl.objectives import ClipPPOLoss, ValueEstimators
from tensordict.nn import TensorDictModule, TensorDictSequential

# ── Paths ──
SCRIPT_DIR = Path(__file__).resolve().parent
SCENARIOS_DIR = SCRIPT_DIR / "scenarios"
RESULTS_DIR = SCRIPT_DIR / "results"
PLOTS_DIR = SCRIPT_DIR / "plots"

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION — Tuned for best quality on CPU with 8GB RAM
# ══════════════════════════════════════════════════════════════════════════════
CONFIG = dict(
    # Environment
    n_agents=3,
    max_steps=200,
    num_envs=32,            # 32 parallel envs (CPU-friendly)

    # Training scale — substantial for meaningful comparison
    total_frames=600_000,   # 600K frames per scenario/seed
    frames_per_batch=6_000, # Collect 6K frames per PPO update (100 batches total)
    minibatch_size=400,     # 15 minibatches per epoch
    n_minibatch_iters=8,    # 8 PPO epochs per batch (more updates = better sample efficiency)

    # Optimiser
    lr=3e-4,                # Slightly lower LR for stability
    gamma=0.99,
    gae_lambda=0.95,
    clip_epsilon=0.2,
    entropy_coeff=0.01,
    critic_coeff=1.0,
    clip_grad_norm=True,
    max_grad_norm=5.0,

    # Network
    hidden_dim=256,
    share_params=True,
    centralised_critic=True,

    # Device
    device="cpu",

    # Active scenario
    contact_reward_coeff=0.5,

    # Seeds
    seeds=[0, 1, 2],
)


# ══════════════════════════════════════════════════════════════════════════════
#  Scenario loading
# ══════════════════════════════════════════════════════════════════════════════

def _load_scenario_class(scenario_name: str):
    fmap = {
        "original": SCENARIOS_DIR / "balance_original.py",
        "active":   SCENARIOS_DIR / "balance_active.py",
    }
    if scenario_name not in fmap:
        raise ValueError(f"Unknown scenario '{scenario_name}'. Choose from {list(fmap)}")
    path = fmap[scenario_name]
    spec = importlib.util.spec_from_file_location(f"balance_{scenario_name}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.Scenario


def make_env(scenario_name, cfg, seed, **scenario_kwargs):
    ScenarioCls = _load_scenario_class(scenario_name)
    env = VmasEnv(
        scenario=ScenarioCls(),
        num_envs=cfg["num_envs"],
        device=cfg["device"],
        seed=seed,
        continuous_actions=True,
        max_steps=cfg["max_steps"],
        n_agents=cfg["n_agents"],
        **scenario_kwargs,
    )
    env = TransformedEnv(
        env,
        RewardSum(in_keys=[env.reward_key], out_keys=[("agents", "episode_reward")])
    )
    return env


# ══════════════════════════════════════════════════════════════════════════════
#  Policy & Critic
# ══════════════════════════════════════════════════════════════════════════════

def build_policy_and_critic(env, cfg):
    n_agents = len(env.agents)
    obs_spec = env.full_observation_spec["agents", "observation"]
    act_spec = env.full_action_spec["agents", "action"]
    obs_size = obs_spec.shape[-1]
    act_size = act_spec.shape[-1]

    # Policy: shared-parameter MLP → TanhNormal distribution
    policy_net = MultiAgentMLP(
        n_agent_inputs=obs_size,
        n_agent_outputs=2 * act_size,
        n_agents=n_agents,
        centralised=False,
        share_params=cfg["share_params"],
        depth=2,
        num_cells=cfg["hidden_dim"],
        activation_class=nn.Tanh,
    )
    policy_module = TensorDictModule(
        policy_net,
        in_keys=[("agents", "observation")],
        out_keys=[("agents", "loc_scale")],
    )
    unbind = TensorDictModule(
        lambda x: (
            x[..., :act_size],
            torch.nn.functional.softplus(x[..., act_size:]) + 1e-4,
        ),
        in_keys=[("agents", "loc_scale")],
        out_keys=[("agents", "loc"), ("agents", "scale")],
    )
    policy = ProbabilisticActor(
        module=TensorDictSequential(policy_module, unbind),
        in_keys=[("agents", "loc"), ("agents", "scale")],
        out_keys=[("agents", "action")],
        distribution_class=TanhNormal,
        distribution_kwargs={
            "low": act_spec.space.low,
            "high": act_spec.space.high,
        },
        return_log_prob=True,
        log_prob_key=("agents", "sample_log_prob"),
    )

    # Critic: centralised (sees all obs), shared params
    critic_net = MultiAgentMLP(
        n_agent_inputs=obs_size,
        n_agent_outputs=1,
        n_agents=n_agents,
        centralised=cfg["centralised_critic"],
        share_params=cfg["share_params"],
        depth=2,
        num_cells=cfg["hidden_dim"],
        activation_class=nn.Tanh,
    )
    critic = TensorDictModule(
        critic_net,
        in_keys=[("agents", "observation")],
        out_keys=[("agents", "state_value")],
    )
    return policy, critic


# ══════════════════════════════════════════════════════════════════════════════
#  CSV Logger
# ══════════════════════════════════════════════════════════════════════════════

class CSVLogger:
    def __init__(self, log_dir: str):
        os.makedirs(log_dir, exist_ok=True)
        self.path = os.path.join(log_dir, "progress.csv")
        self._file = None
        self._writer = None
        self._keys = None

    def log(self, data: dict):
        if self._writer is None:
            self._keys = list(data.keys())
            self._file = open(self.path, "w", newline="")
            self._writer = csv.DictWriter(self._file, fieldnames=self._keys)
            self._writer.writeheader()
        self._writer.writerow({k: data.get(k, "") for k in self._keys})
        self._file.flush()

    def close(self):
        if self._file:
            self._file.close()


# ══════════════════════════════════════════════════════════════════════════════
#  Training Loop
# ══════════════════════════════════════════════════════════════════════════════

def train(scenario_name: str, seed: int, cfg: dict):
    device = cfg["device"]
    log_dir = str(RESULTS_DIR / scenario_name / f"seed_{seed}")
    os.makedirs(log_dir, exist_ok=True)
    logger = CSVLogger(log_dir)

    torch.manual_seed(seed)

    # Extra kwargs for the active scenario
    scenario_kwargs = {}
    if scenario_name == "active":
        scenario_kwargs["contact_reward_coeff"] = cfg.get("contact_reward_coeff", 0.5)

    # ── Environment ──
    env_fn = lambda: make_env(scenario_name, cfg, seed, **scenario_kwargs)
    env = env_fn()

    # ── Policy & Critic ──
    policy, critic = build_policy_and_critic(env, cfg)
    policy = policy.to(device)
    critic = critic.to(device)

    # ── PPO Loss ──
    loss_module = ClipPPOLoss(
        actor_network=policy,
        critic_network=critic,
        clip_epsilon=cfg["clip_epsilon"],
        entropy_coeff=cfg["entropy_coeff"],
        critic_coeff=cfg["critic_coeff"],
        normalize_advantage=True,
        normalize_advantage_exclude_dims=(-2,),  # exclude agent dimension
    )
    loss_module.set_keys(
        reward=env.reward_key,
        action=("agents", "action"),
        done=("agents", "done"),
        terminated=("agents", "terminated"),
        sample_log_prob=("agents", "sample_log_prob"),
        value=("agents", "state_value"),
    )
    loss_module.make_value_estimator(
        ValueEstimators.GAE,
        gamma=cfg["gamma"],
        lmbda=cfg["gae_lambda"],
    )

    optim = torch.optim.Adam(loss_module.parameters(), lr=cfg["lr"])

    # ── Collector ──
    collector = SyncDataCollector(
        create_env_fn=env_fn,
        policy=policy,
        frames_per_batch=cfg["frames_per_batch"],
        total_frames=cfg["total_frames"],
        device=device,
        storing_device=device,
    )

    # ── Replay buffer ──
    replay_buffer = TensorDictReplayBuffer(
        storage=LazyTensorStorage(cfg["frames_per_batch"], device=device),
        sampler=SamplerWithoutReplacement(),
        batch_size=cfg["minibatch_size"],
    )

    # ── Training ──
    total_frames_so_far = 0
    n_iters = 0
    t_start = time.time()
    best_reward = float("-inf")
    total_batches = cfg["total_frames"] // cfg["frames_per_batch"]

    print(f"\n{'='*70}")
    print(f"  MAPPO Training: {scenario_name.upper()} | seed={seed} | device={device}")
    print(f"  Frames: {cfg['total_frames']:,} | Batch: {cfg['frames_per_batch']:,} "
          f"| PPO epochs: {cfg['n_minibatch_iters']} | LR: {cfg['lr']}")
    print(f"  Estimated batches: {total_batches}")
    print(f"{'='*70}")

    n_agents_actual = len(env.agents)

    for batch in collector:
        n_iters += 1
        total_frames_so_far += batch.numel()

        # ── Expand root done/terminated to per-agent dimension ──
        # VMAS done is [batch, time, 1]; reward/value are [batch, time, n_agents, 1]
        # GAE needs all shapes to match, so expand done to agent dim.
        for key in ["done", "terminated"]:
            root_val = batch.get(("next", key), None)
            if root_val is not None:
                expanded = root_val.unsqueeze(-2).expand(
                    *root_val.shape[:-1], n_agents_actual, 1
                )
                batch.set(("next", "agents", key), expanded)
            # Also set at the current step level if present
            root_cur = batch.get(key, None)
            if root_cur is not None:
                expanded_cur = root_cur.unsqueeze(-2).expand(
                    *root_cur.shape[:-1], n_agents_actual, 1
                )
                batch.set(("agents", key), expanded_cur)

        # ── Compute GAE ──
        with torch.no_grad():
            loss_module.value_estimator(
                batch,
                params=loss_module.critic_network_params,
                target_params=loss_module.target_critic_network_params,
            )

        # ── Minibatch PPO updates ──
        replay_buffer.extend(batch.reshape(-1))
        train_losses = []
        obj_losses = []
        critic_losses = []
        entropy_losses = []

        for _ in range(cfg["n_minibatch_iters"]):
            for mb in replay_buffer:
                loss_vals = loss_module(mb)
                loss = (
                    loss_vals["loss_objective"]
                    + loss_vals["loss_critic"]
                    + loss_vals["loss_entropy"]
                )
                # NaN guard: skip update if loss is NaN
                if torch.isnan(loss) or torch.isinf(loss):
                    optim.zero_grad()
                    continue
                loss.backward()
                if cfg["clip_grad_norm"]:
                    nn.utils.clip_grad_norm_(loss_module.parameters(), cfg["max_grad_norm"])
                optim.step()
                optim.zero_grad()
                train_losses.append(loss.item())
                obj_losses.append(loss_vals["loss_objective"].item())
                critic_losses.append(loss_vals["loss_critic"].item())
                entropy_losses.append(loss_vals["loss_entropy"].item())

        mean_loss = np.mean(train_losses) if train_losses else float("nan")
        mean_obj = np.mean(obj_losses) if obj_losses else float("nan")
        mean_critic = np.mean(critic_losses) if critic_losses else float("nan")
        mean_entropy = np.mean(entropy_losses) if entropy_losses else float("nan")

        # ── Episode reward ──
        # Episode endings are in ("next", "done"); episode_reward is in ("next", "agents", "episode_reward")
        mean_reward = float("nan")
        try:
            next_td = batch.get("next")
            if next_td is not None:
                done_vals = next_td.get("done", None)
                ep_rew_vals = next_td.get(("agents", "episode_reward"), None)
                if done_vals is not None and ep_rew_vals is not None:
                    done_mask = done_vals.squeeze(-1)  # [num_envs, time]
                    if done_mask.any():
                        # ep_rew: [num_envs, time, n_agents, 1] or [num_envs, time, n_agents]
                        while ep_rew_vals.dim() > done_mask.dim() + 1:
                            ep_rew_vals = ep_rew_vals.squeeze(-1)
                        # Average over agents: [num_envs, time]
                        ep_rew_agent_mean = ep_rew_vals.mean(dim=-1)
                        mean_reward = ep_rew_agent_mean[done_mask].mean().item()
        except Exception:
            pass
        if not np.isnan(mean_reward) and mean_reward > best_reward:
            best_reward = mean_reward

        elapsed = time.time() - t_start
        fps = total_frames_so_far / max(elapsed, 1e-6)
        progress = total_frames_so_far / cfg["total_frames"] * 100
        eta_s = (cfg["total_frames"] - total_frames_so_far) / max(fps, 1)

        log_data = {
            "step": total_frames_so_far,
            "iteration": n_iters,
            "mean_reward": mean_reward,
            "best_reward": best_reward,
            "mean_loss": mean_loss,
            "loss_objective": mean_obj,
            "loss_critic": mean_critic,
            "loss_entropy": mean_entropy,
            "fps": fps,
            "elapsed_s": elapsed,
        }
        logger.log(log_data)

        # Print every 3 iterations or at start/end
        if n_iters % 3 == 1 or n_iters <= 2 or total_frames_so_far >= cfg["total_frames"]:
            eta_min = eta_s / 60
            print(
                f"  [{scenario_name:>8}] {progress:5.1f}%  iter={n_iters:4d}  "
                f"frames={total_frames_so_far:>9,}  "
                f"rew={mean_reward:>8.2f}  best={best_reward:>8.2f}  "
                f"loss={mean_loss:.4f}  fps={fps:.0f}  "
                f"ETA={eta_min:.1f}min"
            )

    collector.shutdown()
    logger.close()
    env.close()

    elapsed_total = time.time() - t_start
    print(f"\n  DONE [{scenario_name}|seed={seed}] in {elapsed_total/60:.1f} min  "
          f"| best_reward={best_reward:.2f} | log={log_dir}")
    return log_dir


# ══════════════════════════════════════════════════════════════════════════════
#  Plotting
# ══════════════════════════════════════════════════════════════════════════════

def generate_comparison_plots(cfg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    PLOTS_DIR.mkdir(exist_ok=True)
    seeds = cfg["seeds"]

    def load_scenario_data(scenario_name):
        seed_data = {}
        for seed in seeds:
            csv_path = RESULTS_DIR / scenario_name / f"seed_{seed}" / "progress.csv"
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                seed_data[seed] = df
        return seed_data

    def compute_mean_std_reward(seed_data, column="mean_reward"):
        if not seed_data:
            return None, None, None
        min_len = min(len(df) for df in seed_data.values())
        all_steps = list(seed_data.values())[0]["step"].values[:min_len]
        all_rewards = np.array([df[column].values[:min_len] for df in seed_data.values()])
        # Smooth with rolling window
        window = max(1, min_len // 20)
        smoothed = np.array([
            pd.Series(row).rolling(window, min_periods=1).mean().values
            for row in all_rewards
        ])
        mean = np.nanmean(smoothed, axis=0)
        std = np.nanstd(smoothed, axis=0)
        return all_steps, mean, std

    configs_meta = {
        "original": {"color": "#2196F3", "label": "Original Balance"},
        "active":   {"color": "#FF5722", "label": "Balance + Contact Reward"},
    }

    # ── Plot 1: Reward curves ──
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left: reward comparison
    ax = axes[0]
    for name, meta in configs_meta.items():
        seed_data = load_scenario_data(name)
        if not seed_data:
            continue
        steps, mean, std = compute_mean_std_reward(seed_data)
        ax.plot(steps, mean, color=meta["color"], label=meta["label"], linewidth=2)
        ax.fill_between(steps, mean - std, mean + std, color=meta["color"], alpha=0.2)

    ax.set_xlabel("Training Frames", fontsize=13)
    ax.set_ylabel("Mean Episode Reward", fontsize=13)
    ax.set_title("Reward: Original vs Contact-Reward", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Right: loss comparison
    ax = axes[1]
    for name, meta in configs_meta.items():
        seed_data = load_scenario_data(name)
        if not seed_data:
            continue
        steps, mean, std = compute_mean_std_reward(seed_data, column="mean_loss")
        ax.plot(steps, mean, color=meta["color"], label=meta["label"], linewidth=2)
        ax.fill_between(steps, mean - std, mean + std, color=meta["color"], alpha=0.15)

    ax.set_xlabel("Training Frames", fontsize=13)
    ax.set_ylabel("Mean Loss", fontsize=13)
    ax.set_title("Loss: Original vs Contact-Reward", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.suptitle(f"MAPPO on Balance: {len(seeds)} seeds, {cfg['n_agents']} agents, "
                 f"{cfg['total_frames']:,} frames",
                 fontsize=15, y=1.02)
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "comparison_reward_loss.png", dpi=150, bbox_inches="tight")
    print(f"  Saved: {PLOTS_DIR / 'comparison_reward_loss.png'}")

    # ── Plot 2: Per-seed reward curves ──
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    for idx, (name, meta) in enumerate(configs_meta.items()):
        ax = axes[idx]
        seed_data = load_scenario_data(name)
        for seed, df in seed_data.items():
            window = max(1, len(df) // 20)
            smoothed = df["mean_reward"].rolling(window, min_periods=1).mean()
            ax.plot(df["step"], smoothed, label=f"Seed {seed}", linewidth=1.5, alpha=0.8)
        ax.set_xlabel("Training Frames", fontsize=12)
        ax.set_ylabel("Mean Episode Reward" if idx == 0 else "", fontsize=12)
        ax.set_title(meta["label"], fontsize=13)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("Per-Seed Reward Curves", fontsize=14, y=1.01)
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "per_seed_rewards.png", dpi=150, bbox_inches="tight")
    print(f"  Saved: {PLOTS_DIR / 'per_seed_rewards.png'}")

    # ── Summary table ──
    print(f"\n{'='*70}")
    print(f"  FINAL PERFORMANCE (last 20% of training, mean +/- std over seeds)")
    print(f"{'='*70}")
    print(f"  {'Scenario':<30} {'Mean Reward':<20} {'Best Reward':<15}")
    print(f"  {'-'*65}")

    for name in ["original", "active"]:
        seed_data = load_scenario_data(name)
        if not seed_data:
            print(f"  {name:<30} {'N/A':<20}")
            continue

        final_rewards = []
        final_best = []
        for seed, df in seed_data.items():
            cutoff = int(0.8 * len(df))
            final_rewards.append(df["mean_reward"].iloc[cutoff:].mean())
            final_best.append(df["best_reward"].iloc[-1] if "best_reward" in df.columns else df["mean_reward"].max())

        final_rewards = np.array(final_rewards)
        final_best = np.array(final_best)
        label = "Original Balance" if name == "original" else "Balance + Contact Reward"
        print(f"  {label:<30} {np.mean(final_rewards):>7.2f} +/- {np.std(final_rewards):.2f}"
              f"       {np.mean(final_best):>7.2f}")

    print(f"{'='*70}")
    plt.close("all")


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Full-scale MAPPO comparison training")
    parser.add_argument("--total-frames", type=int, default=None,
                        help="Override total frames (default: 600K)")
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                        help="Override seed list")
    parser.add_argument("--num-envs", type=int, default=None,
                        help="Override number of parallel envs")
    parser.add_argument("--skip-plot", action="store_true",
                        help="Skip plot generation")
    parser.add_argument("--scenario", type=str, default=None,
                        choices=["original", "active"],
                        help="Train only one scenario (default: both)")
    args = parser.parse_args()

    cfg = CONFIG.copy()
    if args.total_frames:
        cfg["total_frames"] = args.total_frames
    if args.seeds:
        cfg["seeds"] = args.seeds
    if args.num_envs:
        cfg["num_envs"] = args.num_envs

    scenarios = [args.scenario] if args.scenario else ["original", "active"]
    seeds = cfg["seeds"]

    total_runs = len(scenarios) * len(seeds)
    est_time_per_run = cfg["total_frames"] / 3000  # rough estimate: ~3000 fps on CPU
    est_total_min = (est_time_per_run * total_runs) / 60

    print(f"\n{'#'*70}")
    print(f"  MAPPO FULL COMPARISON: Original vs Contact-Reward Balance")
    print(f"{'#'*70}")
    print(f"  Scenarios:    {scenarios}")
    print(f"  Seeds:        {seeds}")
    print(f"  Total runs:   {total_runs}")
    print(f"  Frames/run:   {cfg['total_frames']:,}")
    print(f"  Envs:         {cfg['num_envs']}")
    print(f"  PPO epochs:   {cfg['n_minibatch_iters']}")
    print(f"  LR:           {cfg['lr']}")
    print(f"  Network:      2×{cfg['hidden_dim']} Tanh, shared={cfg['share_params']}")
    print(f"  Est. time:    ~{est_total_min:.0f} min total")
    print(f"{'#'*70}\n")

    all_start = time.time()
    completed = 0

    for scenario_name in scenarios:
        for seed in seeds:
            completed += 1
            print(f"\n>>> Run {completed}/{total_runs}: {scenario_name} seed={seed}")
            try:
                train(scenario_name, seed, cfg)
            except Exception as e:
                print(f"\n  ERROR in {scenario_name}/seed_{seed}: {e}")
                traceback.print_exc()
                continue

    total_time = time.time() - all_start
    print(f"\n{'#'*70}")
    print(f"  ALL TRAINING COMPLETE in {total_time/60:.1f} minutes")
    print(f"{'#'*70}")

    if not args.skip_plot:
        print("\nGenerating comparison plots...")
        try:
            generate_comparison_plots(cfg)
        except Exception as e:
            print(f"  Plot generation failed: {e}")
            traceback.print_exc()

    print(f"\nResults in: {RESULTS_DIR}")
    print(f"Plots in:   {PLOTS_DIR}")


if __name__ == "__main__":
    main()
