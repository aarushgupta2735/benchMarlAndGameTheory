"""
Train MAPPO on original and active-contact balance scenarios.

Uses TorchRL's multi-agent PPO pipeline with VmasEnv.  This gives us full
control over which scenario class (original vs. active) is loaded, while
reusing the same efficient vectorised training loop.

Usage:
    conda activate benchmarl

    # Train original balance scenario
    python train_mappo.py --scenario original --seed 0

    # Train active-contact balance scenario
    python train_mappo.py --scenario active --seed 0

    # Run full comparison (3 seeds × 2 scenarios)
    python train_mappo.py --run-all

Results are saved to ./results/<scenario_name>/seed_<N>/
"""

import argparse
import csv
import importlib.util
import math
import os
import time
from pathlib import Path

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


# ---------------------------------------------------------------------------
#  Scenario loading helpers
# ---------------------------------------------------------------------------

def _load_scenario_class(scenario_name: str):
    """Import the Scenario class from our local scenario files."""
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


def make_env(scenario_name, num_envs, device, seed, max_steps, n_agents,
             continuous_actions=True, **scenario_kwargs):
    """Create a VmasEnv with our custom scenario class instance."""
    ScenarioCls = _load_scenario_class(scenario_name)
    env = VmasEnv(
        scenario=ScenarioCls(),
        num_envs=num_envs,
        device=device,
        seed=seed,
        continuous_actions=continuous_actions,
        max_steps=max_steps,
        n_agents=n_agents,
        categorical_actions=True,
        clamp_actions=True,
        **scenario_kwargs,
    )
    env = TransformedEnv(env, RewardSum(in_keys=[env.reward_key], out_keys=[("agents", "episode_reward")]))
    return env


# ---------------------------------------------------------------------------
#  Build MAPPO policy + critic
# ---------------------------------------------------------------------------

def build_policy_and_critic(env, cfg):
    n_agents = len(env.agents)
    obs_spec = env.full_observation_spec["agents", "observation"]
    act_spec = env.full_action_spec["agents", "action"]
    obs_size = obs_spec.shape[-1]
    act_size = act_spec.shape[-1]

    # ── Policy (shared-parameter MLP) ──
    policy_net = MultiAgentMLP(
        n_agent_inputs=obs_size,
        n_agent_outputs=2 * act_size,   # mean + std for TanhNormal
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
    # Split loc/scale and wrap in a probabilistic actor
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

    # ── Critic (centralised: sees all agents' observations) ──
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


# ---------------------------------------------------------------------------
#  CSV Logger
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
#  Training Loop
# ---------------------------------------------------------------------------

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
    env = make_env(
        scenario_name=scenario_name,
        num_envs=cfg["num_envs"],
        device=device,
        seed=seed,
        max_steps=cfg["max_steps"],
        n_agents=cfg["n_agents"],
        **scenario_kwargs,
    )

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
        create_env_fn=lambda: make_env(
            scenario_name=scenario_name,
            num_envs=cfg["num_envs"],
            device=device,
            seed=seed,
            max_steps=cfg["max_steps"],
            n_agents=cfg["n_agents"],
            **scenario_kwargs,
        ),
        policy=policy,
        frames_per_batch=cfg["frames_per_batch"],
        total_frames=cfg["total_frames"],
        device=device,
        storing_device=device,
    )

    # ── Replay buffer (for minibatch iteration) ──
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
    n_agents_actual = len(env.agents)

    print(f"\n{'='*60}")
    print(f"  MAPPO Training: {scenario_name} | seed={seed} | device={device}")
    print(f"  Total frames: {cfg['total_frames']:,} | Frames/batch: {cfg['frames_per_batch']:,}")
    print(f"{'='*60}\n")

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

        # ── Logging ──
        mean_loss = sum(train_losses) / max(len(train_losses), 1)

        # Episode reward: endings in ("next", "done"); ep_reward in ("next", "agents", "episode_reward")
        mean_reward = float("nan")
        try:
            next_td = batch.get("next")
            if next_td is not None:
                done_vals = next_td.get("done", None)
                ep_rew_vals = next_td.get(("agents", "episode_reward"), None)
                if done_vals is not None and ep_rew_vals is not None:
                    done_mask = done_vals.squeeze(-1)
                    if done_mask.any():
                        while ep_rew_vals.dim() > done_mask.dim() + 1:
                            ep_rew_vals = ep_rew_vals.squeeze(-1)
                        ep_rew_agent_mean = ep_rew_vals.mean(dim=-1)
                        mean_reward = ep_rew_agent_mean[done_mask].mean().item()
        except Exception:
            pass
        if not math.isnan(mean_reward) and mean_reward > best_reward:
            best_reward = mean_reward

        elapsed = time.time() - t_start
        fps = total_frames_so_far / max(elapsed, 1e-6)

        log_data = {
            "step": total_frames_so_far,
            "iteration": n_iters,
            "mean_reward": mean_reward,
            "best_reward": best_reward,
            "mean_loss": mean_loss,
            "fps": fps,
            "elapsed_s": elapsed,
        }
        logger.log(log_data)

        if n_iters % 5 == 1 or total_frames_so_far >= cfg["total_frames"]:
            print(
                f"  [{scenario_name}] iter={n_iters:4d}  "
                f"frames={total_frames_so_far:>8,}  "
                f"rew={mean_reward:>8.2f}  best={best_reward:>8.2f}  "
                f"loss={mean_loss:.4f}  "
                f"fps={fps:.0f}"
            )

    collector.shutdown()
    logger.close()
    env.close()

    print(f"\n  Training complete. Results: {log_dir}/progress.csv")
    return log_dir


# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------

def get_default_config():
    return dict(
        # Environment
        n_agents=3,
        max_steps=200,
        num_envs=32,
        # Training
        total_frames=300_000,
        frames_per_batch=6_000,
        minibatch_size=400,
        n_minibatch_iters=4,
        # Optimiser
        lr=5e-4,
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
    )


def main():
    parser = argparse.ArgumentParser(
        description="Train MAPPO on Balance scenarios (original vs contact-reward)"
    )
    parser.add_argument("--scenario", type=str, choices=["original", "active"],
                        help="Which scenario to train")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-all", action="store_true",
                        help="Run both scenarios with multiple seeds")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--n-agents", type=int, default=3)
    parser.add_argument("--total-frames", type=int, default=300_000)
    parser.add_argument("--frames-per-batch", type=int, default=6_000)
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--contact-reward-coeff", type=float, default=0.5,
                        help="Contact reward coefficient for the active scenario")

    args = parser.parse_args()
    cfg = get_default_config()

    # Override from CLI
    cfg["n_agents"] = args.n_agents
    cfg["total_frames"] = args.total_frames
    cfg["frames_per_batch"] = args.frames_per_batch
    cfg["num_envs"] = args.num_envs
    cfg["lr"] = args.lr
    cfg["device"] = args.device
    cfg["contact_reward_coeff"] = args.contact_reward_coeff

    if args.run_all:
        for scn in ["original", "active"]:
            for s in args.seeds:
                train(scn, s, cfg)
        print(f"\n  All done! Run: python compare_results.py")
    elif args.scenario:
        train(args.scenario, args.seed, cfg)
    else:
        parser.print_help()
        print("\nExamples:")
        print("  python train_mappo.py --scenario original --seed 0")
        print("  python train_mappo.py --scenario active --seed 0")
        print("  python train_mappo.py --run-all")


if __name__ == "__main__":
    main()
