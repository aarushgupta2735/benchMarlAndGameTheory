"""Quick debug: print batch keys from SyncDataCollector."""
import importlib.util
from pathlib import Path
import torch
from torchrl.collectors import SyncDataCollector
from torchrl.envs.libs.vmas import VmasEnv
from torchrl.envs import RewardSum, TransformedEnv
from torchrl.modules import MultiAgentMLP, ProbabilisticActor, TanhNormal
from tensordict.nn import TensorDictModule, TensorDictSequential
from torch import nn

SCENARIOS_DIR = Path(__file__).resolve().parent / "scenarios"

def _load(name):
    fmap = {"original": "balance_original.py"}
    spec = importlib.util.spec_from_file_location(f"balance_{name}", str(SCENARIOS_DIR / fmap[name]))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.Scenario

def make():
    ScenarioCls = _load("original")
    env = VmasEnv(scenario=ScenarioCls(), num_envs=4, device="cpu", seed=0,
                  continuous_actions=True, max_steps=50, n_agents=3)
    env = TransformedEnv(env, RewardSum(in_keys=[env.reward_key], out_keys=[("agents","episode_reward")]))
    return env

env = make()
obs_spec = env.full_observation_spec["agents", "observation"]
act_spec = env.full_action_spec["agents", "action"]
obs_size = obs_spec.shape[-1]
act_size = act_spec.shape[-1]
n_agents = len(env.agents)

policy_net = MultiAgentMLP(n_agent_inputs=obs_size, n_agent_outputs=2*act_size,
    n_agents=n_agents, centralised=False, share_params=True, depth=2, num_cells=64, activation_class=nn.Tanh)
policy_module = TensorDictModule(policy_net, in_keys=[("agents","observation")], out_keys=[("agents","loc_scale")])
unbind = TensorDictModule(lambda x: x.split(act_size, dim=-1), in_keys=[("agents","loc_scale")], out_keys=[("agents","loc"),("agents","scale")])
policy = ProbabilisticActor(module=TensorDictSequential(policy_module, unbind),
    in_keys=[("agents","loc"),("agents","scale")], out_keys=[("agents","action")],
    distribution_class=TanhNormal, distribution_kwargs={"low": act_spec.space.low, "high": act_spec.space.high},
    return_log_prob=True, log_prob_key=("agents","sample_log_prob"))

print("reward_key:", env.reward_key)
print("done_keys:", env.done_keys)
print("done_keys_groups:", env.done_keys_groups if hasattr(env, "done_keys_groups") else "N/A")
print()

collector = SyncDataCollector(
    create_env_fn=make, policy=policy,
    frames_per_batch=200, total_frames=200, device="cpu", storing_device="cpu")

for batch in collector:
    print("BATCH shape:", batch.shape)
    print("\nALL KEYS (nested):")
    for k in sorted(str(x) for x in batch.keys(True, True)):
        print(f"  {k}")
    print("\nNEXT sub-keys:")
    if "next" in batch.keys():
        nxt = batch["next"]
        for k in sorted(str(x) for x in nxt.keys(True, True)):
            print(f"  next.{k}")
    break

collector.shutdown()
env.close()
print("\nDONE")
