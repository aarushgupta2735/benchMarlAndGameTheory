#  Modified VMAS Balance Scenario — with Rod-Contact Activity Reward
#  Based on: https://github.com/proroklab/VectorizedMultiAgentSimulator
#  Modification: Agents receive a small bonus for maintaining contact with the rod (line).
#
#  REWARD DESIGN & CALIBRATION
#  ───────────────────────────
#  Original rewards:
#    pos_rew  ∈ [-100, +100] per step  (shaping_factor=100, package-to-goal distance delta)
#    ground_rew = -10                   (penalty when line/package hits the floor)
#
#  Contact reward (new):
#    Each agent that is within `contact_threshold` distance of the line gets
#    +contact_reward_coeff per step. The total contact bonus is summed over all
#    agents and shared (same as the other rewards).
#
#    Default: contact_reward_coeff = 0.5, contact_threshold = agent_radius * 3.5
#
#    With 3 agents all in contact: max bonus = 3 × 0.5 = 1.5 per step
#    This is ~1.5% of the pos_rew scale — enough to break ties in favour of
#    staying near the rod, but never enough to override the main task objective.
#
#    The coefficient can be tuned via the `contact_reward_coeff` kwarg.

import torch

from vmas import render_interactively
from vmas.simulator.core import Agent, Box, Landmark, Line, Sphere, World
from vmas.simulator.heuristic_policy import BaseHeuristicPolicy
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.utils import Color, ScenarioUtils, Y


class Scenario(BaseScenario):
    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        self.n_agents = kwargs.pop("n_agents", 3)
        self.package_mass = kwargs.pop("package_mass", 5)
        self.random_package_pos_on_line = kwargs.pop("random_package_pos_on_line", True)

        # ── NEW: contact-reward hyperparameters ──
        self.contact_reward_coeff = kwargs.pop("contact_reward_coeff", 0.5)
        self.contact_threshold = kwargs.pop("contact_threshold", None)  # auto-set below

        ScenarioUtils.check_kwargs_consumed(kwargs)

        assert self.n_agents > 1

        self.line_length = 0.8
        self.agent_radius = 0.03

        # Auto-calibrate contact threshold: ~3.5× agent radius gives a small
        # "near the rod" zone.  This is generous enough that agents actively
        # supporting the rod always qualify, but far-away agents do not.
        if self.contact_threshold is None:
            self.contact_threshold = self.agent_radius * 3.5

        self.shaping_factor = 100
        self.fall_reward = -10

        self.visualize_semidims = False

        # Make world
        world = World(batch_dim, device, gravity=(0.0, -0.05), y_semidim=1)
        # Add agents
        for i in range(self.n_agents):
            agent = Agent(
                name=f"agent_{i}",
                shape=Sphere(self.agent_radius),
                u_multiplier=0.7,
            )
            world.add_agent(agent)

        goal = Landmark(
            name="goal",
            collide=False,
            shape=Sphere(),
            color=Color.LIGHT_GREEN,
        )
        world.add_landmark(goal)
        self.package = Landmark(
            name="package",
            collide=True,
            movable=True,
            shape=Sphere(),
            mass=self.package_mass,
            color=Color.RED,
        )
        self.package.goal = goal
        world.add_landmark(self.package)

        self.line = Landmark(
            name="line",
            shape=Line(length=self.line_length),
            collide=True,
            movable=True,
            rotatable=True,
            mass=5,
            color=Color.BLACK,
        )
        world.add_landmark(self.line)

        self.floor = Landmark(
            name="floor",
            collide=True,
            shape=Box(length=10, width=1),
            color=Color.WHITE,
        )
        world.add_landmark(self.floor)

        self.pos_rew = torch.zeros(batch_dim, device=device, dtype=torch.float32)
        self.ground_rew = self.pos_rew.clone()
        self.contact_rew = self.pos_rew.clone()  # NEW
        self.n_agents_in_contact = torch.zeros(batch_dim, device=device, dtype=torch.float32)  # NEW

        return world

    def reset_world_at(self, env_index: int = None):
        goal_pos = torch.cat(
            [
                torch.zeros(
                    (1, 1) if env_index is not None else (self.world.batch_dim, 1),
                    device=self.world.device,
                    dtype=torch.float32,
                ).uniform_(
                    -1.0,
                    1.0,
                ),
                torch.zeros(
                    (1, 1) if env_index is not None else (self.world.batch_dim, 1),
                    device=self.world.device,
                    dtype=torch.float32,
                ).uniform_(
                    0.0,
                    self.world.y_semidim,
                ),
            ],
            dim=1,
        )
        line_pos = torch.cat(
            [
                torch.zeros(
                    (1, 1) if env_index is not None else (self.world.batch_dim, 1),
                    device=self.world.device,
                    dtype=torch.float32,
                ).uniform_(
                    -1.0 + self.line_length / 2,
                    1.0 - self.line_length / 2,
                ),
                torch.full(
                    (1, 1) if env_index is not None else (self.world.batch_dim, 1),
                    -self.world.y_semidim + self.agent_radius * 2,
                    device=self.world.device,
                    dtype=torch.float32,
                ),
            ],
            dim=1,
        )
        package_rel_pos = torch.cat(
            [
                torch.zeros(
                    (1, 1) if env_index is not None else (self.world.batch_dim, 1),
                    device=self.world.device,
                    dtype=torch.float32,
                ).uniform_(
                    (
                        -self.line_length / 2 + self.package.shape.radius
                        if self.random_package_pos_on_line
                        else 0.0
                    ),
                    (
                        self.line_length / 2 - self.package.shape.radius
                        if self.random_package_pos_on_line
                        else 0.0
                    ),
                ),
                torch.full(
                    (1, 1) if env_index is not None else (self.world.batch_dim, 1),
                    self.package.shape.radius,
                    device=self.world.device,
                    dtype=torch.float32,
                ),
            ],
            dim=1,
        )

        for i, agent in enumerate(self.world.agents):
            agent.set_pos(
                line_pos
                + torch.tensor(
                    [
                        -(self.line_length - agent.shape.radius) / 2
                        + i
                        * (self.line_length - agent.shape.radius)
                        / (self.n_agents - 1),
                        -self.agent_radius * 2,
                    ],
                    device=self.world.device,
                    dtype=torch.float32,
                ),
                batch_index=env_index,
            )

        self.line.set_pos(
            line_pos,
            batch_index=env_index,
        )
        self.package.goal.set_pos(
            goal_pos,
            batch_index=env_index,
        )
        self.line.set_rot(
            torch.zeros(1, device=self.world.device, dtype=torch.float32),
            batch_index=env_index,
        )
        self.package.set_pos(
            line_pos + package_rel_pos,
            batch_index=env_index,
        )

        self.floor.set_pos(
            torch.tensor(
                [
                    0,
                    -self.world.y_semidim
                    - self.floor.shape.width / 2
                    - self.agent_radius,
                ],
                device=self.world.device,
            ),
            batch_index=env_index,
        )
        self.compute_on_the_ground()
        if env_index is None:
            self.global_shaping = (
                torch.linalg.vector_norm(
                    self.package.state.pos - self.package.goal.state.pos, dim=1
                )
                * self.shaping_factor
            )
        else:
            self.global_shaping[env_index] = (
                torch.linalg.vector_norm(
                    self.package.state.pos[env_index]
                    - self.package.goal.state.pos[env_index]
                )
                * self.shaping_factor
            )

    def compute_on_the_ground(self):
        self.on_the_ground = self.world.is_overlapping(
            self.line, self.floor
        ) + self.world.is_overlapping(self.package, self.floor)

    def _compute_contact_reward(self):
        """Compute the rod-contact activity reward.

        For each agent, we measure its distance to the line (rod).
        If the distance is within `contact_threshold`, the agent counts as
        "in contact" and contributes `contact_reward_coeff` to the shared
        team reward.

        This uses VMAS's `world.get_distance()` which returns the distance
        between entity boundaries (negative = overlapping).
        """
        self.contact_rew[:] = 0
        self.n_agents_in_contact = torch.zeros(
            self.world.batch_dim, device=self.world.device, dtype=torch.float32
        )

        for agent in self.world.agents:
            dist_to_line = self.world.get_distance(agent, self.line)
            in_contact = dist_to_line <= self.contact_threshold
            self.n_agents_in_contact += in_contact.float()
            self.contact_rew += in_contact.float() * self.contact_reward_coeff

    def reward(self, agent: Agent):
        is_first = agent == self.world.agents[0]

        if is_first:
            self.pos_rew[:] = 0
            self.ground_rew[:] = 0

            self.compute_on_the_ground()
            self.package_dist = torch.linalg.vector_norm(
                self.package.state.pos - self.package.goal.state.pos, dim=1
            )

            self.ground_rew[self.on_the_ground] = self.fall_reward

            global_shaping = self.package_dist * self.shaping_factor
            self.pos_rew = self.global_shaping - global_shaping
            self.global_shaping = global_shaping

            # ── NEW: compute contact reward once per step ──
            self._compute_contact_reward()

        return self.ground_rew + self.pos_rew + self.contact_rew

    def observation(self, agent: Agent):
        # Same observations as original — no changes needed.
        # The agent can implicitly learn that staying near the rod is good
        # from the reward signal; we don't need to add contact as an obs.
        return torch.cat(
            [
                agent.state.pos,
                agent.state.vel,
                agent.state.pos - self.package.state.pos,
                agent.state.pos - self.line.state.pos,
                self.package.state.pos - self.package.goal.state.pos,
                self.package.state.vel,
                self.line.state.vel,
                self.line.state.ang_vel,
                self.line.state.rot % torch.pi,
            ],
            dim=-1,
        )

    def done(self):
        return self.on_the_ground + self.world.is_overlapping(
            self.package, self.package.goal
        )

    def info(self, agent: Agent):
        info = {
            "pos_rew": self.pos_rew,
            "ground_rew": self.ground_rew,
            "contact_rew": self.contact_rew,           # NEW: for logging
            "n_agents_in_contact": self.n_agents_in_contact,  # NEW: for analysis
        }
        return info


class HeuristicPolicy(BaseHeuristicPolicy):
    def compute_action(self, observation: torch.Tensor, u_range: float) -> torch.Tensor:
        batch_dim = observation.shape[0]

        index_package_goal_pos = 8
        dist_package_goal = observation[
            :, index_package_goal_pos : index_package_goal_pos + 2
        ]

        y_distance_ge_0 = dist_package_goal[:, Y] >= 0

        if self.continuous_actions:
            action_agent = torch.clamp(
                torch.stack(
                    [
                        torch.zeros(batch_dim, device=observation.device),
                        -dist_package_goal[:, Y],
                    ],
                    dim=1,
                ),
                min=-u_range,
                max=u_range,
            )
            action_agent[:, Y][y_distance_ge_0] = 0
        else:
            action_agent = torch.full((batch_dim,), 4, device=observation.device)
            action_agent[y_distance_ge_0] = 0
        return action_agent


if __name__ == "__main__":
    render_interactively(
        __file__,
        n_agents=3,
        package_mass=5,
        random_package_pos_on_line=True,
        control_two_agents=True,
        contact_reward_coeff=0.5,  # Try tweaking this!
    )
