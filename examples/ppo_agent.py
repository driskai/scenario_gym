import argparse
import os
import random
from typing import Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor
    from torch.distributions import Beta
except ImportError:
    raise ImportError(
        """
This example requires torch.
You can install it directly with `pip install torch==1.11.0`.
Or, you can install additional requirements for examples with:
`pip install scenario_gym[examples]`.
"""
    )

import scenario_gym
from scenario_gym.action import VehicleAction
from scenario_gym.agent import Agent
from scenario_gym.controller import Controller, VehicleController
from scenario_gym.entity import Entity
from scenario_gym.manager import ScenarioManager
from scenario_gym.metrics import Metric
from scenario_gym.scenario import Scenario
from scenario_gym.sensor import RasterizedMapSensor, Sensor
from scenario_gym.state import State


class PPOModel(nn.Module):
    """PPO model outputing parameters of a Beta distribution over the actions."""

    def __init__(
        self,
        n_actions: int,
        input_channels: int,
        d_model: int,
        kernel_size: int,
        stride: int,
    ):
        super().__init__()
        self.n_actions = n_actions
        self.conv1 = nn.Conv2d(
            input_channels,
            d_model,
            kernel_size=kernel_size,
            stride=stride,
        )
        self.conv2 = nn.Conv2d(
            d_model, d_model, kernel_size=kernel_size, stride=stride
        )
        self.linear = nn.Linear(d_model, d_model)
        self.policy_linear = nn.Linear(d_model, 2 * n_actions)
        self.value_linear = nn.Linear(d_model, 1)

    def forward(self, s):
        return self.policy(s)

    def base(self, s):
        """
        (batch, C, W, H) -> (batch, d_model)
        """
        x = F.relu(self.conv1(s))
        x, _ = self.conv2(x).flatten(2).max(2)
        x = F.relu(self.linear(x))
        return x

    def policy(self, s):
        """
        (batch, C, W, H) -> (batch, n_actions, 2)
        """
        return torch.exp(self.policy_linear(self.base(s))).reshape(
            -1, self.n_actions, 2
        )

    def value(self, s):
        """
        (batch, C, W, H) -> (batch,)
        """
        return self.value_linear(self.base(s)).squeeze(1)


class AdvantageBuffer:
    """
    Experience replay buffer that also stores advantage estimates.
    """

    def __init__(self, buffer_size: int):
        self.buffer_size = buffer_size
        self.reset()

    def reset(self) -> None:
        self.state = []
        self.action = []
        self.reward = []
        self.done = []
        self.state_next = []
        self.probs = []
        self.advs = []
        self.tds = np.array([])
        self.n = 0

    def __len__(self) -> int:
        return len(self.state)

    def update(
        self,
        s: np.ndarray,
        a: np.ndarray,
        r: float,
        d: bool,
        s_next: np.ndarray,
        pi: np.ndarray,
    ) -> None:
        self.state.append(s)
        self.action.append(a)
        self.reward.append(r)
        self.done.append(d)
        self.state_next.append(s_next)
        self.probs.append(pi)
        self.advs.append(None)
        if self.n < self.buffer_size:
            self.n += 1
        else:
            self.state.pop(0)
            self.action.pop(0)
            self.reward.pop(0)
            self.done.pop(0)
            self.state_next.pop(0)
            self.probs.pop(0)
            self.advs.pop(0)
            self.tds = self.tds[-self.buffer_size :]

    @property
    def data(self) -> Tuple[Tensor, ...]:
        s = torch.from_numpy(np.array(self.state, dtype=np.float32))
        a = torch.from_numpy(np.array(self.action, dtype=np.float32))
        r = torch.from_numpy(np.array(self.reward, dtype=np.float32))
        d = torch.from_numpy(np.array(self.done, dtype=np.float32))
        s_next = torch.from_numpy(np.array(self.state_next, dtype=np.float32))
        pi = torch.from_numpy(np.array(self.probs, dtype=np.float32))
        advs = torch.from_numpy(np.array(self.advs, dtype=np.float32)).squeeze()
        tds = torch.from_numpy(self.tds.astype(np.float32))
        return s, a, r, d, s_next, pi, advs, tds

    def sample(self, batch_size) -> Tuple[Tensor, ...]:
        idxs = np.random.choice(len(self), (batch_size,), replace=False)
        s = torch.from_numpy(np.array(self.state, dtype=np.float32)[idxs])
        a = torch.from_numpy(np.array(self.action, dtype=np.float32)[idxs])
        r = torch.from_numpy(np.array(self.reward, dtype=np.float32)[idxs])
        d = torch.from_numpy(np.array(self.done, dtype=np.float32)[idxs])
        s_next = torch.from_numpy(np.array(self.state_next, dtype=np.float32)[idxs])
        pi = torch.from_numpy(np.array(self.probs, dtype=np.float32)[idxs])
        advs = torch.from_numpy(
            np.array(self.advs, dtype=np.float32)[idxs]
        ).squeeze()
        tds = torch.from_numpy(self.tds[idxs].astype(np.float32))
        return s, a, r, d, s_next, pi, advs, tds


class MapSensor(RasterizedMapSensor):
    def _step(self, state):
        obs = super()._step(state)
        return obs.map


class PPOAgent(Agent):
    """An agent implementing a PPO policy."""

    def __init__(
        self,
        entity: Entity,
        controller: Controller,
        sensor: Sensor,
        buffer: AdvantageBuffer,
        model: PPOModel,
        optimizer: torch.optim.Adam,
        target_speed: float = 5.0,
        max_steer: float = np.pi / 4,
        gamma: float = 0.99,
        lmda: float = 0.8,
        v_weight: float = 0.9,
        e_weight: float = 0.01,
        epsilon: float = 0.01,
    ):
        super().__init__(entity, controller, sensor)
        self.max_steer = max_steer
        self.target_speed = target_speed
        self.buffer = buffer
        self.model = model
        self.optimizer = optimizer
        self.train()
        self.gamma = gamma
        self.lmda = lmda
        self.v_weight = v_weight
        self.e_weight = e_weight
        self.epsilon = epsilon

    def _reset(self):
        self.s_prev = None
        self.a_prev = None
        self.r_prev = None
        self.pi_prev = None
        self.ep_length = 0
        self.ep_reward = 0.0

    def step(self, state: State):
        obs = self.sensor.step(state).transpose(2, 0, 1)[1:, :, :]
        self.r_prev = self.get_reward(state)
        if self.s_prev is not None and self.training:
            self.buffer.update(
                self.s_prev,
                self.a_prev,
                self.r_prev,
                state.is_done,
                obs,
                self.pi_prev,
            )
        self.ep_length += 1
        self.ep_reward += self.r_prev
        self.s_prev = obs
        action = self._step(obs)
        self.last_action = action
        return self.controller.step(state, action)

    def _step(self, obs: np.ndarray):
        s = torch.from_numpy(obs[None]).float()  # (1, ...)
        with torch.no_grad():
            a = self.model(s).squeeze()  # (2,)
        dist = Beta(a[0] + 1e-10, a[1] + 1e-10)
        if self.training:
            a = dist.sample()
        else:
            a = a[0] / (a.sum() + 1e-10)
        self.pi_prev = dist.log_prob(a).numpy()
        self.a_prev = a.numpy()
        steer = (self.a_prev * 2.0 - 1.0) * self.max_steer
        return VehicleAction(
            0.5 * (self.target_speed - self.controller.speed),
            steer,
        )

    def train(self) -> None:
        self.model.train()
        self.training = True

    def eval(self) -> None:
        self.model.eval()
        self.training = False

    def get_reward(self, state: State) -> float:
        return 0.1

    def compute_advantages(self, l: int) -> None:
        s = np.array(self.buffer.state[-l:], dtype=np.float32)
        r = np.array(self.buffer.reward[-l:], dtype=np.float32)
        d = np.array(self.buffer.done[-l:], dtype=np.float32)
        s_next = np.array(self.buffer.state_next[-l:], dtype=np.float32)

        with torch.no_grad():
            s = torch.from_numpy(s)
            v = self.model.value(s).cpu().numpy()
            s_next = torch.from_numpy(s_next)
            v_next = self.model.value(s_next).cpu().numpy()
        td_target = r + self.gamma * v * (1.0 - d)
        delta = td_target - v  # (l)

        # calculate advantage
        advantages, adv = [], 0.0
        for d in delta[::-1]:
            adv = self.gamma * self.lmda * adv + d
            advantages.append([adv])
        advantages.reverse()

        # update tds
        if self.buffer.tds.shape == (0,):
            self.buffer.tds = td_target
        else:
            self.buffer.tds = np.concatenate(
                [self.buffer.tds, td_target],
                axis=0,
            )[-self.buffer.buffer_size :]
        self.buffer.advs[-l:] = advantages

    def compute_loss(
        self,
        s: Tensor,
        a: Tensor,
        r: Tensor,
        d: Tensor,
        s_next: Tensor,
        pi: Tensor,
        advs: Tensor,
        tds: Tensor,
    ) -> Tensor:
        policy = self.model.policy(s).squeeze(1)  # (batch, 2)
        dist = Beta(policy[:, 0], policy[:, 1])
        pi_new = dist.log_prob(a)
        ratio = pi_new - pi
        actor_loss = torch.minimum(
            ratio * advs,
            torch.clamp(
                ratio,
                np.log(1.0 - self.epsilon),
                np.log(1.0 + self.epsilon),
            )
            * advs,
        )
        e = dist.entropy()
        v_loss = 0.5 * (self.model.value(s) - tds).squeeze().pow(2)
        loss = -actor_loss + self.v_weight * v_loss - self.e_weight * e
        return loss.mean()

    def update_model(self, batch_size: int = 32) -> float:
        s, a, r, d, s_next, pi_old, advs, tds = self.buffer.sample(batch_size)
        loss = self.compute_loss(s, a, r, d, s_next, pi_old, advs, tds)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()


class PPOConfig(ScenarioManager):
    """Holds a replay buffer and neural network for a PPO agent."""

    PARAMETERS = {
        "timestep": 0.25,
        "terminal_conditions": ["ego_off_road"],
        "buffer_size": 256 * 8,
        "input_channels": 1,
        "d_model": 12,
        "kernel_size": 3,
        "stride": 1,
        "lr": 1e-2,
        "gamma": 0.99,
        "lmda": 0.9,
        "v_weight": 0.95,
        "e_weight": 0.0,
        "epsilon": 0.25,
        "max_steer": 0.3,
        "max_time": 120.0,
    }

    def __init__(
        self,
        config_path: str = None,
        **kwargs,
    ):
        super().__init__(config_path=config_path, **kwargs)
        self.terminal_conditions.append(lambda s: s.t > self.PARAMETERS["max_time"])
        self.buffer = AdvantageBuffer(self.buffer_size)
        self.model = PPOModel(
            1, self.input_channels, self.d_model, self.kernel_size, self.stride
        )
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def create_agent(self, scenario: Scenario, entity: Entity) -> Agent:
        if entity.ref == "ego":
            controller = VehicleController(entity, max_steer=self.max_steer)
            sensor = MapSensor(entity)
            return PPOAgent(
                entity,
                controller,
                sensor,
                self.buffer,
                self.model,
                self.optimizer,
                gamma=self.gamma,
                lmda=self.lmda,
                v_weight=self.v_weight,
                e_weight=self.e_weight,
                epsilon=self.epsilon,
                max_steer=self.max_steer,
            )

    def save_config(self, path: Optional[str] = None) -> None:
        """Save the parameters and state dict."""
        super().save_config(path=path + ".yml")
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            path + "_state_dict",
        )


class EpisodicReward(Metric):
    def __init__(self, entity_ref: str = "ego"):
        super().__init__()
        self.entity_ref = entity_ref
        self.r = 0.0
        self.agent = None

    def _reset(self, state):
        self.r = 0.0
        self.agent = state.agents[self.entity_ref]

    def _step(self, state):
        self.r += self.agent.r_prev

    def get_state(self) -> float:
        return self.r


def seed_all(seed: int) -> None:
    """Make deterministic."""
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a PPO agent to steer a car."
    )
    parser.add_argument(
        "--seed",
        default=11,
        type=int,
        help="Set the random seed.",
    )
    parser.add_argument(
        "--max_steer",
        default=0.3,
        type=float,
        help="Maximum steering angle.",
    )
    parser.add_argument(
        "--gamma",
        default=0.99,
        type=float,
        help="Discount factor.",
    )
    parser.add_argument(
        "--lmda",
        default=0.97,
        type=float,
        help="Bootstrap factor.",
    )
    parser.add_argument(
        "--epsilon",
        default=0.2,
        type=float,
        help="Clip for loss function.",
    )
    parser.add_argument(
        "--v_weight",
        default=0.95,
        type=float,
        help="Relative weight for value loss" "against the policy loss.",
    )
    parser.add_argument(
        "--e_weight",
        default=0.0,
        type=float,
        help="Relative weight for entropy" "against the policy loss.",
    )
    parser.add_argument(
        "--lr",
        default=1e-2,
        type=float,
        help="Learning rate.",
    )
    parser.add_argument(
        "--episodes",
        default=375,
        type=int,
        help="Number of training episodes.",
    )
    parser.add_argument(
        "--batch_size",
        default=256,
        type=int,
        help="Batch size of model updates.",
    )
    parser.add_argument(
        "--scenario_path",
        default=None,
        type=str,
        help="Path to the scenario file.",
    )
    parser.add_argument(
        "--verbose",
        default=-1,
        type=int,
        help="Print frequency per episode of loss and reward. -1 is no printing.",
    )
    return parser.parse_args()


def run(FLAGS):
    seed_all(FLAGS.seed)
    config = PPOConfig(
        gamma=FLAGS.gamma,
        lmda=FLAGS.lmda,
        epsilon=FLAGS.epsilon,
        v_weight=FLAGS.v_weight,
        e_weight=FLAGS.e_weight,
        lr=FLAGS.lr,
        max_steer=FLAGS.max_steer,
    )
    gym = scenario_gym.ScenarioGym(
        timestep=config.timestep,
        terminal_conditions=config.terminal_conditions,
    )
    gym.load_scenario(FLAGS.scenario_path, create_agent=config.create_agent)
    metric = EpisodicReward()
    gym.metrics.append(metric)

    rewards, loss = 0.0, 0.0
    for episode in range(FLAGS.episodes):
        agent = gym.state.agents["ego"]
        agent.train()
        gym.rollout()
        agent.buffer.update(
            agent.s_prev,
            agent.a_prev,
            agent.r_prev,
            True,
            agent.sensor._step(gym.state).transpose(2, 0, 1)[1:, :, :],
            agent.pi_prev,
        )
        agent.compute_advantages(agent.ep_length)
        if len(agent.buffer) > FLAGS.batch_size:
            loss = agent.update_model(FLAGS.batch_size)
        if FLAGS.verbose > 0:
            rewards += metric.get_state() / FLAGS.verbose
            if episode % FLAGS.verbose == 0:
                print(
                    "Episode {} Reward {:.4} Loss {:.4}".format(
                        episode, rewards, loss
                    )
                )
                rewards = 0.0

    # save a result
    torch.save(config.model.state_dict(), "./test_state_dict")
    config.save_config("./ppo")

    # record a video of the agent
    gym.reset_scenario()
    gym.state.agents["ego"].eval()
    gym.rollout(render=True)


if __name__ == "__main__":
    flags = parse_args()

    if flags.scenario_path is None:
        flags.scenario_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "tests",
            "input_files",
            "Scenarios",
            "d9726503-e04a-4e8b-b487-8805ef790c93.xosc",
        )
    run(flags)
