from typing import Dict, List, Optional, TypeVar

import numpy as np
from scipy.interpolate import interp1d

from scenario_gym.trajectory import Trajectory
from scenario_gym.utils import ArrayLike

from .base import Entity

State = TypeVar("State")


class BatchReplayEntity:
    """
    A single object used to represent multiple entities.

    Will replay exact trajectories from OpenScenario files. Computation is
    vectorized for efficiency.
    """

    def __init__(self, timestep: Optional[float] = None):
        """Init the batch entity with no assigned entities."""
        self.entities: List[Entity] = []
        self.trajectories: List[Trajectory] = []
        self.timestep = timestep
        self.max_t = 0.0

    def step(self, state: State) -> Dict[Entity, ArrayLike]:
        """
        Take a single step in the gym.

        Returns the pose of each entity at the next timestamp.
        """
        new_poses = {}
        if len(self.entities) > 0:
            pos = self.fn(state.next_t)  # (m, num_ents)
            for e, p in zip(self.entities, pos):
                new_poses[e] = p
        return new_poses

    def add_entities(
        self,
        entities: List[Entity],
        trajs: List[Trajectory],
    ) -> None:
        """
        Add entities that are to be batched together.

        This will reset the entities in the scenario so all entities
        must be passed at once.

        Parameters
        ----------
        entities : List[Entity]
            The entities to be used.

        trajs : List[Trajectory]
            The trajectory for each entity.

        """
        self.entities.clear()
        self.trajectories.clear()
        self.max_t = 0.0
        if entities:
            self.entities.extend(entities)
            self.trajectories.extend(trajs)

            num_ents = len(self.entities)
            datas = []
            for t in self.trajectories:
                d = np.nan_to_num(t.data)
                if d.shape[0] == 1:
                    d = np.repeat(d, 2, axis=0)
                    d[-1, 0] += 1e-1  # to prevent nan
                datas.append(d)

            m = datas[0].shape[1] - 1
            ts = np.array(
                sorted(list(set([t for d in datas for t in d[:, 0]])))
            )  # (N,)
            self.max_t = ts[-1]

            interpd = []
            for d in datas:
                x = interp1d(
                    d[:, 0],
                    d[:, 1:].T,
                    bounds_error=False,
                    fill_value=(d[0, 1:], d[-1, 1:]),
                )(
                    ts
                ).T  # (N, m)
                interpd.append(x)

            X = np.concatenate(interpd, axis=1)  # (N, num_ents * m)
            if self.timestep:
                all_ts = np.arange(0.0, self.max_t, self.timestep)
                all_Xs = interp1d(
                    ts,
                    X.T,
                    bounds_error=False,
                    fill_value=(X[0], X[-1]),
                )(all_ts).T
                self.fn = lambda t: all_Xs[np.abs(all_ts - t).argmin()].reshape(
                    num_ents, m
                )
            else:
                interp = interp1d(
                    ts,
                    X.T,
                    bounds_error=False,
                    fill_value=(X[0], X[-1]),
                )
                self.fn = lambda t: interp(t).reshape(num_ents, m)
