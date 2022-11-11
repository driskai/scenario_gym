# Implement Mobileye's 5 rules of Responsibility-Sensitive Safety (RSS) as
# a metric compatible with driving gym.
# The five rules are
#   1)  Safe longitudinal distance to vehicles in the same lane
#   2)  Safe lateral distance to vehicles when merging lane
#   3)  Forfeiting right of way at junctions when another road user undermines
#       correct priority
#   4)  Additional caution in areas of occlusion
#   5)  Avoid a collision by any means without jeopardising safety of other
#       road users

import string
from enum import Enum
from typing import Dict, List

from scenario_gym.metrics import Metric
from scenario_gym.road_network import road_network
from scenario_gym.state import State

from .callback import RSSDistances


class Rules(Enum):
    """Enumerate the five rules."""

    safe_longitudinal = 0
    safe_lateral = 1


class RSSBehaviourDetection:
    """
    Find behaviours corresponding to the five RSS rules.

    Instantiates class based on current timestep state and
    parameters. Calls each of the behaviour methods, one per
    RSS rule. These methods are lightweight, with most computation
    performed in the rss callback. Each method returns a bool
    value corresponding to whether or not the rule is obeyed at each
    timestep -- if the rule has not already been failed.
    """

    # Initialise and call the behaviour functions
    def __init__(
        self,
        metrics: Dict,
        ego: Dict,
        entities: List[Dict],
        safe_distances: List[List[float]],
        road_network: road_network,
        dt: float,
        intersect: string,
        collisions,
    ):
        self.metrics = metrics
        self.ego = ego
        self.entities = entities
        self.safe_distances = safe_distances
        self.road_network = road_network
        self.dt = dt
        self.intersect = intersect
        self.collisions = collisions

    def __call__(self):
        """Call behaviour methods for current timestep."""
        outcomes = {}
        for rule in Rules:
            outcome = getattr(self, rule.name)
            outcomes[rule.name] = outcome()
        intersect = self.intersect
        return outcomes, intersect

    # Behaviour functions
    def safe_longitudinal(self) -> bool:
        """
        Rule 1: ego at a safe longitudinal distance.

        Returns True if longitudinal distance is less than the minimum
        safe distance when the lateral distance is also unsafe, with the safe
        longitudinal distance crossed last.
        """
        if not self.metrics["safe_longitudinal"]:
            # Already found
            return True
        # Check if any entities have been flagged as being unsafe longitudinally.
        for entity_record in self.intersect:
            if "unsafe_longitudinal" in entity_record:
                return False
        return True

    def safe_lateral(self) -> bool:
        """
        Rule 2: ego at a safe lateral distance.

        Returns True if lateral distance is less than the minimum
        safe distance when the longitudinal distance is also unsafe, with the
        safe lateral distance crossed last.
        """
        if not self.metrics["safe_lateral"]:
            # Already found
            return True
        # Check if any entities have been flagged as being unsafe laterally.
        for entity_record in self.intersect:
            if "unsafe_lateral" in entity_record:
                return False
        return True


class RSS(Metric):
    """
    Determine if the ego follow the 5 rules of RSS.

    _reset() resets state with each rule set to True by default

    _step() calls RSSBehaviourDetection to check if the ego is
    obeying the RSS rules at the current timestep. rss/callback
    is responsible for bulk of the computation and is called prior
    to the metric call at each timestep.

    get_state() returns dictionary of bools, one per rule
    --  True: the rule is obeyed at every timestep
    --  False: the rule is disobeyed in at least one timestep
        (may not be fault of ego - separate metric for blame)
    """

    required_callbacks = [RSSDistances]

    def _reset(self, state: State) -> None:
        """Reset behaviour."""
        self.rss_callback = self.callbacks[0]
        self.behaviour = None
        self.ego = state.scenario.entities[0]
        self.metrics_ = {rule.name: True for rule in Rules}

    def _step(self, state: State) -> None:
        """Update the metric to find behaviour at the point of interest."""
        if state.t == 0.0:
            # Require at least two poses to calculate velocity
            return

        ego, entities, safe_distances, intersect = (
            self.rss_callback.ego_params,
            self.rss_callback.entity_params,
            self.rss_callback.safe_distances,
            self.rss_callback.intersect,
        )
        rules = RSSBehaviourDetection(
            metrics=self.metrics_,
            ego=ego,
            entities=entities,
            safe_distances=safe_distances,
            road_network=state.scenario.road_network,
            dt=state.dt,
            intersect=self.rss_callback.intersect,
            collisions=state.collisions(),
        )

        outcomes, intersect = rules()
        self.intersect = intersect
        for rule, outcome in outcomes.items():
            if outcome is False:
                self.metrics_[rule] = outcome

    def get_state(self) -> Dict[str, bool]:
        """Return state."""
        return self.metrics_
