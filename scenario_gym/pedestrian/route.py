import random
from itertools import chain
from typing import Dict, List, Optional, Tuple

import numpy as np

from scenario_gym.road_network import RoadNetwork


class RouteFinder:
    """Find routes along walkable areas in the road network."""

    def __init__(self, rn: RoadNetwork):
        """Construct the graph representation."""
        self.rn = rn
        (
            self.graph,
            self.node_to_idx,
            self.node_data,
        ) = make_pedestrian_connection_graph(rn)

    def find_route(
        self, start: np.ndarray, finish: np.ndarray
    ) -> Optional[List[Tuple[float, float]]]:
        """Find the route or return None if one cannot be found."""
        return find_route(self.graph, self.node_data, start, finish)

    def generate_route(
        self,
        n: int,
        start: Optional[np.ndarray] = None,
        no_repeat: bool = False,
    ) -> List[Tuple[float, float]]:
        """Generate a route by a random walk."""
        if start is not None:
            n0 = min(
                self.node_data,
                key=lambda x: np.linalg.norm(self.node_data[x] - start),
            )
            route = [n0]
        else:
            route = [random.choice(self.graph.keys())]
        while len(route) < n:
            suc = self.graph[route[-1]]
            if no_repeat:
                suc = list(set(suc).difference(route))
            if not suc:
                break
            route.append(random.choice(suc))
        return [self.node_data[i] for i in route]


def make_pedestrian_connection_graph(
    rn: RoadNetwork,
) -> Tuple[Dict[int, List[int]], Dict[str, int], Dict[int, Tuple[float, float]]]:
    """
    Build a graph representing the walkable surface of the road network.

    Nodes are positions in the pavements and crossings with edges connecting
    nodes if pedestrians can walk between them.

    Parameters
    ----------
    rn : RoadNetwork
        The road network.

    Returns
    -------
    graph : Dict[int, List[int]]
        The graph represented as a dictionary with integer node indexes and
        lists of neighbours of each node as values.

    node_to_idx : Dict[str, int]
        A dictionary giving a string identifier of each node index. The string
        identifier is '{road_object_id}_{index_in_object}'.

    node_data : Dict[int, Tuple[float, float]]
        A dictionary giving the xy coordinates of each node.

    """
    graph = {}
    node_to_idx = {}
    node_data = {}
    pavement_coords = {}
    for p in rn.pavements:
        pavement_coords[p.id] = np.array(
            [
                np.array(p.center.interpolate(x, normalized=False).xy).squeeze()
                for x in np.linspace(0.0, p.center.length, int(p.center.length))
            ]
        )

    crossing_coords = {}
    for c in rn.crossings:
        crossing_coords[c.id] = np.array(
            [
                np.array(c.center.interpolate(x, normalized=False).xy).squeeze()
                for x in np.linspace(0.0, c.center.length, int(c.center.length))
            ]
        )

    for obj, coords in chain(pavement_coords.items(), crossing_coords.items()):
        for i, (x, y) in enumerate(coords):
            node_to_idx[f"{obj}_{i}"] = len(node_to_idx)
            graph[node_to_idx[f"{obj}_{i}"]] = []
            node_data[node_to_idx[f"{obj}_{i}"]] = (x, y)

    for obj, coords in chain(pavement_coords.items(), crossing_coords.items()):
        for i in range(len(coords) - 1):
            graph[node_to_idx[f"{obj}_{i}"]].append(node_to_idx[f"{obj}_{i+1}"])
            graph[node_to_idx[f"{obj}_{i+1}"]].append(node_to_idx[f"{obj}_{i}"])

    for c in rn.crossings:
        for p in c.pavements:
            c_coords, p_coords = crossing_coords[c.id], pavement_coords[p]
            c_idx, p_idx = np.unravel_index(
                np.linalg.norm(
                    c_coords[:, None, :] - p_coords[None, :, :], axis=-1
                ).argmin(),
                (c_coords.shape[0], p_coords.shape[0]),
            )
            graph[node_to_idx[f"{c.id}_{c_idx}"]].append(
                node_to_idx[f"{p}_{p_idx}"]
            )
            graph[node_to_idx[f"{p}_{p_idx}"]].append(
                node_to_idx[f"{c.id}_{c_idx}"]
            )
    return graph, node_to_idx, node_data


def shortest_path(
    graph: Dict[int, List[int]],
    start: int,
    goal: int,
) -> Optional[List[int]]:
    """
    Find the shortest path between two nodes in the graph.

    Returns
    -------
    path : Optional[List[int]]
        The path as a list of nodes. Will return None if the start and goal are
        not path connected.

    """
    explored = []
    queue = [[start]]
    if start == goal:
        return [start]
    while queue:
        path = queue.pop(0)
        node = path[-1]
        if node not in explored:
            neighbours = graph[node]
            for neighbour in neighbours:
                new_path = path.copy()
                new_path.append(neighbour)
                queue.append(new_path)
                if neighbour == goal:
                    return new_path
            explored.append(node)


def find_route(
    graph: Dict[int, List[int]],
    node_data: Dict[int, Tuple[float, float]],
    start: np.ndarray,
    finish: np.ndarray,
) -> Optional[np.ndarray]:
    """
    Find the shortest path in the walkable area given by the graph.

    First finds the closest nodes to the start and finish then connects
    them with the shortest path in the graph. Will return None if the
    start and finish are not path connected.
    """
    if not node_data:
        return np.array([start] + [finish])
    start_node = min(
        node_data, key=lambda n: np.linalg.norm(np.array(node_data[n]) - start)
    )
    end_node = min(
        node_data,
        key=lambda n: np.linalg.norm(np.array(node_data[n]) - finish),
    )
    route = shortest_path(graph, start_node, end_node)
    if route is None:
        return None
    xy = [list(node_data[n]) for n in route]
    return np.array([start] + xy + [finish])
