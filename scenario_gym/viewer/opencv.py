from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from shapely.geometry import LinearRing, LineString, Polygon

from scenario_gym.entity import Entity, Pedestrian, Vehicle
from scenario_gym.state import State

from .base import Viewer
from .utils import to_ego_frame, vec2pix

Color = Tuple[int, int, int]


class OpenCVViewer(Viewer):
    """
    General display class to visualise the Scenario Gym.

    Rendering behaviour can be customised by changing the layers that are rendered.
    The available layers and default colors are given by
    `OpenCVViewer._renderable_layers`.
    """

    _renderable_layers: Dict[str, Optional[Color]] = {
        "driveable_surface": (128, 128, 128),
        "driveable_surface_boundary": (250, 250, 250),
        "walkable_surface": (210, 210, 210),
        "buildings": (128, 128, 128),
        "roads": (128, 128, 128),
        "intersections": (128, 128, 128),
        "lanes": (128, 128, 128),
        "road_centers": (180, 180, 180),
        "lane_centers": (255, 178, 102),
        "text": (250, 250, 250),
    }

    def __init__(
        self,
        magnification: int = 10,
        fps: float = 30,
        headless_rendering: bool = True,
        render_entity: str = "ego",
        render_layers: Optional[List[str]] = None,
        codec: str = "avc1",
        line_thickness: int = 1,
        width: int = 100,
        height: int = 100,
        window_name: str = "frame",
        **colors: Color,
    ):
        """
        Init the viewer.

        Parameters
        ----------
        magnification : int
            The number of pixels per meter.

        fps : float
            The frames per second.

        headless_rendering : bool
            Whether to render to an mp4 file or a popup window.

        output_path : Optional[str]
            The filepath if using headless rendering.

        render_entity : str
            The reference of the entity to use for centering the camera.

        render_layers : List[str]
            The layers to render. A list of all possible layers is given in
            `Viewer._renderable_layers`.

        codec : str
            Codec to encode videos e.g. mp4v or avc1.

        line_thickness : int
            Thickness of lines in pixels.

        width : int
            Width of the camera window in meters.

        height : int
            Height of the camera window in meters.

        window_name : str
            Name for the rendering window in non-headless mode.

        colors : Color
            Color changes as keyword arguments.

        """
        super().__init__()
        self.headless_rendering = headless_rendering
        self.codec = codec
        self.w = width
        self.h = height
        self.mag = magnification
        self.fps = fps
        self.entity_ref = render_entity
        self.window_name = window_name
        self.line_thickness = line_thickness
        self.centring_position = np.array([self.w / 2, self.h / 2])
        self.origin = np.array([0.0, 0.0])
        self._coords_cache = {}

        if render_layers is None:
            render_layers = [
                "driveable_surface_boundary",
                "road_centers",
                "lane_centers",
                "buildings",
            ]

        for layer in render_layers:
            if layer != "entity" and layer != "text":
                try:
                    getattr(self, f"render_{layer}")
                except AttributeError:
                    raise NotImplementedError(
                        f"Rendering method `render_{layer}` is not implemented."
                    )
        self.render_layers = render_layers

        for k, v in {
            "background": (0, 0, 0),
            "entity_front": (250, 250, 250),
            **self._renderable_layers,
            **colors,
        }.items():
            if not all(
                (
                    isinstance(v, tuple),
                    len(v) == 3,
                    all(isinstance(x, int) for x in v),
                    0 <= min(v),
                    255 >= max(v),
                )
            ):
                raise ValueError(
                    f"{v} is not a valid color for {k}. Must be a tuple of 3 ints."
                )
            setattr(self, f"{k}_color", v)

        self.base_frame = (
            np.ones(
                [int(self.mag * self.w), int(self.mag * self.h), 3],
                dtype=np.uint8,
            )
            * np.array(self.background_color, dtype=np.uint8)[None, None, :]
        )
        self._frame = self.reset_frame()
        self._state = self._entity_colour_dict = None
        self.video_writer = None

    def reset_frame(self) -> np.ndarray:
        """Reset the frame to a black image."""
        return self.base_frame.copy()

    def render(self, state: State) -> Optional[int]:
        """
        Display the state of the gym at a given time.

        This will visualise a top-down view of the gym centrered around the
        ego agent.
        """
        self._state = state
        key = None
        self.draw_frame(state, e_ref=self.entity_ref)

        if self.headless_rendering:
            self.video_writer.write(self._frame)
        else:
            cv2.imshow(self.window_name, self._frame)
            wait_time = 0 if self.fps == 0 else 1000 // self.fps
            key = cv2.waitKey(wait_time)

        # reset the frame to delete current step
        self._frame = self.reset_frame()
        return key

    def reset(self, output_path: Optional[str] = None) -> None:
        """Reset at the start of a scenario rollout."""
        super().reset(output_path)
        self._frame = self.reset_frame()
        self._state = self._entity_colour_dict = None
        self.video_writer = None
        self._coords_cache = {}
        if self.headless_rendering:
            fourcc = cv2.VideoWriter_fourcc(*self.codec)
            self.video_writer = cv2.VideoWriter(
                output_path,
                fourcc,
                self.fps,
                (self.w * self.mag, self.h * self.mag),
            )

    def close(self) -> None:
        """Clean up any open files for the rendering."""
        cv2.destroyAllWindows()
        if self.video_writer:
            self.video_writer.release()

    def draw_frame(self, state: State, e_ref: Optional[str] = "ego") -> None:
        """Render the given state around a given entity (by reference)."""
        ego_pose = self.get_center_pose(state, e_ref)

        for layer in self.render_layers:
            if layer != "entity" and layer != "text":
                getattr(self, f"render_{layer}")(state, ego_pose)

        for entity_idx, (entity, pose) in enumerate(state.poses.items()):
            self.draw_entity(state, entity_idx, entity, pose, ego_pose)

        if "text" in self.render_layers:
            self.render_text(state)

    def get_center_pose(self, state: State, e_ref: str) -> np.ndarray:
        """Get the pose for the center of the frame."""
        ego_pose = None
        if e_ref is not None:
            entity = state.scenario.entity_by_name(e_ref)
            if entity is None:
                entity = state.scenario.entities[0]
            ego_pose = state.poses[entity]
        return ego_pose

    @property
    def entity_colors(self) -> Dict[int, Color]:
        """Store the entity color dict."""
        if self._entity_colour_dict is None:
            if self._state is None:
                raise ValueError(
                    "Cannot query the entity colors if state is not set."
                )
            self._entity_colour_dict = {
                i: self.get_entity_color(i, e)
                for i, e in enumerate(self._state.scenario.entities)
            }

        return self._entity_colour_dict

    def get_entity_color(self, entity_idx: int, entity: Entity) -> Color:
        """Get the color to draw the given entity."""
        if entity_idx == 0:
            c = (0, 0, 128)
        elif isinstance(entity, Vehicle):
            c = (128, 0, 0)
        elif isinstance(entity, Pedestrian):
            c = (0, 128, 0)
        else:
            c = (221, 160, 221)
        return c

    def draw_entity(
        self,
        state: State,
        entity_idx: int,
        entity: Entity,
        pose: np.ndarray,
        ego_pose: np.ndarray,
    ) -> None:
        """Draw the given entity onto the frame."""
        # get the bounding box in the global frame
        bbox_coords = entity.get_bounding_box_points(pose)

        # convert to the egos frame (rotated 90)
        bbox_in_ego = to_ego_frame(bbox_coords, ego_pose)

        # render front of the entity
        l = np.linalg.norm(bbox_in_ego[0] - bbox_in_ego[1])
        w = 0.25 / l
        front_bbox = np.array(
            [
                # 0.5 * (bbox_in_ego[2] + bbox_in_ego[1]),
                bbox_in_ego[2],
                bbox_in_ego[1],
                bbox_in_ego[1] + w * (bbox_in_ego[0] - bbox_in_ego[1]),
                bbox_in_ego[2] + w * (bbox_in_ego[3] - bbox_in_ego[2]),
            ]
        )

        # add to frame
        c = self.entity_colors[entity_idx]
        xy = vec2pix(bbox_in_ego, mag=self.mag, h=self.h, w=self.w)
        xy_front = vec2pix(front_bbox, mag=self.mag, h=self.h, w=self.w)
        cv2.fillPoly(self._frame, [xy], c)
        cv2.fillPoly(self._frame, [xy_front], self.entity_front_color)

    def render_text(self, state: State) -> None:
        """Add text to the frame."""
        v = np.linalg.norm(state.velocities[state.scenario.entities[0]][:3])
        cv2.putText(
            self._frame,
            "Ego speed: {:.2f}".format(v),
            (10, int(0.9 * self.mag * self.h)),  # bottom left corner of text
            cv2.FONT_HERSHEY_SIMPLEX,  # font
            1,  # font scale
            self.text_color,  # font color
            1,  # thickness
            2,  # line type
        )

    def draw_geom(
        self,
        geom: Union[Polygon, LineString, LinearRing],
        ego_pose: np.ndarray,
        c: Color,
        use_cache: bool = False,
    ) -> None:
        """Render a polygon or linestring to the frame."""
        if not isinstance(geom, (Polygon, LineString, LinearRing)):
            raise TypeError(f"{type(geom)} not supported.")

        xy = to_ego_frame(self.get_coords(geom, use_cache=use_cache), ego_pose)
        xy = vec2pix(xy, mag=self.mag, h=self.h, w=self.w)
        if isinstance(geom, LineString):
            cv2.polylines(self._frame, [xy], False, c, self.line_thickness)
        else:
            cv2.fillPoly(self._frame, [xy], c)
            for interior in geom.interiors:
                # cannot cache as the id is different each time
                xy = to_ego_frame(np.array(interior.xy).T)
                xy = vec2pix(xy, mag=self.mag, h=self.h, w=self.w)
                cv2.fillPoly(self._frame, [xy], self.background_color)

    def get_coords(
        self,
        geom: Union[Polygon, LineString, LinearRing],
        use_cache: bool = True,
    ) -> np.ndarray:
        """Get the coordinates for the given geometry."""
        if id(geom) in self._coords_cache:
            return self._coords_cache[id(geom)]
        coords = np.array(
            geom.exterior.xy if isinstance(geom, Polygon) else geom.xy
        ).T
        if use_cache:
            self._coords_cache[id(geom)] = coords
        return coords

    def render_driveable_surface(
        self,
        state: State,
        ego_pose: np.ndarray,
    ) -> None:
        """Render the driveable surface."""
        road_network = state.scenario.road_network
        for geom in road_network.driveable_surface.geoms:
            self.draw_geom(
                geom, ego_pose, self.driveable_surface_color, use_cache=False
            )

    def render_driveable_surface_boundary(
        self,
        state: State,
        ego_pose: np.ndarray,
    ) -> None:
        """Render the boundary of the driveable surface."""
        road_network = state.scenario.road_network
        for geom in road_network.driveable_surface.geoms:
            self.draw_geom(
                geom.exterior,
                ego_pose,
                self.driveable_surface_boundary_color,
                use_cache=False,
            )
            for interior in geom.interiors:
                self.draw_geom(
                    interior,
                    ego_pose,
                    self.driveable_surface_boundary_color,
                    use_cache=False,
                )

    def render_walkable_surface(
        self,
        state: State,
        ego_pose: np.ndarray,
    ) -> None:
        """Render the walkable surface."""
        road_network = state.scenario.road_network
        for g in road_network.pavements + road_network.crossings:
            self.draw_geom(g.boundary, ego_pose, self.walkable_surface_color)

    def render_buildings(
        self,
        state: State,
        ego_pose: np.ndarray,
    ) -> None:
        """Render any buildings."""
        road_network = state.scenario.road_network
        for b in road_network.buildings:
            self.draw_geom(b.boundary, ego_pose, self.buildings_color)

    def render_roads(
        self,
        state: State,
        ego_pose: np.ndarray,
    ) -> None:
        """Render the road boundary polygons."""
        road_network = state.scenario.road_network
        for r in road_network.roads:
            self.draw_geom(r.boundary, ego_pose, self.roads_color)

    def render_road_centers(
        self,
        state: State,
        ego_pose: np.ndarray,
    ) -> None:
        """Render the road centers."""
        road_network = state.scenario.road_network
        for r in road_network.roads:
            self.draw_geom(r.center, ego_pose, self.road_centers_color)

    def render_lanes(
        self,
        state: State,
        ego_pose: np.ndarray,
    ) -> None:
        """Render the lane boudnary polygons."""
        road_network = state.scenario.road_network
        for l in road_network.lanes:
            self.draw_geom(l.boundary, ego_pose, self.lanes_color)

    def render_lane_centers(
        self,
        state: State,
        ego_pose: np.ndarray,
    ) -> None:
        """Render the lane centers."""
        road_network = state.scenario.road_network
        for l in road_network.lanes:
            self.draw_geom(l.center, ego_pose, self.lane_centers_color)

    def render_intersections(
        self,
        state: State,
        ego_pose: np.ndarray,
    ) -> None:
        """Render the road boundary polygons."""
        road_network = state.scenario.road_network
        for i in road_network.intersections:
            self.draw_geom(i.boundary, ego_pose, self.intersections_color)
