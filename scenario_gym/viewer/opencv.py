from typing import List, Optional

import cv2
import numpy as np
from shapely.geometry import LineString, Polygon

from scenario_gym.road_network import RoadNetwork
from scenario_gym.state import State

from .base import Viewer
from .utils import to_ego_frame, vec2pix


class OpenCVViewer(Viewer):
    """
    General display class to visualise the Scenario Gym.

    Rendering behaviour can be customised by changing the layers
    that are rendered. The available layers are given by
    `Viewer._renderable_layers`.
    """

    _renderable_layers = [
        "driveable_surface",
        "walkable_surface",
        "lanes",
        "road_center",
        "lane_center",
        "lane_connecter_center",
        "centers",
    ]

    def __init__(
        self,
        output_path: Optional[str],
        magnification: int = 10,
        fps: float = 30,
        headless_rendering: bool = True,
        render_entity: str = "ego",
        render_layers: Optional[List[str]] = None,
        codec: str = "avc1",
        line_thickness: int = 3,
        width: int = 100,
        height: int = 100,
        window_name: str = "frame",
        **colors,
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

        colors
            Color changes as keyword arguments.

        """
        super().__init__(output_path)
        self.w = width
        self.h = height
        self.mag = magnification
        self.fps = fps
        self.entity_ref = render_entity
        self.window_name = window_name

        self.background_color = (255, 255, 255)
        self.r_center_color = None
        self.l_center_color = None
        self.l_connector_color = None
        self.rn_color = None
        self.lane_color = None
        self.pavement_color = None
        self.crossing_color = None
        self.building_color = None

        if render_layers is None:
            render_layers = ["driveable_surface", "walkable_surface"]

        self.set_colors(render_layers, **colors)
        self.line_thickness = line_thickness

        self.centring_position = np.array([self.w / 2, self.w / 2])
        self.origin = np.array([0.0, 0.0])

        self.base_frame = (
            np.ones(
                [int(self.mag * self.h), int(self.mag * self.w), 3],
                dtype=np.uint8,
            )
            * np.array(self.background_color, dtype=np.uint8)[None, None, :]
        )
        self._frame = self.reset_frame()
        self._state = self._entity_colour_dict = None
        self.headless_rendering = headless_rendering

        # set the video-writer if in headless mode
        self.video_writer = None
        if self.headless_rendering:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            self.video_writer = cv2.VideoWriter(
                output_path,
                fourcc,
                self.fps,
                (self.w * self.mag, self.h * self.mag),
            )

    def set_colors(self, layers: List[str], **colors) -> None:
        """Set the colors used for rendering each layer."""
        if "driveable_surface" in layers:
            self.rn_color = (128, 128, 128)
        if "walkable_surface" in layers:
            self.pavement_color = (200, 200, 200)
            self.crossing_color = (220, 220, 220)
        if "lanes" in layers:
            self.lane_color = (255, 178, 102)
        if "pavement" in layers:
            self.pavement_color = (200, 200, 200)
        if "crossing" in layers:
            self.crossing_color = (220, 220, 220)
        if "building" in layers:
            self.building_color = (10, 10, 10)
        if "centers" in layers:
            self.r_center_color = (255, 178, 102)
            self.l_center_color = (255, 178, 102)
            self.l_connector_color = (255, 178, 102)
        if "road_center" in layers:
            self.r_center_color = (255, 255, 255)  # (255, 178, 102)
        if "lane_center" in layers:
            self.l_center_color = (255, 178, 102)
        if "lane_connecter_center" in layers:
            self.l_connector_color = (255, 178, 102)
        for c, v in colors.items():
            setattr(self, c, v)

    def reset_frame(self) -> np.ndarray:
        """Reset the frame to a black image."""
        return self.base_frame.copy()

    def render(self, state: State) -> Optional[int]:
        """
        Display the state of the gym at a given time.

        This will visualise a top-down view of the gym centrered around the
        ego agent.
        """
        key = None
        self.draw_frame(state, e_ref=self.entity_ref)

        frame = np.flip(self._frame, axis=1)
        if self.headless_rendering:
            self.video_writer.write(frame)
        else:
            cv2.imshow(self.window_name, frame)
            wait_time = 0 if self.fps == 0 else 1000 // self.fps
            key = cv2.waitKey(wait_time)

        # reset the frame to delete current step
        self._frame = self.reset_frame()
        return key

    def draw_geom(self, geom, ego_pose, c) -> None:
        """
        Render a polygon or linestring to the frame.

        Polygons must have no interiors.
        """
        if isinstance(geom, Polygon):
            xy = to_ego_frame(np.array(geom.exterior.xy).T, ego_pose)
            xy = vec2pix(xy, mag=self.mag, h=self.h, w=self.w)
            cv2.fillPoly(self._frame, [xy], c)
            for interior in geom.interiors:
                xy = to_ego_frame(np.array(interior.xy).T, ego_pose)
                xy = vec2pix(xy, mag=self.mag, h=self.h, w=self.w)
                cv2.fillPoly(self._frame, [xy], self.background_color)

        elif isinstance(geom, LineString):
            xy = to_ego_frame(np.array(geom.xy).T, ego_pose)
            xy = vec2pix(xy, mag=self.mag, h=self.h, w=self.w)
            cv2.polylines(self._frame, [xy], False, c, self.line_thickness)

    def draw_road_network_layers(
        self, road_network: RoadNetwork, ego_pose: np.ndarray
    ):
        """Render the different road network layers."""
        if self.rn_color is not None:
            self.draw_driveable_surface(road_network, ego_pose)

        if self.pavement_color or self.crossing_color:
            for p in road_network.pavements:
                self.draw_geom(p.boundary, ego_pose, self.pavement_color)
            for c in road_network.crossings:
                self.draw_geom(c.boundary, ego_pose, self.crossing_color)

        if self.building_color is not None:
            for b in road_network.buildings:
                self.draw_geom(b.boundary, ego_pose, self.building_color)

        if self.r_center_color or self.l_center_color or self.lane_color:
            for r in road_network.roads:
                if self.lane_color or self.l_center_color:
                    for l in r.lanes:
                        if self.lane_color:
                            self.draw_geom(l.boundary, ego_pose, self.lane_color)
                        if self.l_center_color:
                            self.draw_geom(l.center, ego_pose, self.l_center_color)
                if self.r_center_color:
                    self.draw_geom(r.center, ego_pose, self.r_center_color)
        if self.l_connector_color:
            for i in road_network.intersections:
                for l in i.lanes:
                    self.draw_geom(l.center, ego_pose, self.l_connector_color)

    def draw_driveable_surface(
        self, road_network: RoadNetwork, ego_pose: np.ndarray
    ):
        """Render driveable surface."""
        for geom in road_network.driveable_surface.geoms:
            # draw exterior
            xy = to_ego_frame(np.array(geom.exterior.xy).T, ego_pose)
            xy = vec2pix(xy, mag=self.mag, h=self.h, w=self.w)
            cv2.fillPoly(self._frame, [xy], self.rn_color)

            # remove interiors
            for interior in geom.interiors:
                xy = to_ego_frame(np.array(interior.xy).T, ego_pose)
                xy = vec2pix(xy, mag=self.mag, h=self.h, w=self.w)
                cv2.fillPoly(self._frame, [xy], self.background_color)

    def draw_frame(self, state: State, e_ref: str = "ego") -> None:
        """Render the given state around a given entity (by reference)."""
        if e_ref:
            if e_ref in state.scenario.agents:
                ego_pose = state.scenario.agents[e_ref].entity.pose
            else:
                ego_pose = state.scenario.entities[0].pose
        else:
            ego_pose = np.zeros(6)
        self.draw_road_network_layers(state.scenario.road_network, ego_pose)

        if self._entity_colour_dict is None:
            self._entity_colour_dict = {
                i: (0, 0, 0)
                if i == 0
                else tuple(map(int, np.random.randint(256, size=3)))
                for i in range(len(state.scenario.entities))
            }

        for i, entity in enumerate(state.scenario.entities):
            c = self._entity_colour_dict[i]

            # get the bounding box in the global frame
            bbox_coords = entity.get_bounding_box_points()

            # convert to the egos frame (rotated 90)
            bbox_in_ego = to_ego_frame(bbox_coords, ego_pose, vertical=True)

            # add to frame
            xy = vec2pix(bbox_in_ego, mag=self.mag, h=self.h, w=self.w)
            # cv2.polylines(self._frame, [xy], 1, c, 5)
            cv2.fillPoly(self._frame, [xy], c)

    def close(self):
        """Clean up any open files for the rendering."""
        cv2.destroyAllWindows()
        if self.video_writer:
            self.video_writer.release()
