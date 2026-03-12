#!/usr/bin/env python3

import math
from pathlib import Path
from typing import Tuple

import numpy as np
import rospy
from geometry_msgs.msg import Point
from nav_msgs.msg import OccupancyGrid
from planning_msgs.msg import car_info
from visualization_msgs.msg import Marker, MarkerArray


class FrontendStreetMapEnvironment:
    def __init__(self) -> None:
        self.world_frame = rospy.get_param("~frontend_street_map/world_frame", "velodyne")
        self.map_topic = rospy.get_param("~frontend_street_map/map_topic", "/guided_frontend_street_map")
        self.marker_topic = rospy.get_param(
            "~frontend_street_map/marker_topic", "/guided_frontend_street_map/markers"
        )
        self.car_pose_topic = rospy.get_param("~frontend_street_map/car_pose_topic", "/car_pos")
        self.dataset_path = Path(
            rospy.get_param(
                "~frontend_street_map/dataset_path",
                str(
                    Path(__file__).resolve().parents[1]
                    / "model_base_astar"
                    / "neural-astar"
                    / "planning-datasets"
                    / "data"
                    / "street"
                    / "mixed_064_moore_c16.npz"
                ),
            )
        )
        self.split = rospy.get_param("~frontend_street_map/split", "train")
        self.map_index = int(rospy.get_param("~frontend_street_map/map_index", 0))
        self.random_index = bool(rospy.get_param("~frontend_street_map/random_index", False))
        self.resolution = float(rospy.get_param("~frontend_street_map/resolution", 0.25))
        self.auto_refresh = bool(rospy.get_param("~frontend_street_map/auto_refresh", False))
        self.refresh_period = float(rospy.get_param("~frontend_street_map/refresh_period", 5.0))
        self.publish_default_start = bool(rospy.get_param("~frontend_street_map/publish_default_start", True))
        self.default_start_yaw = float(rospy.get_param("~frontend_street_map/default_start_yaw", 0.0))
        self.seed = int(rospy.get_param("~frontend_street_map/seed", 123))

        self.rng = np.random.default_rng(self.seed)
        self.maps = self._load_map_bank(self.dataset_path, self.split)
        self.map_pub = rospy.Publisher(self.map_topic, OccupancyGrid, queue_size=1, latch=True)
        self.marker_pub = rospy.Publisher(self.marker_topic, MarkerArray, queue_size=1, latch=True)
        self.car_pose_pub = rospy.Publisher(self.car_pose_topic, car_info, queue_size=1, latch=True)

        self.current_map_index = 0
        self.publish_map()
        if self.auto_refresh:
            rospy.Timer(rospy.Duration(self.refresh_period), self._timer_callback)

    def _load_map_bank(self, dataset_path: Path, split: str) -> np.ndarray:
        if not dataset_path.exists():
            raise FileNotFoundError(f"street dataset not found: {dataset_path}")
        data = np.load(str(dataset_path))
        split_key = {"train": "arr_0", "valid": "arr_4", "test": "arr_8"}[split]
        if split_key not in data.files:
            raise KeyError(f"{split_key} not found in {dataset_path}")
        maps = np.asarray(data[split_key], dtype=np.float32)
        if maps.ndim != 3:
            raise ValueError(f"{split_key} must be [N,H,W], got {maps.shape}")
        # planning-datasets arr_* street maps use 1=free, 0=obstacle.
        return 1.0 - maps

    def _pick_map_index(self) -> int:
        if self.random_index:
            return int(self.rng.integers(0, self.maps.shape[0]))
        return max(0, min(int(self.map_index), int(self.maps.shape[0] - 1)))

    def _origin(self, width: int, height: int) -> Tuple[float, float]:
        return -0.5 * width * self.resolution, -0.5 * height * self.resolution

    def _cell_center(self, x: int, y: int, origin_x: float, origin_y: float, z: float = 0.0) -> Point:
        p = Point()
        p.x = origin_x + (x + 0.5) * self.resolution
        p.y = origin_y + (y + 0.5) * self.resolution
        p.z = z
        return p

    def _publish_default_start(self, occ_map: np.ndarray, origin_x: float, origin_y: float) -> None:
        if not self.publish_default_start:
            return
        free_y, free_x = np.where(occ_map < 0.5)
        if free_x.size == 0:
            rospy.logwarn("street map has no free cells for default start pose")
            return
        pick = int(self.rng.integers(0, free_x.size))
        x = int(free_x[pick])
        y = int(free_y[pick])
        pt = self._cell_center(x, y, origin_x, origin_y)

        msg = car_info()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = self.world_frame
        msg.x = pt.x
        msg.y = pt.y
        msg.yaw = self.default_start_yaw
        msg.speedDrietion = 1
        msg.speed = 0.0
        msg.turnAngle = 0.0
        msg.yawrate = 0.0
        self.car_pose_pub.publish(msg)

    def publish_map(self) -> None:
        self.current_map_index = self._pick_map_index()
        occ_map = self.maps[self.current_map_index]
        height, width = occ_map.shape
        origin_x, origin_y = self._origin(width, height)

        grid = OccupancyGrid()
        grid.header.stamp = rospy.Time.now()
        grid.header.frame_id = self.world_frame
        grid.info.map_load_time = grid.header.stamp
        grid.info.resolution = self.resolution
        grid.info.width = width
        grid.info.height = height
        grid.info.origin.position.x = origin_x
        grid.info.origin.position.y = origin_y
        grid.info.origin.orientation.w = 1.0
        grid.data = [100 if v >= 0.5 else 0 for v in occ_map.reshape(-1)]
        self.map_pub.publish(grid)

        markers = MarkerArray()

        obstacle_marker = Marker()
        obstacle_marker.header.frame_id = self.world_frame
        obstacle_marker.header.stamp = grid.header.stamp
        obstacle_marker.ns = "frontend_street_map"
        obstacle_marker.id = 0
        obstacle_marker.type = Marker.CUBE_LIST
        obstacle_marker.action = Marker.ADD
        obstacle_marker.pose.orientation.w = 1.0
        obstacle_marker.scale.x = self.resolution
        obstacle_marker.scale.y = self.resolution
        obstacle_marker.scale.z = 0.05
        obstacle_marker.color.r = 0.62
        obstacle_marker.color.g = 0.62
        obstacle_marker.color.b = 0.62
        obstacle_marker.color.a = 0.95
        for y in range(height):
            for x in range(width):
                if occ_map[y, x] >= 0.5:
                    obstacle_marker.points.append(self._cell_center(x, y, origin_x, origin_y))
        markers.markers.append(obstacle_marker)

        boundary = Marker()
        boundary.header.frame_id = self.world_frame
        boundary.header.stamp = grid.header.stamp
        boundary.ns = "frontend_street_map"
        boundary.id = 1
        boundary.type = Marker.LINE_STRIP
        boundary.action = Marker.ADD
        boundary.pose.orientation.w = 1.0
        boundary.scale.x = 0.08
        boundary.color.r = 0.35
        boundary.color.g = 0.35
        boundary.color.b = 0.35
        boundary.color.a = 1.0
        boundary.points = [
            self._cell_center(0, 0, origin_x, origin_y),
            self._cell_center(width - 1, 0, origin_x, origin_y),
            self._cell_center(width - 1, height - 1, origin_x, origin_y),
            self._cell_center(0, height - 1, origin_x, origin_y),
            self._cell_center(0, 0, origin_x, origin_y),
        ]
        markers.markers.append(boundary)
        self.marker_pub.publish(markers)

        self._publish_default_start(occ_map, origin_x, origin_y)
        rospy.loginfo(
            "Published street map split=%s index=%d size=%dx%d resolution=%.3f",
            self.split,
            self.current_map_index,
            width,
            height,
            self.resolution,
        )

    def _timer_callback(self, _event) -> None:
        self.publish_map()


def main() -> None:
    rospy.init_node("frontend_street_map_env")
    FrontendStreetMapEnvironment()
    rospy.spin()


if __name__ == "__main__":
    main()
