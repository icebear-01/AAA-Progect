#!/usr/bin/env python3
"""
Subscribe hybrid_astar_curvature (MarkerArray) and save curvature vs. s as PNG.

Usage:
    rosrun hybrid_aStar save_curvature_png.py --output /tmp/hybrid_curvature.png --timeout 5.0
"""
import argparse
import sys
import rospy
from visualization_msgs.msg import MarkerArray
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


class CurvatureSaver:
    def __init__(self, output_path: str, timeout: float, topic: str):
        self.output_path = output_path
        self.timeout = timeout
        self.received = False
        self.sub = rospy.Subscriber(topic, MarkerArray, self.callback, queue_size=1)
        self.topic = topic

    def callback(self, msg: MarkerArray):
        if self.received:
            return
        data = {}
        for mk in msg.markers:
            if mk.type != mk.LINE_STRIP or not mk.points:
                continue
            # Use namespace to identify raw/smoothed curves.
            key = mk.ns if mk.ns else f"curve_{mk.id}"
            s_vals = [p.x for p in mk.points]
            k_vals = [p.y for p in mk.points]
            data[key] = (s_vals, k_vals)

        if not data:
            rospy.logwarn("No LINE_STRIP markers with points found in MarkerArray.")
            return

        plt.figure(figsize=(8, 4))
        for idx, (name, (s, k)) in enumerate(sorted(data.items())):
            label = name
            plt.plot(s, k, label=label, linewidth=2)

        plt.xlabel("Arc length s (m)")
        plt.ylabel("Curvature kappa (1/m)")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend()
        plt.tight_layout()

        plt.savefig(self.output_path, dpi=200)
        rospy.loginfo("Saved curvature plot to %s", self.output_path)
        self.received = True
        rospy.signal_shutdown("done")

    def spin(self):
        rospy.loginfo("Waiting for MarkerArray on %s ...", self.topic)
        start = rospy.Time.now()
        rate = rospy.Rate(20)
        while not rospy.is_shutdown():
            if self.received:
                return
            if self.timeout > 0 and (rospy.Time.now() - start).to_sec() > self.timeout:
                rospy.logerr("Timeout %.1fs waiting for curvature markers on %s.", self.timeout, self.topic)
                rospy.signal_shutdown("timeout")
                return
            rate.sleep()


def main():
    parser = argparse.ArgumentParser(description="Save hybrid_astar_curvature MarkerArray to PNG.")
    parser.add_argument("--output", default="/tmp/hybrid_astar_curvature.png",
                        help="Path to save the PNG (default: /tmp/hybrid_astar_curvature.png)")
    parser.add_argument("--timeout", type=float, default=5.0,
                        help="Seconds to wait for the topic before quitting (default: 5s, 0 to wait forever)")
    parser.add_argument("--topic", default="hybrid_astar_curvature",
                        help="MarkerArray topic (default: hybrid_astar_curvature)")
    args = parser.parse_args(rospy.myargv(argv=sys.argv)[1:])

    rospy.init_node("save_hybrid_astar_curvature_png", anonymous=True)
    saver = CurvatureSaver(args.output, args.timeout, args.topic)
    saver.spin()


if __name__ == "__main__":
    main()
