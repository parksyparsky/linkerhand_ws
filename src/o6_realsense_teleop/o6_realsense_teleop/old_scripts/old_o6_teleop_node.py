#!/usr/bin/env python3
import os
import math
import time
import cv2
import numpy as np
import mediapipe as mp
from ament_index_python.packages import get_package_share_directory

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState
from cv_bridge import CvBridge


# Standard 21-node MediaPipe hand skeleton connections
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index
    (5, 9), (9, 10), (10, 11), (11, 12),   # Middle
    (9, 13), (13, 14), (14, 15), (15, 16), # Ring
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20) # Pinky & Palm
]


class O6RealSenseTeleopNode(Node):
    def __init__(self):
        super().__init__('o6_realsense_teleop_node')

        # ---------------- Parameters ----------------
        self.declare_parameter('camera_topic', '/camera/camera/color/image_raw')
        self.declare_parameter('hand_side', 'right')
        self.declare_parameter('sdk_value_mode', 'raw_255')
        self.declare_parameter('joint_order', 'standard_o6')
        self.declare_parameter('show_cv_window', True)
        self.declare_parameter('smoothing_alpha', 0.35)

        camera_topic = self.get_parameter('camera_topic').get_parameter_value().string_value
        self.hand_side = self.get_parameter('hand_side').get_parameter_value().string_value.lower()
        self.sdk_mode = self.get_parameter('sdk_value_mode').get_parameter_value().string_value
        self.joint_order = self.get_parameter('joint_order').get_parameter_value().string_value
        self.show_window = self.get_parameter('show_cv_window').get_parameter_value().bool_value
        self.alpha = self.get_parameter('smoothing_alpha').get_parameter_value().double_value

        cmd_topic = f'/cb_{self.hand_side}_hand_control_cmd'

        # ---------------- ROS 2 Pub/Sub ----------------
        self.bridge = CvBridge()
        self.image_sub = self.create_subscription(Image, camera_topic, self.image_callback, 10)
        self.joint_pub = self.create_publisher(JointState, cmd_topic, 10)
        self.debug_img_pub = self.create_publisher(Image, '~/skeleton_overlay', 10)

        # ---------------- MediaPipe Tasks API Setup ----------------
        pkg_share = get_package_share_directory('o6_realsense_teleop')
        model_path = os.path.join(pkg_share, 'resource', 'hand_landmarker.task')

        BaseOptions = mp.tasks.BaseOptions
        HandLandmarker = mp.tasks.vision.HandLandmarker
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.60,
            min_hand_presence_confidence=0.60,
            min_tracking_confidence=0.60,
        )
        self.landmarker = HandLandmarker.create_from_options(options)
        self.last_ts_ms = 0

        self.smoothed_ratios = np.zeros(6, dtype=np.float32)

        self.get_logger().info(
            f'O6 Teleop Ready! Subscribed: {camera_topic} -> Publishing: {cmd_topic}'
        )

    @staticmethod
    def _dist(p1, p2):
        return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2 + (p1.z - p2.z) ** 2)

    @staticmethod
    def _remap_clamp(val, in_min, in_max):
        norm = (val - in_min) / (in_max - in_min + 1e-6)
        return float(np.clip(norm, 0.0, 1.0))

    def compute_o6_ratios(self, lm):
        wrist = lm[0]
        palm_scale = self._dist(wrist, lm[9]) + 1e-6

        index_curl = 1.0 - self._remap_clamp(self._dist(wrist, lm[8]) / palm_scale, 0.85, 2.05)
        middle_curl = 1.0 - self._remap_clamp(self._dist(wrist, lm[12]) / palm_scale, 0.85, 2.15)
        ring_curl = 1.0 - self._remap_clamp(self._dist(wrist, lm[16]) / palm_scale, 0.80, 2.00)
        pinky_curl = 1.0 - self._remap_clamp(self._dist(wrist, lm[20]) / palm_scale, 0.75, 1.85)

        thumb_flex = 1.0 - self._remap_clamp(self._dist(lm[4], lm[17]) / palm_scale, 0.55, 1.35)
        thumb_abd = 1.0 - self._remap_clamp(self._dist(lm[4], lm[5]) / palm_scale, 0.35, 1.10)

        if self.joint_order == 'base_first':
            raw_ratios = np.array([thumb_flex, index_curl, middle_curl, ring_curl, pinky_curl, thumb_abd])
            names = ['thumb_flex', 'index', 'middle', 'ring', 'pinky', 'thumb_abd']
        else:
            raw_ratios = np.array([thumb_flex, thumb_abd, index_curl, middle_curl, ring_curl, pinky_curl])
            names = ['thumb_flex', 'thumb_abd', 'index', 'middle', 'ring', 'pinky']

        self.smoothed_ratios = self.alpha * raw_ratios + (1.0 - self.alpha) * self.smoothed_ratios
        return self.smoothed_ratios, names

    def format_sdk_positions(self, ratios):
        if self.sdk_mode == 'raw_255':
            return [float(round((1.0 - r) * 255.0, 1)) for r in ratios]
        return [float(round(r, 3)) for r in ratios]

    def draw_hud(self, frame, landmarks, names, positions, ratios):
        h, w, _ = frame.shape

        # Draw pure OpenCV skeleton bones and 21 nodes
        if landmarks:
            pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
            for start_idx, end_idx in HAND_CONNECTIONS:
                cv2.line(frame, pts[start_idx], pts[end_idx], (0, 220, 110), 2, cv2.LINE_AA)

            for idx, (cx, cy) in enumerate(pts):
                if idx in [0, 4, 8, 12, 16, 20]:
                    cv2.circle(frame, (cx, cy), 6, (0, 255, 255), -1)
                    cv2.putText(frame, str(idx), (cx + 7, cy - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
                else:
                    cv2.circle(frame, (cx, cy), 4, (255, 120, 0), -1)

        # Telemetry HUD box
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (290, 235), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

        status_color = (0, 255, 0) if landmarks else (0, 0, 255)
        status_text = f'O6 ({self.hand_side.upper()}) - TRACKING' if landmarks else 'NO HAND DETECTED'
        cv2.putText(frame, status_text, (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, status_color, 2, cv2.LINE_AA)

        for i, (name, val, r) in enumerate(zip(names, positions, ratios)):
            y = 65 + i * 27
            cv2.putText(frame, f'{name:<10}: {val:>5}', (20, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1, cv2.LINE_AA)
            bar_x, bar_y, bar_w, bar_h = 175, y - 12, 100, 14
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (80, 80, 80), 1)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + int(bar_w * r), bar_y + bar_h), (0, 200, 255), -1)

    def image_callback(self, msg: Image):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'cv_bridge exception: {e}')
            return

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Strictly monotonically increasing timestamp in ms for VIDEO mode
        now_ms = int(time.monotonic() * 1000)
        if now_ms <= self.last_ts_ms:
            now_ms = self.last_ts_ms + 1
        self.last_ts_ms = now_ms

        result = self.landmarker.detect_for_video(mp_image, now_ms)

        names = ['thumb_flex', 'thumb_abd', 'index', 'middle', 'ring', 'pinky']
        positions = [0.0] * 6
        active_landmarks = None

        if result.hand_landmarks:
            active_landmarks = result.hand_landmarks[0]
            ratios, names = self.compute_o6_ratios(active_landmarks)
            positions = self.format_sdk_positions(ratios)

            js_msg = JointState()
            js_msg.header.stamp = self.get_clock().now().to_msg()
            js_msg.name = names
            js_msg.position = positions
            js_msg.velocity = [200.0] * 6 if self.sdk_mode == 'raw_255' else [1.0] * 6
            js_msg.effort = [200.0] * 6 if self.sdk_mode == 'raw_255' else [0.8] * 6
            self.joint_pub.publish(js_msg)
        else:
            ratios = self.smoothed_ratios

        self.draw_hud(frame, active_landmarks, names, positions, ratios)

        out_msg = self.bridge.cv2_imgmsg_from_cv2(frame, encoding='bgr8') if hasattr(self.bridge, 'cv2_imgmsg_from_cv2') else self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        out_msg.header = msg.header
        self.debug_img_pub.publish(out_msg)

        if self.show_window:
            cv2.imshow('RealSense LinkerHand O6 Skeleton View', frame)
            cv2.waitKey(1)

    def destroy_node(self):
        self.landmarker.close()
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = O6RealSenseTeleopNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()