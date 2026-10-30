#!/usr/bin/env python3
import os
import math
import time
import cv2
import numpy as np
import open3d as o3d
import mediapipe as mp
from ament_index_python.packages import get_package_share_directory

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo, JointState
from geometry_msgs.msg import PointStamped
from cv_bridge import CvBridge


# 1. Human Hand 21-node MediaPipe skeleton connections
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),         # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),         # Index
    (5, 9), (9, 10), (10, 11), (11, 12),    # Middle
    (9, 13), (13, 14), (14, 15), (15, 16),  # Ring
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20) # Pinky & Palm
]

# 2. Object Bone Nodes defined in STL coordinates (in meters, converted from STL mm)
# Node 0 is the Saddle Grabbing Point!
STL_BONE_NODES_MM = np.array([
    [-34.0,   4.5, 22.0],  # Node 0: SADDLE_GRASP (Waist / Neck)
    [ 40.6,   1.5, 22.0],  # Node 1: CLAMP_TIP (Front Jaw)
    [-103.5,  9.0, 22.0],  # Node 2: HANDLE_BASE (Back T-bar center)
    [-91.0,  16.8, 22.0],  # Node 3: T_WING_TOP (Top T-handle grip)
    [-91.0,   2.5, 22.0],  # Node 4: T_WING_BOT (Bottom T-handle grip)
], dtype=np.float64) / 1000.0  # Convert mm to meters

# Connections between STL bone nodes to draw the object skeleton
OBJECT_BONE_CONNECTIONS = [
    (1, 0),  # Clamp Tip -> Saddle Node
    (0, 2),  # Saddle Node -> Handle Base
    (2, 3),  # Handle Base -> Top T-Wing
    (2, 4),  # Handle Base -> Bottom T-Wing
]


class O6RealSenseTeleopNode(Node):
    def __init__(self):
        super().__init__('o6_realsense_teleop_node')

        # ---------------- Parameters ----------------
        self.declare_parameter('camera_topic', '/camera/camera/color/image_raw')
        self.declare_parameter('depth_topic', '/camera/camera/aligned_depth_to_color/image_raw')
        self.declare_parameter('camera_info_topic', '/camera/camera/aligned_depth_to_color/camera_info')
        self.declare_parameter('stl_filename', 'holder.stl')
        self.declare_parameter('hand_side', 'right')
        self.declare_parameter('sdk_value_mode', 'raw_255')
        self.declare_parameter('joint_order', 'standard_o6')
        self.declare_parameter('show_cv_window', True)
        self.declare_parameter('smoothing_alpha', 0.35)
        
        # Grabbing Position Target Zone (in Camera Frame meters & mm)
        self.declare_parameter('auto_grasp_on_proximity', True)
        self.declare_parameter('grasp_distance_mm', 120.0) # Trigger when saddle depth <= 120mm
        self.declare_parameter('grasp_xy_tolerance_px', 65) # Saddle must be within 65px of grab crosshair

        camera_topic = self.get_parameter('camera_topic').get_parameter_value().string_value
        depth_topic = self.get_parameter('depth_topic').get_parameter_value().string_value
        info_topic = self.get_parameter('camera_info_topic').get_parameter_value().string_value
        stl_fname = self.get_parameter('stl_filename').get_parameter_value().string_value
        self.hand_side = self.get_parameter('hand_side').get_parameter_value().string_value.lower()
        self.sdk_mode = self.get_parameter('sdk_value_mode').get_parameter_value().string_value
        self.joint_order = self.get_parameter('joint_order').get_parameter_value().string_value
        self.show_window = self.get_parameter('show_cv_window').get_parameter_value().bool_value
        self.alpha = self.get_parameter('smoothing_alpha').get_parameter_value().double_value
        
        self.auto_grasp = self.get_parameter('auto_grasp_on_proximity').get_parameter_value().bool_value
        self.grasp_dist_mm = self.get_parameter('grasp_distance_mm').get_parameter_value().double_value
        self.grasp_tol_px = self.get_parameter('grasp_xy_tolerance_px').get_parameter_value().integer_value

        cmd_topic = f'/cb_{self.hand_side}_hand_control_cmd'

        # ---------------- ROS 2 Pub/Sub ----------------
        self.bridge = CvBridge()
        self.image_sub = self.create_subscription(Image, camera_topic, self.image_callback, 10)
        self.depth_sub = self.create_subscription(Image, depth_topic, self.depth_callback, 10)
        self.info_sub = self.create_subscription(CameraInfo, info_topic, self.info_callback, 10)
        
        self.joint_pub = self.create_publisher(JointState, cmd_topic, 10)
        self.saddle_pub = self.create_publisher(PointStamped, '~/saddle_target_point', 10)
        self.debug_img_pub = self.create_publisher(Image, '~/skeleton_overlay', 10)

        self.latest_depth_frame = None
        self.intrinsics = None

        # ---------------- Load STL & Prepare Point Cloud ----------------
        pkg_share = get_package_share_directory('o6_realsense_teleop')
        stl_path = os.path.join(pkg_share, 'resource', stl_fname)
        self.stl_pcd = None
        self.stl_center = np.mean(STL_BONE_NODES_MM, axis=0)

        if os.path.exists(stl_path):
            mesh = o3d.io.read_triangle_mesh(stl_path)
            mesh.scale(0.001, center=(0, 0, 0)) # Convert STL from mm to meters
            self.stl_pcd = mesh.sample_points_uniformly(number_of_points=600)
            self.stl_pcd.estimate_normals()
            self.get_logger().info(f'Loaded STL mesh for Bone Node tracking: {stl_path}')
        else:
            self.get_logger().warn(
                f'STL file not found at {stl_path}! Using PCA geometric bone estimation instead.'
            )

        # ---------------- MediaPipe Tasks API Setup ----------------
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

        # Target finger curl ratios when grabbing the saddle
        if self.joint_order == 'base_first':
            self.grasp_target_ratios = np.array([0.85, 0.90, 0.90, 0.90, 0.90, 0.75], dtype=np.float32)
        else:
            self.grasp_target_ratios = np.array([0.85, 0.75, 0.90, 0.90, 0.90, 0.90], dtype=np.float32)

    def depth_callback(self, msg: Image):
        try:
            self.latest_depth_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        except Exception as e:
            self.get_logger().error(f'Depth cv_bridge exception: {e}')

    def info_callback(self, msg: CameraInfo):
        if self.intrinsics is None:
            self.intrinsics = (msg.k[0], msg.k[4], msg.k[2], msg.k[5])
            self.get_logger().info('Camera intrinsics locked.')

    @staticmethod
    def _dist(p1, p2):
        return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2 + (p1.z - p2.z) ** 2)

    @staticmethod
    def _remap_clamp(val, in_min, in_max):
        norm = (val - in_min) / (in_max - in_min + 1e-6)
        return float(np.clip(norm, 0.0, 1.0))

    def get_joint_names(self):
        if self.joint_order == 'base_first':
            return ['thumb_flex', 'index', 'middle', 'ring', 'pinky', 'thumb_abd']
        return ['thumb_flex', 'thumb_abd', 'index', 'middle', 'ring', 'pinky']

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
            raw_ratios = np.array([thumb_flex, index_curl, middle_curl, ring_curl, pinky_curl, thumb_abd], dtype=np.float32)
        else:
            raw_ratios = np.array([thumb_flex, thumb_abd, index_curl, middle_curl, ring_curl, pinky_curl], dtype=np.float32)

        self.smoothed_ratios = self.alpha * raw_ratios + (1.0 - self.alpha) * self.smoothed_ratios
        return self.smoothed_ratios, self.get_joint_names()

    def format_sdk_positions(self, ratios):
        if self.sdk_mode == 'raw_255':
            return [float(round((1.0 - r) * 255.0, 1)) for r in ratios]
        return [float(round(r, 3)) for r in ratios]

    def detect_stl_bone_nodes(self, frame, header):
        """
        Extracts 3D point cloud of the object in the workspace, aligns the STL bone nodes
        via PCA + ICP, and projects the 5 object bone nodes onto the 2D image.
        """
        if self.latest_depth_frame is None or self.intrinsics is None:
            return None

        dh, dw = self.latest_depth_frame.shape[:2]
        fx, fy, cx, cy = self.intrinsics

        # 1. Crop a central detection workspace (220x220 px)
        roi_r = 110
        u1, u2 = max(0, dw // 2 - roi_r), min(dw, dw // 2 + roi_r)
        v1, v2 = max(0, dh // 2 - roi_r), min(dh, dh // 2 + roi_r)

        roi_depth = self.latest_depth_frame[v1:v2, u1:u2]
        valid_mask = (roi_depth > 30) & (roi_depth < 400) # Detect objects 3cm to 40cm away

        if np.count_nonzero(valid_mask) < 200:
            return None

        # Isolate foreground object (closest 35mm of depth in ROI)
        min_d = np.percentile(roi_depth[valid_mask], 10)
        obj_mask = valid_mask & (roi_depth <= min_d + 35)

        vs, us = np.where(obj_mask)
        if len(us) < 150:
            return None

        # Downsample pixels for fast real-time 3D processing
        step = max(1, len(us) // 500)
        us_full = us[::step] + u1
        vs_full = vs[::step] + v1
        zs_m = roi_depth[vs[::step], us[::step]].astype(np.float64) / 1000.0
        xs_m = (us_full - cx) * zs_m / fx
        ys_m = (vs_full - cy) * zs_m / fy
        obs_pts = np.stack([xs_m, ys_m, zs_m], axis=-1)

        # 2. Align STL Bone Skeleton to Observed 3D Point Cloud
        obs_center = np.mean(obs_pts, axis=0)
        centered_obs = obs_pts - obs_center

        # Compute principal axes (PCA) of the observed clamp
        cov = np.cov(centered_obs, rowvar=False)
        eigvals, eigvecs = np.linalg.eigh(cov)
        order = np.argsort(eigvals)[::-1]
        R_pca = eigvecs[:, order] # Columns: [Primary X-axis, Secondary Y-axis, Z-axis]
        if np.linalg.det(R_pca) < 0:
            R_pca[:, 2] *= -1

        # Build initial 4x4 transformation matrix from STL frame to Camera frame
        T_init = np.eye(4)
        T_init[:3, :3] = R_pca
        T_init[:3, 3] = obs_center - R_pca @ self.stl_center

        # Refine with Open3D ICP if STL mesh is loaded
        if self.stl_pcd is not None:
            obs_pcd = o3d.geometry.PointCloud()
            obs_pcd.points = o3d.utility.Vector3dVector(obs_pts)
            reg = o3d.pipelines.registration.registration_icp(
                self.stl_pcd, obs_pcd, 0.02, T_init,
                o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=15)
            )
            T_final = reg.transformation
        else:
            T_final = T_init

        # 3. Transform the 5 STL Bone Nodes into 3D Camera Coordinates
        ones = np.ones((STL_BONE_NODES_MM.shape[0], 1))
        nodes_homo = np.hstack([STL_BONE_NODES_MM, ones])
        transformed_nodes_3d = (T_final @ nodes_homo.T).T[:, :3]

        # 4. Project 3D Bone Nodes onto 2D RGB Image Pixels
        projected_px = []
        for pt in transformed_nodes_3d:
            px_u = int((pt[0] * fx / pt[2]) + cx)
            px_v = int((pt[1] * fy / pt[2]) + cy)
            projected_px.append((px_u, px_v))

        # Node 0 is the Saddle Grabbing Point
        saddle_3d = transformed_nodes_3d[0]
        saddle_px = projected_px[0]
        saddle_depth_mm = float(saddle_3d[2] * 1000.0)

        # Publish 3D Saddle Point for Robotic Arm / Controller
        pt_msg = PointStamped()
        pt_msg.header = header
        pt_msg.point.x = float(saddle_3d[0])
        pt_msg.point.y = float(saddle_3d[1])
        pt_msg.point.z = float(saddle_3d[2])
        self.saddle_pub.publish(pt_msg)

        return {
            'nodes_2d': projected_px,
            'nodes_3d': transformed_nodes_3d,
            'saddle_px': saddle_px,
            'saddle_3d': saddle_3d,
            'depth_mm': saddle_depth_mm
        }

    def draw_hud(self, frame, landmarks, names, positions, ratios, obj_skeleton, auto_grasping):
        h, w, _ = frame.shape
        cx_goal, cy_goal = w // 2, h // 2

        # Draw Grabbing Position Target Crosshair Zone
        zone_color = (0, 255, 0) if auto_grasping else (255, 200, 0)
        cv2.circle(frame, (cx_goal, cy_goal), self.grasp_tol_px, zone_color, 1, cv2.LINE_AA)
        cv2.drawMarker(frame, (cx_goal, cy_goal), zone_color, cv2.MARKER_CROSS, 16, 1)
        cv2.putText(frame, "GRAB TARGET ZONE", (cx_goal - 60, cy_goal - self.grasp_tol_px - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, zone_color, 1, cv2.LINE_AA)

        # 1. Draw Human Hand Skeleton (MediaPipe)
        if landmarks:
            pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
            for start_idx, end_idx in HAND_CONNECTIONS:
                cv2.line(frame, pts[start_idx], pts[end_idx], (0, 220, 110), 2, cv2.LINE_AA)
            for idx, (px, py) in enumerate(pts):
                cv2.circle(frame, (px, py), 5, (0, 255, 255) if idx in [0,4,8,12,16,20] else (255, 120, 0), -1)

        # 2. Draw STL Object Bone Nodes & Skeleton
        if obj_skeleton is not None:
            nodes_2d = obj_skeleton['nodes_2d']
            # Draw Object Bones
            for s_idx, e_idx in OBJECT_BONE_CONNECTIONS:
                cv2.line(frame, nodes_2d[s_idx], nodes_2d[e_idx], (255, 0, 255), 3, cv2.LINE_AA)

            # Draw Object Nodes (Node 0 = Saddle Grasp Node)
            node_labels = ["0:SADDLE", "1:TIP", "2:BASE", "3:TOP", "4:BOT"]
            for idx, (nx, ny) in enumerate(nodes_2d):
                if idx == 0:
                    # Highlight Node 0 (Saddle) prominently
                    s_col = (0, 255, 0) if auto_grasping else (0, 0, 255)
                    cv2.circle(frame, (nx, ny), 9, s_col, -1)
                    cv2.circle(frame, (nx, ny), 14, (255, 255, 255), 2)
                    cv2.putText(frame, f"SADDLE ({obj_skeleton['depth_mm']:.0f}mm)", (nx + 16, ny + 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, s_col, 2, cv2.LINE_AA)
                else:
                    cv2.circle(frame, (nx, ny), 5, (255, 255, 0), -1)
                    cv2.putText(frame, node_labels[idx], (nx + 6, ny - 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1, cv2.LINE_AA)

        # 3. Telemetry HUD Box
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (315, 275), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

        status_color = (0, 255, 255) if auto_grasping else ((0, 255, 0) if landmarks else (0, 165, 255))
        status_text = 'HOLDING SADDLE (GRASPING)' if auto_grasping else ('TELEOP ACTIVE' if landmarks else 'TRACKING STL BONES')
        cv2.putText(frame, status_text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.52, status_color, 2, cv2.LINE_AA)

        for i, (name, val, r) in enumerate(zip(names, positions, ratios)):
            y = 65 + i * 27
            cv2.putText(frame, f'{name:<10}: {val:>5}', (20, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1, cv2.LINE_AA)
            bar_x, bar_y, bar_w, bar_h = 185, y - 12, 100, 14
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (80, 80, 80), 1)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + int(bar_w * r), bar_y + bar_h), (0, 200, 255), -1)

        if obj_skeleton is not None:
            sx, sy, sz = obj_skeleton['saddle_3d']
            cv2.putText(frame, f"Saddle Node 0: ({sx:+.2f}, {sy:+.2f}, {sz:.2f})m", (20, 255),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 180), 1, cv2.LINE_AA)
        else:
            cv2.putText(frame, "STL Object: Place in Center Crosshair", (20, 255),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (150, 150, 150), 1, cv2.LINE_AA)

    def publish_joints(self, names, positions):
        js_msg = JointState()
        js_msg.header.stamp = self.get_clock().now().to_msg()
        js_msg.name = names
        js_msg.position = positions
        js_msg.velocity = [200.0] * 6 if self.sdk_mode == 'raw_255' else [1.0] * 6
        js_msg.effort = [200.0] * 6 if self.sdk_mode == 'raw_255' else [0.8] * 6
        self.joint_pub.publish(js_msg)

    def image_callback(self, msg: Image):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'cv_bridge exception: {e}')
            return

        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        now_ms = int(time.monotonic() * 1000)
        if now_ms <= self.last_ts_ms:
            now_ms = self.last_ts_ms + 1
        self.last_ts_ms = now_ms

        # 1. Detect Human Hand Nodes
        result = self.landmarker.detect_for_video(mp_image, now_ms)
        names = self.get_joint_names()
        active_landmarks = None

        # 2. Detect STL Object Bone Nodes
        obj_skeleton = self.detect_stl_bone_nodes(frame, msg.header)

        # 3. Check if Saddle Node (Node 0) is inside the Grabbing Position Zone
        auto_grasping = False
        if self.auto_grasp and obj_skeleton is not None:
            sx, sy = obj_skeleton['saddle_px']
            dist_from_center = math.hypot(sx - (w // 2), sy - (h // 2))
            if dist_from_center <= self.grasp_tol_px and obj_skeleton['depth_mm'] <= self.grasp_dist_mm:
                auto_grasping = True

        # 4. Actuate LinkerHand O6
        if auto_grasping:
            self.smoothed_ratios = self.alpha * self.grasp_target_ratios + (1.0 - self.alpha) * self.smoothed_ratios
            ratios = self.smoothed_ratios
            positions = self.format_sdk_positions(ratios)
            self.publish_joints(names, positions)
        elif result.hand_landmarks:
            active_landmarks = result.hand_landmarks[0]
            ratios, names = self.compute_o6_ratios(active_landmarks)
            positions = self.format_sdk_positions(ratios)
            self.publish_joints(names, positions)
        else:
            ratios = self.smoothed_ratios
            positions = self.format_sdk_positions(ratios)

        # 5. Draw Skeletons & Publish
        self.draw_hud(frame, active_landmarks, names, positions, ratios, obj_skeleton, auto_grasping)

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