#!/usr/bin/env python3
import os
import math
import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo, JointState
from geometry_msgs.msg import PointStamped
from cv_bridge import CvBridge


# =====================================================================
# 1. CLAMP SKELETON CONNECTIONS (L12_3m_Clothespin_2Finger_T_Handle_V5)
# Nodes: 0=SADDLE, 1=TIP, 2=BASE, 3=T_WING_TOP, 4=T_WING_BOT
# =====================================================================
OBJECT_BONE_CONNECTIONS = [
    (1, 0),  # Clamp Tip -> Saddle Waist
    (0, 2),  # Saddle Waist -> T-Handle Base
    (2, 3),  # T-Handle Base -> Top Wing
    (2, 4),  # T-Handle Base -> Bottom Wing
]


# =====================================================================
# 2. LINKERHAND O6 URDF KINEMATICS (From linkerhand_o6_right.urdf)
# Computes the Robotic Hand Base, Palmar Grasp Center, and 5 Fingertips
# =====================================================================
def _euler_to_rot(r, p, y):
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float64)
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float64)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)
    return Rz @ Ry @ Rx


def _axis_angle_to_rot(axis, angle):
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    x, y, z = axis
    c, s = math.cos(angle), math.sin(angle)
    C = 1.0 - c
    return np.array([
        [x*x*C + c,   x*y*C - z*s, x*z*C + y*s],
        [y*x*C + z*s, y*y*C + c,   y*z*C - x*s],
        [z*x*C - y*s, z*y*C + x*s, z*z*C + c  ]
    ], dtype=np.float64)


def _joint_tf(xyz, rpy, axis=None, angle=0.0):
    T = np.eye(4, dtype=np.float64)
    R = _euler_to_rot(*rpy)
    if axis is not None and abs(angle) > 1e-9:
        R = R @ _axis_angle_to_rot(axis, angle)
    T[:3, :3] = R
    T[:3, 3] = xyz
    return T


def compute_linkerhand_o6_key_nodes(ratios_std):
    """
    Computes the key URDF nodes for LinkerHand O6 in `hand_base_link` coordinates:
      Node 0: GRASP_CENTER (Palmar pocket where the saddle is grabbed)
      Node 1: hand_base_link (Wrist root)
      Node 2..6: thumb_tip, index_tip, middle_tip, ring_tip, pinky_tip
    Uses exact joint limits and mimic multipliers (1.83 thumb, 0.89 fingers) from URDF.
    """
    q_th_p = float(ratios_std[0]) * 0.52
    q_th_y = float(ratios_std[1]) * 1.54
    q_in   = float(ratios_std[2]) * 1.57
    q_mi   = float(ratios_std[3]) * 1.57
    q_ri   = float(ratios_std[4]) * 1.57
    q_pi   = float(ratios_std[5]) * 1.57

    T_base = np.eye(4, dtype=np.float64)
    T_grasp = T_base @ _joint_tf([0.028, 0.008, 0.078], [0, 0, 0])

    # Thumb tip
    T_th = (T_base
            @ _joint_tf([0.011508, 0.022975, 0.032794], [0, 0, 0], [0, 0, -1], q_th_y)
            @ _joint_tf([0.0061649, 0.010678, -0.004891], [0, -1.1529, -2.0944], [0, 1, 0], q_th_p)
            @ _joint_tf([0.0037776, 0, 0.045368], [0, 0, 0], [0, 1, 0], 1.83 * q_th_p)
            @ _joint_tf([-0.002, 0, 0.044], [0, 0, 0]))

    # Index tip
    T_in = (T_base
            @ _joint_tf([0.0024758, 0.02419, 0.098779], [-0.05236, 0, 0], [0, 1, 0], q_in)
            @ _joint_tf([-0.0052516, 0, 0.036625], [0, 0, 0], [0, 1, 0], 0.89 * q_in)
            @ _joint_tf([0.016, 0, 0.035], [0, 0, 0]))

    # Middle tip
    T_mi = (T_base
            @ _joint_tf([0.00052576, 0.00634, 0.1027], [0, 0, 0], [0, 1, 0], q_mi)
            @ _joint_tf([-0.0052516, 0, 0.036625], [0, 0, 0], [0, 1, 0], 0.89 * q_mi)
            @ _joint_tf([0.016, 0, 0.035], [0, 0, 0]))

    # Ring tip
    T_ri = (T_base
            @ _joint_tf([0.0010258, -0.011135, 0.098767], [0.05236, 0, 0], [0, 1, 0], q_ri)
            @ _joint_tf([-0.0052516, 0, 0.036625], [0, 0, 0], [0, 1, 0], 0.89 * q_ri)
            @ _joint_tf([0.016, 0, 0.035], [0, 0, 0]))

    # Pinky tip
    T_pi = (T_base
            @ _joint_tf([0.0024758, -0.028372, 0.092741], [0.087266, 0, 0], [0, 1, 0], q_pi)
            @ _joint_tf([-0.0052516, 0, 0.036625], [0, 0, 0], [0, 1, 0], 0.89 * q_pi)
            @ _joint_tf([0.016, 0, 0.035], [0, 0, 0]))

    tfs = [T_grasp, T_base, T_th, T_in, T_mi, T_ri, T_pi]
    return np.array([T[:3, 3] for T in tfs], dtype=np.float64)


class O6SaddleGraspNode(Node):
    def __init__(self):
        super().__init__('o6_realsense_teleop_node')

        # ---------------- Parameters (100% compatible with launch.py) ----------------
        self.declare_parameter('camera_topic', '/camera/camera/color/image_raw')
        self.declare_parameter('depth_topic', '/camera/camera/aligned_depth_to_color/image_raw')
        self.declare_parameter('camera_info_topic', '/camera/camera/aligned_depth_to_color/camera_info')
        self.declare_parameter('stl_filename', 'L12_3m_clothespin_2finger_t_handle.stl')
        self.declare_parameter('hand_side', 'right')
        self.declare_parameter('sdk_value_mode', 'raw_255')
        self.declare_parameter('joint_order', 'standard_o6')
        self.declare_parameter('show_cv_window', True)
        self.declare_parameter('smoothing_alpha', 0.30)
        self.declare_parameter('target_marker_id', 0)

        # Grasp Distance & Workspace Parameters
        self.declare_parameter('auto_grasp_on_proximity', True)
        self.declare_parameter('grasp_distance_mm', 80.0)    # Close hand when Hand Node <-> Saddle <= 80mm
        self.declare_parameter('release_distance_mm', 115.0) # Open hand when Hand Node <-> Saddle >= 115mm
        self.declare_parameter('max_workspace_depth_mm', 550.0)

        camera_topic = self.get_parameter('camera_topic').get_parameter_value().string_value
        depth_topic = self.get_parameter('depth_topic').get_parameter_value().string_value
        info_topic = self.get_parameter('camera_info_topic').get_parameter_value().string_value
        self.hand_side = self.get_parameter('hand_side').get_parameter_value().string_value.lower()
        self.sdk_mode = self.get_parameter('sdk_value_mode').get_parameter_value().string_value
        self.joint_order = self.get_parameter('joint_order').get_parameter_value().string_value
        self.show_window = self.get_parameter('show_cv_window').get_parameter_value().bool_value
        self.alpha = self.get_parameter('smoothing_alpha').get_parameter_value().double_value

        self.auto_grasp = self.get_parameter('auto_grasp_on_proximity').get_parameter_value().bool_value
        self.grasp_dist_mm = self.get_parameter('grasp_distance_mm').get_parameter_value().double_value
        self.release_dist_mm = self.get_parameter('release_distance_mm').get_parameter_value().double_value
        self.max_depth_mm = self.get_parameter('max_workspace_depth_mm').get_parameter_value().double_value

        cmd_topic = f'/cb_{self.hand_side}_hand_control_cmd'
        state_topic = f'/cb_{self.hand_side}_hand_state'

        # ---------------- ROS 2 Pub/Sub ----------------
        self.bridge = CvBridge()
        self.image_sub = self.create_subscription(Image, camera_topic, self.image_callback, 10)
        self.depth_sub = self.create_subscription(Image, depth_topic, self.depth_callback, 10)
        self.info_sub = self.create_subscription(CameraInfo, info_topic, self.info_callback, 10)
        self.state_sub = self.create_subscription(JointState, state_topic, self.hand_state_callback, 10)

        self.joint_pub = self.create_publisher(JointState, cmd_topic, 10)
        self.saddle_pub = self.create_publisher(PointStamped, '~/saddle_target_point', 10)
        self.debug_img_pub = self.create_publisher(Image, '~/skeleton_overlay', 10)

        self.latest_depth_frame = None
        self.intrinsics = None
        self.is_grasping = False

        # Tracking state with automatic timeout clearing (prevents ghost skeletons!)
        self.last_hand_centroid = None
        self.last_clamp_centroid = None
        self.smoothed_hand_nodes_3d = None
        self.smoothed_clamp_nodes_3d = None
        self.last_clamp_axis = None
        self.hand_missing_frames = 0
        self.clamp_missing_frames = 0

        # Joint ratios in standard_o6 order: [thumb_flex, thumb_abd, index, middle, ring, pinky]
        self.current_std_ratios = np.array([0.05, 0.65, 0.05, 0.05, 0.05, 0.05], dtype=np.float32)
        self.measured_std_ratios = None

        self.open_std_ratios  = np.array([0.05, 0.65, 0.05, 0.05, 0.05, 0.05], dtype=np.float32)
        self.grasp_std_ratios = np.array([0.85, 0.85, 0.90, 0.90, 0.90, 0.90], dtype=np.float32)

        self.get_logger().info(
            f'LinkerHand O6 Node + Cross-Sectional Saddle Detector Ready!\n'
            f'  Grasp Trigger Distance: <= {self.grasp_dist_mm:.1f} mm'
        )

    def hand_state_callback(self, msg: JointState):
        if len(msg.position) >= 6:
            pos = np.array(msg.position[:6], dtype=np.float32)
            r = np.clip(1.0 - (pos / 255.0), 0.0, 1.0) if self.sdk_mode == 'raw_255' else np.clip(pos, 0.0, 1.0)
            if self.joint_order == 'base_first':
                self.measured_std_ratios = np.array([r[0], r[5], r[1], r[2], r[3], r[4]], dtype=np.float32)
            else:
                self.measured_std_ratios = r

    def depth_callback(self, msg: Image):
        try:
            self.latest_depth_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        except Exception as e:
            self.get_logger().error(f'Depth cv_bridge exception: {e}')

    def info_callback(self, msg: CameraInfo):
        if self.intrinsics is None:
            self.intrinsics = (msg.k[0], msg.k[4], msg.k[2], msg.k[5])
            self.get_logger().info('Camera intrinsics locked.')

    def get_joint_names_and_ratios(self, std_ratios):
        if self.joint_order == 'base_first':
            names = ['thumb_flex', 'index', 'middle', 'ring', 'pinky', 'thumb_abd']
            ordered = np.array([std_ratios[0], std_ratios[2], std_ratios[3],
                                std_ratios[4], std_ratios[5], std_ratios[1]], dtype=np.float32)
        else:
            names = ['thumb_flex', 'thumb_abd', 'index', 'middle', 'ring', 'pinky']
            ordered = std_ratios
        return names, ordered

    def format_sdk_positions(self, ordered_ratios):
        if self.sdk_mode == 'raw_255':
            return [float(round((1.0 - r) * 255.0, 1)) for r in ordered_ratios]
        return [float(round(r, 3)) for r in ordered_ratios]

    def project_3d_to_2d(self, pts_3d):
        if self.intrinsics is None:
            return []
        fx, fy, cx, cy = self.intrinsics
        px_list = []
        for pt in pts_3d:
            z = max(float(pt[2]), 0.01)
            u = int((pt[0] * fx / z) + cx)
            v = int((pt[1] * fy / z) + cy)
            px_list.append((u, v))
        return px_list

    def _subsample_and_deproject(self, us, vs, depth_img):
        """
        Subsamples (us, vs) and computes 3D points together so len(pts_3d) == len(u_sub) == len(v_sub)
        is strictly guaranteed (fixes the IndexError crash).
        """
        fx, fy, cx, cy = self.intrinsics
        step = max(1, len(us) // 600)
        u_sub = us[::step]
        v_sub = vs[::step]
        z_m = depth_img[v_sub, u_sub].astype(np.float64) / 1000.0
        x_m = (u_sub - cx) * z_m / fx
        y_m = (v_sub - cy) * z_m / fy
        pts_3d = np.stack([x_m, y_m, z_m], axis=-1)
        return pts_3d, u_sub, v_sub

    def segment_hand_and_clamp(self, frame):
        """
        Separates the foreground depth map into:
          - `hand_pts`: LinkerHand O6 (matte black URDF links / larger palm width)
          - `clamp_pts`: Clothespin T-Handle Clamp (elongated shaft with saddle waist)
        """
        if self.latest_depth_frame is None or self.intrinsics is None:
            return None, None

        depth = self.latest_depth_frame
        dh, dw = depth.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 1. Foreground depth mask (35mm to max_workspace_depth_mm)
        fg_mask = ((depth > 35) & (depth < self.max_depth_mm)).astype(np.uint8) * 255

        # 2. Split along sharp depth jumps (> 20mm) so overlapping objects separate cleanly
        gx = cv2.Sobel(depth.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(depth.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
        depth_edges = (np.hypot(gx, gy) > 70.0).astype(np.uint8) * 255
        fg_mask[depth_edges > 0] = 0

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(fg_mask, connectivity=8)

        blobs = []
        for lbl in range(1, num_labels):
            area = stats[lbl, cv2.CC_STAT_AREA]
            if area < 300:
                continue
            vs, us = np.where(labels == lbl)
            pts_3d, u_sub, v_sub = self._subsample_and_deproject(us, vs, depth)
            if len(pts_3d) < 40:
                continue

            # Voronoi split when the Robotic Hand and Clamp touch during grasping
            if (area > 4000 and self.last_hand_centroid is not None and
                    self.last_clamp_centroid is not None and
                    np.linalg.norm(self.last_hand_centroid - self.last_clamp_centroid) < 0.16):
                d_hand = np.linalg.norm(pts_3d - self.last_hand_centroid, axis=1)
                d_clamp = np.linalg.norm(pts_3d - self.last_clamp_centroid, axis=1)
                mask_h = d_hand < d_clamp
                mask_c = ~mask_h
                if np.count_nonzero(mask_h) > 35 and np.count_nonzero(mask_c) > 35:
                    blobs.append(self._analyze_blob(pts_3d[mask_h], gray, u_sub[mask_h], v_sub[mask_h], dw, dh))
                    blobs.append(self._analyze_blob(pts_3d[mask_c], gray, u_sub[mask_c], v_sub[mask_c], dw, dh))
                    continue

            blobs.append(self._analyze_blob(pts_3d, gray, u_sub, v_sub, dw, dh))

        if not blobs:
            return None, None

        hand_pts, clamp_pts = None, None
        if len(blobs) == 1:
            b = blobs[0]
            # Only classify a single blob as the LinkerHand if it is distinctly matte-black & wide
            if b['is_black_hand'] and b['hand_score'] > b['clamp_score'] + 0.5:
                hand_pts = b['pts']
            else:
                clamp_pts = b['pts']
        else:
            best_pair = None
            best_total = -1e9
            for i in range(len(blobs)):
                for j in range(len(blobs)):
                    if i == j:
                        continue
                    score = blobs[i]['hand_score'] + blobs[j]['clamp_score']
                    if score > best_total:
                        best_total = score
                        best_pair = (blobs[i]['pts'], blobs[j]['pts'])
            if best_pair is not None:
                hand_pts, clamp_pts = best_pair

        return hand_pts, clamp_pts

    def _analyze_blob(self, pts_3d, gray, u_sub, v_sub, dw, dh):
        centroid = np.mean(pts_3d, axis=0)
        centered = pts_3d - centroid
        cov = np.cov(centered, rowvar=False)
        eigvals, _ = np.linalg.eigh(cov)
        stds = np.sqrt(np.maximum(np.sort(eigvals)[::-1], 1e-9))

        elongation = stds[0] / (stds[1] + 1e-4)
        width_m = stds[1]

        mean_intensity = float(np.mean(gray[v_sub, u_sub]))
        touches_border = (np.min(u_sub) < 15 or np.max(u_sub) > dw - 15 or np.max(v_sub) > dh - 15)
        is_black_hand = (mean_intensity < 90.0 and width_m > 0.018)

        hand_score = (width_m * 40.0) - (mean_intensity / 100.0) + (0.5 if touches_border else 0.0)
        if self.last_hand_centroid is not None:
            hand_score -= np.linalg.norm(centroid - self.last_hand_centroid) * 6.0

        clamp_score = (elongation * 0.70) - (width_m * 25.0) + (mean_intensity / 140.0)
        if self.last_clamp_centroid is not None:
            clamp_score -= np.linalg.norm(centroid - self.last_clamp_centroid) * 6.0

        return {
            'pts': pts_3d,
            'centroid': centroid,
            'is_black_hand': is_black_hand,
            'hand_score': hand_score,
            'clamp_score': clamp_score
        }

    def update_robot_hand_node(self, hand_pts):
        """
        Tracks the LinkerHand O6 URDF Grasp Node & Fingertip Frame.
        Automatically clears if the robotic hand leaves the camera view.
        """
        if hand_pts is None or len(hand_pts) < 35:
            self.hand_missing_frames += 1
            if self.hand_missing_frames > 3:
                self.smoothed_hand_nodes_3d = None
                self.last_hand_centroid = None
            if self.smoothed_hand_nodes_3d is None:
                return None
        else:
            self.hand_missing_frames = 0
            obs_center = np.mean(hand_pts, axis=0)
            self.last_hand_centroid = obs_center

            active_std = self.measured_std_ratios if self.measured_std_ratios is not None else self.current_std_ratios
            local_nodes = compute_linkerhand_o6_key_nodes(active_std)
            local_center = np.mean(local_nodes, axis=0)

            centered = hand_pts - obs_center
            cov = np.cov(centered, rowvar=False)
            eigvals, eigvecs = np.linalg.eigh(cov)
            order = np.argsort(eigvals)[::-1]

            # Z-axis points along fingers toward the clamp (or upward if clamp not visible)
            z_axis = eigvecs[:, order[0]]
            if self.last_clamp_centroid is not None:
                if np.dot(z_axis, self.last_clamp_centroid - obs_center) < 0:
                    z_axis = -z_axis
            elif z_axis[1] > 0:
                z_axis = -z_axis

            # Project palm width axis into camera plane to prevent depth axis swapping
            cam_z = np.array([0.0, 0.0, 1.0])
            x_axis = -(cam_z - np.dot(cam_z, z_axis) * z_axis)
            x_norm = np.linalg.norm(x_axis)
            x_axis = x_axis / x_norm if x_norm > 1e-6 else eigvecs[:, order[2]]
            y_axis = np.cross(z_axis, x_axis)
            y_axis /= (np.linalg.norm(y_axis) + 1e-12)

            R_hand = np.column_stack([x_axis, y_axis, z_axis])
            target_nodes_3d = (R_hand @ (local_nodes - local_center).T).T + obs_center

            if self.smoothed_hand_nodes_3d is None:
                self.smoothed_hand_nodes_3d = target_nodes_3d
            else:
                self.smoothed_hand_nodes_3d = 0.4 * target_nodes_3d + 0.6 * self.smoothed_hand_nodes_3d

        nodes_2d = self.project_3d_to_2d(self.smoothed_hand_nodes_3d)
        return {
            'nodes_3d': self.smoothed_hand_nodes_3d,
            'nodes_2d': nodes_2d,
            'grasp_center_3d': self.smoothed_hand_nodes_3d[0],
            'grasp_center_2d': nodes_2d[0] if nodes_2d else None
        }

    def update_clamp_saddle_skeleton(self, clamp_pts, header):
        """
        Directly slices the 3D point cloud of the clamp along its shaft to find:
          Node 0: SADDLE (the narrowest waist cross-section in the middle region)
          Node 1: CLAMP_TIP (narrow front jaw end)
          Node 2: HANDLE_BASE (wide T-handle back end)
          Node 3 & 4: T_WING_TOP & T_WING_BOT (top/bottom ears of the T-handle)
        """
        if clamp_pts is None or len(clamp_pts) < 35:
            self.clamp_missing_frames += 1
            if self.clamp_missing_frames > 3:
                self.smoothed_clamp_nodes_3d = None
                self.last_clamp_centroid = None
                self.last_clamp_axis = None
            if self.smoothed_clamp_nodes_3d is None:
                return None
        else:
            self.clamp_missing_frames = 0
            obs_center = np.mean(clamp_pts, axis=0)
            self.last_clamp_centroid = obs_center
            centered = clamp_pts - obs_center

            # 1. Find primary elongation axis (shaft of the T-handle clamp)
            cov = np.cov(centered, rowvar=False)
            eigvals, eigvecs = np.linalg.eigh(cov)
            e1 = eigvecs[:, np.argmax(eigvals)]

            # Keep sign consistent with previous frame first
            if self.last_clamp_axis is not None and np.dot(e1, self.last_clamp_axis) < 0:
                e1 = -e1

            # Construct in-plane perpendicular width axis e2 (immune to 3D Z-thickness swapping!)
            cam_ray = np.array([0.0, 0.0, 1.0])
            e3 = cam_ray - np.dot(cam_ray, e1) * e1
            e3_norm = np.linalg.norm(e3)
            e3 = e3 / e3_norm if e3_norm > 1e-6 else np.array([0.0, 0.0, 1.0])
            e2 = np.cross(e3, e1)
            e2 /= (np.linalg.norm(e2) + 1e-12)

            # 2. Disambiguate Tip (+e1) vs T-Handle Base (-e1) using end-cap width spread
            s_proj = centered @ e1
            w_proj = centered @ e2
            s_min, s_max = np.percentile(s_proj, 3), np.percentile(s_proj, 97)
            length = max(s_max - s_min, 0.04)

            pos_end_w = w_proj[s_proj >= s_max - 0.22 * length]
            neg_end_w = w_proj[s_proj <= s_min + 0.22 * length]
            if len(pos_end_w) > 8 and len(neg_end_w) > 8:
                span_pos = np.percentile(pos_end_w, 90) - np.percentile(pos_end_w, 10)
                span_neg = np.percentile(neg_end_w, 90) - np.percentile(neg_end_w, 10)
                # T-handle wings at Base are wider than the Tip; if +e1 is wider, flip e1
                if span_pos > span_neg * 1.18:
                    e1 = -e1
                    e2 = -e2
                    s_proj = -s_proj
                    w_proj = -w_proj
                    s_min, s_max = -s_max, -s_min

            self.last_clamp_axis = e1

            # 3. Slice along the clamp's middle region [32% .. 68%] to find the narrowest SADDLE waist
            num_slices = 9
            slice_edges = np.linspace(s_min + 0.30 * length, s_max - 0.30 * length, num_slices + 1)
            best_saddle_3d = None
            min_waist_span = 1e9

            for i in range(num_slices):
                in_slice = (s_proj >= slice_edges[i]) & (s_proj < slice_edges[i + 1])
                if np.count_nonzero(in_slice) >= 6:
                    w_slice = w_proj[in_slice]
                    span = np.percentile(w_slice, 85) - np.percentile(w_slice, 15)
                    if span < min_waist_span:
                        min_waist_span = span
                        best_saddle_3d = np.mean(clamp_pts[in_slice], axis=0)

            # Fallback to STL geometric ratio (48% from base to tip) if slice is sparse
            if best_saddle_3d is None:
                s_saddle = s_min + 0.48 * length
                best_saddle_3d = obs_center + s_saddle * e1

            # 4. Extract Tip, Base, and T-Wing 3D Bone Nodes directly from the point cloud
            tip_mask = s_proj >= s_max - 0.12 * length
            node_tip = np.mean(clamp_pts[tip_mask], axis=0) if np.count_nonzero(tip_mask) >= 4 else (obs_center + s_max * e1)

            base_mask = s_proj <= s_min + 0.15 * length
            if np.count_nonzero(base_mask) >= 6:
                base_pts = clamp_pts[base_mask]
                node_base = np.mean(base_pts, axis=0)
                w_base = w_proj[base_mask]
                w_top = np.percentile(w_base, 92)
                w_bot = np.percentile(w_base, 8)
                node_wing_top = node_base + w_top * e2
                node_wing_bot = node_base + w_bot * e2
            else:
                node_base = obs_center + s_min * e1
                node_wing_top = node_base + 0.015 * e2
                node_wing_bot = node_base - 0.015 * e2

            target_clamp_nodes = np.array([
                best_saddle_3d, # Node 0: SADDLE_GRASP
                node_tip,       # Node 1: CLAMP_TIP
                node_base,      # Node 2: HANDLE_BASE
                node_wing_top,  # Node 3: T_WING_TOP
                node_wing_bot   # Node 4: T_WING_BOT
            ], dtype=np.float64)

            if self.smoothed_clamp_nodes_3d is None:
                self.smoothed_clamp_nodes_3d = target_clamp_nodes
            else:
                self.smoothed_clamp_nodes_3d = 0.45 * target_clamp_nodes + 0.55 * self.smoothed_clamp_nodes_3d

        projected_px = self.project_3d_to_2d(self.smoothed_clamp_nodes_3d)
        saddle_3d = self.smoothed_clamp_nodes_3d[0]

        pt_msg = PointStamped()
        pt_msg.header = header
        pt_msg.point.x = float(saddle_3d[0])
        pt_msg.point.y = float(saddle_3d[1])
        pt_msg.point.z = float(saddle_3d[2])
        self.saddle_pub.publish(pt_msg)

        return {
            'nodes_2d': projected_px,
            'nodes_3d': self.smoothed_clamp_nodes_3d,
            'saddle_2d': projected_px[0],
            'saddle_3d': saddle_3d
        }

    def publish_joints(self, names, positions):
        js_msg = JointState()
        js_msg.header.stamp = self.get_clock().now().to_msg()
        js_msg.name = names
        js_msg.position = positions
        js_msg.velocity = [200.0] * 6 if self.sdk_mode == 'raw_255' else [1.0] * 6
        js_msg.effort = [200.0] * 6 if self.sdk_mode == 'raw_255' else [0.8] * 6
        self.joint_pub.publish(js_msg)

    def draw_hud(self, frame, robot_hand, obj_skeleton, dist_mm, names, positions, ordered_ratios):
        # 1. Draw Robotic Hand Grasp Node & URDF Frame (Cyan)
        if robot_hand is not None and len(robot_hand['nodes_2d']) >= 7:
            r_pts = robot_hand['nodes_2d']
            gx, gy = r_pts[0] # Node 0: Palmar Grasp Center
            bx, by = r_pts[1] # Node 1: hand_base_link

            # Line from Wrist Base to Palmar Grasp Node
            cv2.line(frame, (bx, by), (gx, gy), (255, 200, 0), 2, cv2.LINE_AA)
            cv2.circle(frame, (bx, by), 5, (255, 180, 0), -1)

            # Lines from Palmar Grasp Node to the 5 URDF Fingertip Nodes
            for tip_idx in range(2, 7):
                tx, ty = r_pts[tip_idx]
                cv2.line(frame, (gx, gy), (tx, ty), (255, 220, 0), 1, cv2.LINE_AA)
                cv2.circle(frame, (tx, ty), 4, (0, 255, 120), -1)

            # Highlight Robotic Hand Grasp Node prominently
            cv2.circle(frame, (gx, gy), 9, (0, 255, 255), -1)
            cv2.circle(frame, (gx, gy), 14, (255, 255, 255), 2)
            cv2.putText(frame, "ROBOT_HAND_NODE", (gx + 16, gy - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 255), 2, cv2.LINE_AA)

        # 2. Draw Clamp Saddle Skeleton (Magenta)
        if obj_skeleton is not None and len(obj_skeleton['nodes_2d']) >= 5:
            o_pts = obj_skeleton['nodes_2d']
            for s_idx, e_idx in OBJECT_BONE_CONNECTIONS:
                cv2.line(frame, o_pts[s_idx], o_pts[e_idx], (255, 0, 255), 3, cv2.LINE_AA)

            node_labels = ["0:SADDLE", "1:TIP", "2:BASE", "3:TOP", "4:BOT"]
            for idx, (nx, ny) in enumerate(o_pts):
                if idx == 0:
                    s_col = (0, 255, 0) if self.is_grasping else (0, 0, 255)
                    cv2.circle(frame, (nx, ny), 9, s_col, -1)
                    cv2.circle(frame, (nx, ny), 14, (255, 255, 255), 2)
                    cv2.putText(frame, "0:SADDLE", (nx + 16, ny + 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, s_col, 2, cv2.LINE_AA)
                else:
                    cv2.circle(frame, (nx, ny), 5, (255, 255, 0), -1)
                    cv2.putText(frame, node_labels[idx], (nx + 6, ny - 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 0), 1, cv2.LINE_AA)

        # 3. Draw 3D Distance Vector Between Robotic Hand Node and Saddle Node
        if (robot_hand is not None and obj_skeleton is not None and
                robot_hand['grasp_center_2d'] is not None and dist_mm is not None):
            hx, hy = robot_hand['grasp_center_2d']
            sx, sy = obj_skeleton['saddle_2d']
            line_col = (0, 255, 0) if self.is_grasping else (0, 165, 255)
            cv2.line(frame, (hx, hy), (sx, sy), line_col, 2, cv2.LINE_AA)
            mid_x, mid_y = (hx + sx) // 2, (hy + sy) // 2
            cv2.putText(frame, f"{dist_mm:.1f} mm", (mid_x + 8, mid_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, line_col, 2, cv2.LINE_AA)

        # 4. Telemetry HUD Box
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (345, 280), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

        status_color = (0, 255, 0) if self.is_grasping else (0, 200, 255)
        status_text = 'GRABBING SADDLE!' if self.is_grasping else 'STANDBY (WAITING FOR GRASP RANGE)'
        cv2.putText(frame, status_text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.46, status_color, 2, cv2.LINE_AA)

        for i, (name, val, r) in enumerate(zip(names, positions, ordered_ratios)):
            y = 65 + i * 27
            cv2.putText(frame, f'{name:<10}: {val:>5}', (20, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1, cv2.LINE_AA)
            bar_x, bar_y, bar_w, bar_h = 190, y - 12, 100, 14
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (80, 80, 80), 1)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + int(bar_w * r), bar_y + bar_h), (0, 200, 255), -1)

        if dist_mm is not None:
            cv2.putText(frame, f"Hand<->Saddle Dist: {dist_mm:.1f}mm (Grab <= {self.grasp_dist_mm:.0f}mm)",
                        (20, 258), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 255, 180), 1, cv2.LINE_AA)
        else:
            h_stat = "LOCKED" if robot_hand is not None else "None"
            c_stat = "LOCKED" if obj_skeleton is not None else "None"
            cv2.putText(frame, f"Robotic Hand: {h_stat} | Saddle: {c_stat}",
                        (20, 258), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 180, 255), 1, cv2.LINE_AA)

    def image_callback(self, msg: Image):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'cv_bridge exception: {e}')
            return

        # 1. Segment foreground depth into Robotic Hand vs. Clamp Object
        hand_pts, clamp_pts = self.segment_hand_and_clamp(frame)

        # 2. Update Robotic Hand Node (Clears automatically when hand is absent)
        robot_hand = self.update_robot_hand_node(hand_pts)

        # 3. Update Clamp Saddle Skeleton via Cross-Sectional Slicing
        obj_skeleton = self.update_clamp_saddle_skeleton(clamp_pts, msg.header)

        # 4. Compute 3D Distance & Trigger Grasp State Machine
        dist_mm = None
        if robot_hand is not None and obj_skeleton is not None:
            hand_grasp_3d = robot_hand['grasp_center_3d']
            saddle_3d = obj_skeleton['saddle_3d']
            dist_mm = float(np.linalg.norm(hand_grasp_3d - saddle_3d) * 1000.0)

            if self.auto_grasp:
                if not self.is_grasping and dist_mm <= self.grasp_dist_mm:
                    self.is_grasping = True
                    self.get_logger().info(f'Saddle in grasp range ({dist_mm:.1f} mm)! Closing LinkerHand.')
                elif self.is_grasping and dist_mm >= self.release_dist_mm:
                    self.is_grasping = False
                    self.get_logger().info(f'Saddle out of range ({dist_mm:.1f} mm). Opening LinkerHand.')
        elif obj_skeleton is None:
            self.is_grasping = False

        # 5. Actuate LinkerHand O6
        target_std = self.grasp_std_ratios if self.is_grasping else self.open_std_ratios
        self.current_std_ratios = self.alpha * target_std + (1.0 - self.alpha) * self.current_std_ratios

        names, ordered_ratios = self.get_joint_names_and_ratios(self.current_std_ratios)
        positions = self.format_sdk_positions(ordered_ratios)
        self.publish_joints(names, positions)

        # 6. Render HUD & Publish
        self.draw_hud(frame, robot_hand, obj_skeleton, dist_mm, names, positions, ordered_ratios)

        out_msg = self.bridge.cv2_imgmsg_from_cv2(frame, encoding='bgr8') if hasattr(self.bridge, 'cv2_imgmsg_from_cv2') else self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        out_msg.header = msg.header
        self.debug_img_pub.publish(out_msg)

        if self.show_window:
            cv2.imshow('RealSense LinkerHand O6 Skeleton View', frame)
            cv2.waitKey(1)

    def destroy_node(self):
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = O6SaddleGraspNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()