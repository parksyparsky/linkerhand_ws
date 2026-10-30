import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    launch_realsense = LaunchConfiguration('launch_realsense')
    launch_sdk = LaunchConfiguration('launch_sdk')
    hand_side = LaunchConfiguration('hand_side')
    sdk_value_mode = LaunchConfiguration('sdk_value_mode')
    camera_topic = LaunchConfiguration('camera_topic')
    
    # New Auto-Grasp Configurations
    depth_topic = LaunchConfiguration('depth_topic')
    camera_info_topic = LaunchConfiguration('camera_info_topic')
    auto_grasp = LaunchConfiguration('auto_grasp_on_proximity')
    grasp_dist = LaunchConfiguration('grasp_distance_mm')
    marker_id = LaunchConfiguration('target_marker_id')

    # 1. RealSense Camera Driver
    realsense_group = GroupAction(
        scoped=True,
        forwarding=False,
        condition=IfCondition(launch_realsense),
        actions=[
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    os.path.join(
                        get_package_share_directory('realsense2_camera'),
                        'launch',
                        'rs_launch.py'
                    )
                ),
                launch_arguments={
                    'rgb_camera.color_profile': '640x480x30',
                    'depth_module.depth_profile': '640x480x15',
                    'enable_depth': 'true',
                    'align_depth.enable': 'true',
                }.items(),
            )
        ]
    )

    # 2. LinkerHand ROS 2 SDK Hardware Driver (Talks to physical O6 over can0)
    linkerhand_sdk_group = GroupAction(
        scoped=True,
        forwarding=False,
        condition=IfCondition(launch_sdk),
        actions=[
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    os.path.join(
                        get_package_share_directory('linker_hand_ros2_sdk'),
                        'launch',
                        'linker_hand.launch.py'
                    )
                )
            )
        ]
    )

    # 3. O6 Skeleton Tracking & Teleop Node
    teleop_node = Node(
        package='o6_realsense_teleop',
        executable='o6_teleop_node', # Ensure this matches your setup.py entry point
        name='o6_realsense_teleop_node',
        output='screen',
        parameters=[{
            'camera_topic': camera_topic,
            'depth_topic': depth_topic,
            'camera_info_topic': camera_info_topic,
            'hand_side': hand_side,
            'sdk_value_mode': sdk_value_mode,
            'joint_order': 'standard_o6',
            'show_cv_window': True,
            'smoothing_alpha': 0.35,
            
            # New Auto-Grasp Parameters
            'auto_grasp_on_proximity': auto_grasp,
            'grasp_distance_mm': grasp_dist,
            'target_marker_id': marker_id,
        }]
    )

    return LaunchDescription([
        DeclareLaunchArgument('launch_realsense', default_value='true'),
        DeclareLaunchArgument('launch_sdk', default_value='true'),
        DeclareLaunchArgument('hand_side', default_value='right'),
        DeclareLaunchArgument('sdk_value_mode', default_value='raw_255'),
        DeclareLaunchArgument('camera_topic', default_value='/camera/camera/color/image_raw'),
        
        # New Auto-Grasp Arguments
        DeclareLaunchArgument('depth_topic', default_value='/camera/camera/aligned_depth_to_color/image_raw'),
        DeclareLaunchArgument('camera_info_topic', default_value='/camera/camera/aligned_depth_to_color/camera_info'),
        DeclareLaunchArgument('auto_grasp_on_proximity', default_value='false'),
        DeclareLaunchArgument('grasp_distance_mm', default_value='110.0'),
        DeclareLaunchArgument('target_marker_id', default_value='0'), # Expecting marker ID 0 on the clamp
        
        realsense_group,
        linkerhand_sdk_group,
        teleop_node,
    ])