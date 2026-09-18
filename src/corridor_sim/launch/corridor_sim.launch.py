import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.actions import Node
import xacro


def generate_launch_description():
    pkg_dir = get_package_share_directory('corridor_sim')
    world_path = os.path.join(pkg_dir, 'worlds', 'corridor.sdf')
    xacro_file = os.path.join(pkg_dir, 'urdf', 'robot.urdf.xacro')

    robot_desc = xacro.process_file(xacro_file).toxml()

    gz_sim = ExecuteProcess(
        cmd=['ign', 'gazebo', '-r', '-s', world_path],
        output='screen'
    )

    spawn_robot = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=[
            '-string', robot_desc,
            '-name', 'diff_bot',
            '-x', '1.0',
            '-y', '0.0',
            '-z', '0.1'
        ],
        output='screen'
    )

    robot_state_pub = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[
            {
                'robot_description': robot_desc,
                'use_sim_time': True
            }
        ],
        output='screen'
    )

    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        parameters=[
            {
                'config_file': os.path.join(
                    pkg_dir,
                    'config',
                    'bridge.yaml'
                )
            }
        ],
        output='screen'
    )

    wall_shifter = Node(
        package='corridor_sim',
        executable='wall_shifter',
        output='screen'
    )

    noise_injector = Node(
        package='corridor_sim',
        executable='noise_injector',
        output='screen'
    )

    estimator = Node(
        package='corridor_sim',
        executable='task2_estimator',
        output='screen'
    )

    return LaunchDescription([
        gz_sim,
        spawn_robot,
        robot_state_pub,
        bridge,
        wall_shifter,
        noise_injector,
        estimator
    ])