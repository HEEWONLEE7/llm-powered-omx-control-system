from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, RegisterEventHandler, ExecuteProcess, IncludeLaunchDescription
from launch.conditions import IfCondition, UnlessCondition
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, FindExecutable, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
import os

def generate_launch_description():
    # Declare launch arguments
    declared_arguments = [
        DeclareLaunchArgument('start_rviz', default_value='false', description='Whether to execute rviz2'),
        DeclareLaunchArgument('prefix', default_value='""', description='Prefix of the joint and link names'),
        DeclareLaunchArgument('use_sim', default_value='false', description='Start robot in Gazebo simulation.'),
        DeclareLaunchArgument('use_fake_hardware', default_value='false', description='Use fake hardware mirroring command.'),
        DeclareLaunchArgument('fake_sensor_commands', default_value='false', description='Enable fake sensor commands.'),
        DeclareLaunchArgument('port_name', default_value='/dev/ttyACM0', description='Port name for hardware connection.'),
        DeclareLaunchArgument('init_position', default_value='true', description='Whether to launch the init_position node'),
        DeclareLaunchArgument('ros2_control_type', default_value='omx_f', description='Type of ros2_control'),
        DeclareLaunchArgument('init_position_file', default_value='initial_positions.yaml', description='Path to the initial position file'),
    ]

    # Launch configurations
    start_rviz = LaunchConfiguration('start_rviz')
    prefix = LaunchConfiguration('prefix')
    use_sim = LaunchConfiguration('use_sim')
    use_fake_hardware = LaunchConfiguration('use_fake_hardware')
    fake_sensor_commands = LaunchConfiguration('fake_sensor_commands')
    port_name = LaunchConfiguration('port_name')
    init_position = LaunchConfiguration('init_position')
    ros2_control_type = LaunchConfiguration('ros2_control_type')
    init_position_file = LaunchConfiguration('init_position_file')

    # Generate URDF file using xacro
    urdf_file = Command([
        PathJoinSubstitution([FindExecutable(name='xacro')]),
        ' ',
        PathJoinSubstitution([
            FindPackageShare('open_manipulator_description'),
            'urdf',
            'omx_f',
            'omx_f.urdf.xacro',
        ]),
        ' ',
        'prefix:=', prefix, ' ',
        'use_sim:=', use_sim, ' ',
        'use_fake_hardware:=', use_fake_hardware, ' ',
        'fake_sensor_commands:=', fake_sensor_commands, ' ',
        'port_name:=', port_name, ' ',
        'ros2_control_type:=', ros2_control_type,
    ])

    # Paths for configuration files
    controller_manager_config = PathJoinSubstitution([
        FindPackageShare('open_manipulator_bringup'),
        'config', 'omx_f', 'hardware_controller_manager.yaml',
    ])

    rviz_config_file = PathJoinSubstitution([
        FindPackageShare('open_manipulator_description'),
        'rviz', 'open_manipulator.rviz',
    ])

    trajectory_params_file = PathJoinSubstitution([
        FindPackageShare('open_manipulator_bringup'),
        'config', 'omx_f', init_position_file,
    ])

    moveit_config = PathJoinSubstitution([
        FindPackageShare('open_manipulator_moveit_config'),
        'config', 'omx_f.srdf',
    ])

    # Define nodes
    control_node = Node(
        package='controller_manager',
        executable='ros2_control_node',
        parameters=[{'robot_description': urdf_file}, controller_manager_config],
        output='both',
        condition=UnlessCondition(use_sim),
    )

    robot_controller_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['arm_controller', 'gripper_controller', 'joint_state_broadcaster'],
        output='both',
        parameters=[{'robot_description': urdf_file}],
    )

    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{'robot_description': urdf_file, 'use_sim_time': use_sim}],
        output='both',
    )

    joint_trajectory_executor = Node(
        package='open_manipulator_bringup',
        executable='joint_trajectory_executor',
        parameters=[trajectory_params_file],
        output='both',
        condition=IfCondition(init_position),
    )

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        arguments=['-d', rviz_config_file],
        output='both',
        condition=IfCondition(start_rviz),
    )

    bridge_node = ExecuteProcess(
        cmd=["python3", os.path.join(
            os.path.dirname(__file__), "/root/ros2_ws/src/open_manipulator/open_manipulator_playground/src/omx_f_bridge.py"
        )],
        output="screen"
    )

    moveit_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('open_manipulator_moveit_config'),
                'launch',
                'omx_f_moveit.launch.py'
            ])
        ]),
        launch_arguments={
            'use_sim': use_sim,
            'start_rviz': start_rviz
        }.items()
    )

    # Event handlers
    delay_rviz_after_joint_state_broadcaster_spawner = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=robot_controller_spawner, on_exit=[rviz_node]
        )
    )

    delay_joint_trajectory_executor_after_controllers = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=robot_controller_spawner,
            on_exit=[joint_trajectory_executor],
        )
    )

    return LaunchDescription(
        declared_arguments
        + [
            control_node,
            robot_controller_spawner,
            robot_state_publisher_node,
            moveit_launch,   # move_group_node 대신 moveit_launch로 변경
            delay_rviz_after_joint_state_broadcaster_spawner,
            delay_joint_trajectory_executor_after_controllers,
            bridge_node,
        ]
    )
