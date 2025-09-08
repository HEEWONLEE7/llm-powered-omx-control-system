# natural_command_node.py — LLM-only JSON parser integration
# 2025-09-05 — add Cartesian path (with start_state) & joint4 look-up/down correction

# -------------------- import --------------------
import re
import json
import math
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.action import GripperCommand
from control_msgs.msg import GripperCommand as GripperCommandMsg
from rclpy.action import ActionClient
from geometry_msgs.msg import PoseStamped
from moveit_msgs.srv import GetPositionIK, GetCartesianPath
from moveit_msgs.msg import RobotState
import threading
import numpy as np

# LLM-only JSON parser
try:
    from omx_f_llm_parser import parse_to_json
    HAS_LLM = True
except Exception as e:
    HAS_LLM = False
    print(f"[LLM] import failed: {e}")


# -------------------- constants --------------------
L2 = 0.128
L3 = 0.124
L_GRIPPER = 0.126

KEEP_ROTATE_SPEED_DEG_S = 10.0
KEEP_TIMER_HZ = 20.0
KEEP_DT = 1.0 / KEEP_TIMER_HZ


# -------------------- node class --------------------
class NaturalCommandNode(Node):
    def __init__(self):
        super().__init__('natural_command_node')

        # ROS I/O
        self.arm_pub = self.create_publisher(JointTrajectory, '/arm_controller/joint_trajectory', 10)
        self.gripper_client = ActionClient(self, GripperCommand, '/gripper_controller/gripper_cmd')

        # MoveIt services
        self.ik_client = self.create_client(GetPositionIK, '/compute_ik')
        self.cartesian_client = self.create_client(GetCartesianPath, '/compute_cartesian_path')

        while not self.ik_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('⏳ Waiting for /compute_ik service...')
        while not self.cartesian_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('⏳ Waiting for /compute_cartesian_path service...')

        # Joint state snapshots
        self.current_joint1_pos = 0.0
        self.current_joint2_pos = 0.0
        self.current_joint3_pos = 0.0
        self.current_joint4_pos = 0.0
        self.current_z = L2

        # EE pose init
        self.current_ee_pose = PoseStamped()
        self.current_ee_pose.header.frame_id = "world"
        self.current_ee_pose.pose.position.x = 0.0
        self.current_ee_pose.pose.position.y = 0.0
        self.current_ee_pose.pose.position.z = self.current_z
        self.current_ee_pose.pose.orientation.w = 1.0

        # Param
        self.declare_parameter('use_llm', True)
        self.use_llm = bool(self.get_parameter('use_llm').value) and HAS_LLM
        if not HAS_LLM:
            self.get_logger().warn("⚠️ LLM parser not available; commands will be ignored.")
        else:
            self.get_logger().info(f"✅ LLM parser ready (use_llm={self.use_llm})")

        # Continuous rotation state
        self._keep_timer = None
        self._keep_dir = None
        self._keep_w_rad_s = math.radians(KEEP_ROTATE_SPEED_DEG_S)

    # ---- LLM wrapper ----
    def parse_command_with_llm(self, text: str):
        if not self.use_llm:
            return None
        try:
            data = parse_to_json(text)
            if not isinstance(data, dict):
                return None
            for k in ["action", "direction", "value", "unit", "xyz"]:
                if k not in data:
                    return None
            if data["action"] is None:
                return None
            return data
        except Exception as e:
            self.get_logger().error(f"[LLM] parse failed: {e}")
            return None

    # ---- execution entry ----
    def process_command(self, cmd):
        try:
            action = cmd.get("action")

            # STOP
            if action == "stop":
                self._stop_keep()
                self.get_logger().info("⛔ Stopped continuous rotation.")
                return

            if action == "move_xyz":
                x, y, z = cmd["xyz"]
                self.send_ik_request(x, y, z)
                return

            if action == "initialize":
                self.reset_pose()
                return

            if action == "gripper":
                pos_map_deg = {"open": 57.0, "close": 0.0, "reset": 0.0}
                pos_deg = pos_map_deg.get(str(cmd.get("direction") or "").lower(), None)

                if pos_deg is None:
                    self.get_logger().warn(f"⚠️ Unknown gripper direction: {cmd.get('direction')}")
                    return

                # degree → rad 변환
                pos_rad = math.radians(pos_deg)

                goal = GripperCommand.Goal()
                goal.command = GripperCommandMsg()
                goal.command.position = pos_rad
                goal.command.max_effort = 1.0
                self.gripper_client.wait_for_server()
                self.gripper_client.send_goal_async(goal)
                self.get_logger().info(f"🦾 Gripper '{cmd['direction']}' executed")
                return

            # NEW: Look command
            if action == "look":
                direction = (cmd.get("direction") or "").lower()
                self.look_command(direction)
                return

            # ROTATE / MOVE
            if action in ("rotate", "move"):
                direction = cmd.get("direction")
                value = cmd.get("value")
                unit = cmd.get("unit")

                # Continuous ROTATE
                if (action == "rotate") and isinstance(value, str) and value.strip().lower() == "keep":
                    self._start_keep(direction)
                    return

                if action == "rotate":
                    delta = self.get_delta(value, unit, L3)
                    if direction == "left":
                        self.current_joint1_pos += delta
                    elif direction == "right":
                        self.current_joint1_pos -= delta
                    elif direction == "up":
                        self.current_joint3_pos -= delta
                    elif direction == "down":
                        self.current_joint3_pos += delta
                    else:
                        self.get_logger().warn(f"⚠️ Unsupported rotate direction: {direction}")
                        return
                    traj = JointTrajectory()
                    traj.joint_names = ['joint1', 'joint2', 'joint3', 'joint4']
                    pt = JointTrajectoryPoint()
                    pt.positions = [
                        self.current_joint1_pos,
                        self.current_joint2_pos,
                        self.current_joint3_pos,
                        self.current_joint4_pos
                    ]
                    pt.time_from_start.sec = 2
                    traj.points.append(pt)
                    self.arm_pub.publish(traj)
                    self.get_logger().info(f"✅ Rotate {direction} {value}{unit} executed")
                    return

                elif action == "move":
                    direction = cmd.get("direction")
                    value = cmd.get("value")
                    unit = cmd.get("unit")
                    step = self.get_delta(value, unit, 1.0)
                    if step == 0.0:
                        self.get_logger().warn(f"⚠️ invalid or zero distance: {value} {unit}")
                        return
                    if direction == "forward":
                        self.move_forward_backward(+step)
                    elif direction == "backward":
                        self.move_forward_backward(-step)
                    else:
                        self.get_logger().warn(f"⚠️ Unsupported move direction: {direction}")
                    return


            self.get_logger().warn(f"⚠️ Unknown action: {action}")

        except Exception as e:
            self.get_logger().error(f"❌ process_command error: {e}")

    # ---- Cartesian move (with start_state) ----
    def move_forward_backward(self, step_m):
        delta_theta = step_m / L2

        self.current_joint2_pos += delta_theta
        self.current_joint3_pos -= delta_theta

        traj = JointTrajectory()
        traj.joint_names = ['joint1', 'joint2', 'joint3', 'joint4']
        pt = JointTrajectoryPoint()
        pt.positions = [
            self.current_joint1_pos,
            self.current_joint2_pos,
            self.current_joint3_pos,
            self.current_joint4_pos
        ]
        pt.time_from_start.sec = 2
        traj.points.append(pt)
        self.arm_pub.publish(traj)

        self.get_logger().info(
            f"✅ Simplified move step={step_m:+.3f} m "
            f"(joint2={self.current_joint2_pos:.2f}, joint3={self.current_joint3_pos:.2f})"
        )


    # ---- Look up/down with simple joint4 formula ----
    def look_command(self, direction: str):
        theta23 = self.current_joint2_pos + self.current_joint3_pos

        if direction == "up":
            self.current_joint4_pos = -theta23
        elif direction == "down":
            self.current_joint4_pos = -theta23 + math.pi/2
        else:
            self.get_logger().warn(f"⚠️ Unsupported look direction: {direction}")
            return

        traj = JointTrajectory()
        traj.joint_names = ['joint1', 'joint2', 'joint3', 'joint4']
        pt = JointTrajectoryPoint()
        pt.positions = [
            self.current_joint1_pos,
            self.current_joint2_pos,
            self.current_joint3_pos,
            self.current_joint4_pos
        ]
        pt.time_from_start.sec = 2
        traj.points.append(pt)
        self.arm_pub.publish(traj)

        self.get_logger().info(
            f"✅ Look {direction} executed (joint4={self.current_joint4_pos:.2f} rad)"
        )

    # ---- utils ----
    def quaternion_to_matrix(self, q):
        w, x, y, z = q
        return np.array([
            [1 - 2*y*y - 2*z*z,   2*x*y - 2*z*w,     2*x*z + 2*y*w],
            [2*x*y + 2*z*w,       1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
            [2*x*z - 2*y*w,       2*y*z + 2*x*w,     1 - 2*x*x - 2*y*y]
        ])

    def get_delta(self, value, unit, radius):
        if unit and unit.lower() in ("degree", "deg"):
            return math.radians(float(value))
        elif unit and unit.lower() == "cm":
            return (float(value) / 100.0) / radius if radius != 0 else 0.0
        elif unit and unit.lower() in ("mm",):
            return ((float(value) / 1000.0) / radius) if radius != 0 else 0.0
        elif unit and unit.lower() in ("m",):
            return (float(value) / radius) if radius != 0 else 0.0
        elif unit and unit.lower() in ("inch", "in"):
            return ((float(value) * 0.0254) / radius) if radius != 0 else 0.0
        return 0.0

    def reset_pose(self):
        self.current_joint1_pos = 0.0
        self.current_joint2_pos = 0.0
        self.current_joint3_pos = 0.0
        self.current_joint4_pos = 0.0
        self.current_z = L2
        traj = JointTrajectory()
        traj.joint_names = ['joint1', 'joint2', 'joint3', 'joint4']
        pt = JointTrajectoryPoint()
        pt.positions = [0.0, 0.0, 0.0, 0.0]
        pt.time_from_start.sec = 2
        traj.points.append(pt)
        self.arm_pub.publish(traj)
        goal = GripperCommand.Goal()
        goal.command = GripperCommandMsg()
        goal.command.position = 0.00
        goal.command.max_effort = 1.0
        self.gripper_client.wait_for_server()
        self.gripper_client.send_goal_async(goal)
        self.get_logger().info("✅ Initialization complete")

    def send_ik_request(self, x, y, z):
        pose = PoseStamped()
        pose.header.frame_id = "world"
        pose.pose.position.x = float(x)
        pose.pose.position.y = float(y)
        pose.pose.position.z = float(z)
        pose.pose.orientation.w = 1.0
        req = GetPositionIK.Request()
        req.ik_request.group_name = "arm"
        req.ik_request.ik_link_name = "end_effector_link"
        req.ik_request.pose_stamped = pose
        req.ik_request.timeout.sec = 2
        future = self.ik_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        res = future.result()
        if res and res.error_code.val == 1:
            j = res.solution.joint_state
            name_to_pos = dict(zip(j.name, j.position))
            self.current_joint1_pos = name_to_pos.get('joint1', 0.0)
            self.current_joint2_pos = name_to_pos.get('joint2', 0.0)
            self.current_joint3_pos = name_to_pos.get('joint3', 0.0)
            self.current_joint4_pos = name_to_pos.get('joint4', 0.0)
            self.current_z = L2 + L3 * math.sin(self.current_joint3_pos)
            traj = JointTrajectory()
            traj.joint_names = ['joint1', 'joint2', 'joint3', 'joint4']
            pt = JointTrajectoryPoint()
            pt.positions = [
                self.current_joint1_pos,
                self.current_joint2_pos,
                self.current_joint3_pos,
                self.current_joint4_pos
            ]
            pt.time_from_start.sec = 2
            traj.points.append(pt)
            self.arm_pub.publish(traj)
            self.get_logger().info(f"✅ IK-based movement completed: {pt.positions}")
        else:
            code = res.error_code.val if res else -1
            self.get_logger().error(f"❌ IK computation failed (code: {code})")

    # ---- continuous rotate helpers ----
    def _start_keep(self, direction: str):
        d = (direction or "").lower()
        if d not in ("left", "right", "up", "down"):
            self.get_logger().warn(f"⚠️ Unsupported direction for continuous rotate: {direction}")
            return
        self._stop_keep()
        self._keep_dir = d
        self._keep_timer = self.create_timer(KEEP_DT, self._on_keep_tick)
        self.get_logger().info(f"🔄 Continuous rotate '{d}' at {KEEP_ROTATE_SPEED_DEG_S:.1f} deg/s.")

    def _stop_keep(self):
        if self._keep_timer is not None:
            self._keep_timer.cancel()
            self._keep_timer = None
        self._keep_dir = None

    def _on_keep_tick(self):
        if not self._keep_dir:
            return
        traj = JointTrajectory()
        traj.joint_names = ['joint1', 'joint2', 'joint3', 'joint4']
        pt = JointTrajectoryPoint()
        w = self._keep_w_rad_s
        d = self._keep_dir
        if d == "left":
            self.current_joint1_pos += w * KEEP_DT
        elif d == "right":
            self.current_joint1_pos -= w * KEEP_DT
        elif d == "up":
            self.current_joint3_pos -= w * KEEP_DT
        elif d == "down":
            self.current_joint3_pos += w * KEEP_DT
        pt.positions = [
            self.current_joint1_pos,
            self.current_joint2_pos,
            self.current_joint3_pos,
            self.current_joint4_pos
        ]
        pt.time_from_start.sec = 0
        pt.time_from_start.nanosec = int(KEEP_DT * 1e9)
        traj.points.append(pt)
        self.arm_pub.publish(traj)


# ---- background spin worker ----
def _spin_worker(node, stop_evt):
    while rclpy.ok() and not stop_evt.is_set():
        rclpy.spin_once(node, timeout_sec=0.05)

def main():
    rclpy.init()
    node = NaturalCommandNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
