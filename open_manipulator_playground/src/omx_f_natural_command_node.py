# natural_command_node.py — LLM-only JSON parser integration
# 2025-08-12 — add continuous rotate with "keep" + stop (comments in English)

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
from moveit_msgs.srv import GetPositionIK
import threading  # ← ADDED: background spin for timers

# LLM-only JSON parser (loaded once / cached internally)
try:
    HAS_LLM = True
except Exception as e:
    HAS_LLM = False
    print(f"[LLM] import failed: {e}")

# -------------------- constants --------------------
L2 = 0.128
L3 = 0.124
L_GRIPPER = 0.126

# Continuous rotation parameters (for value == "keep")
KEEP_ROTATE_SPEED_DEG_S = 10.0   # deg/s
KEEP_TIMER_HZ = 20.0             # update frequency for smooth motion
KEEP_DT = 1.0 / KEEP_TIMER_HZ


# -------------------- node class --------------------
class NaturalCommandNode(Node):
    def __init__(self):
        super().__init__('natural_command_node')

        # ROS I/O
        self.arm_pub = self.create_publisher(JointTrajectory, '/arm_controller/joint_trajectory', 10)
        self.gripper_client = ActionClient(self, GripperCommand, '/gripper_controller/gripper_cmd')

        self.ik_client = self.create_client(GetPositionIK, '/compute_ik')
        while not self.ik_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('⏳ Waiting for /compute_ik service...')

        # Joint state snapshots (simple internal state for demo)
        self.current_joint1_pos = 0.0
        self.current_joint2_pos = 0.0
        self.current_joint3_pos = 0.0
        self.current_joint4_pos = 0.0
        self.current_z = L2

        # Param: allow toggling LLM usage
        self.declare_parameter('use_llm', True)
        self.use_llm = bool(self.get_parameter('use_llm').value) and HAS_LLM
        if not HAS_LLM:
            self.get_logger().warn("⚠️ LLM parser not available; commands will be ignored (no legacy regex).")
        else:
            self.get_logger().info(f"✅ LLM parser ready (use_llm={self.use_llm})")

        # Continuous rotation state
        self._keep_timer = None
        self._keep_dir = None    # 'left' | 'right' | 'up' | 'down'
        self._keep_w_rad_s = math.radians(KEEP_ROTATE_SPEED_DEG_S)

    # ---- LLM wrapper (replaces legacy keyword extraction) ----
    def parse_command_with_llm(self, text: str):
        """
        Return JSON from LLM-only parser after minimal schema checks.
        Example: {"action":"move","direction":"left","value":10,"unit":"cm","xyz":None}
        Returns None on failure/invalid input.
        """
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
        """
        cmd: LLM JSON (action, direction, value, unit, xyz)
        """
        try:
            action = cmd.get("action")

            # STOP: immediately halt continuous rotation (if any)
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
                pos_map = {"open": 0.010, "close": -0.008, "reset": 0.000}
                pos = pos_map.get(str(cmd.get("direction") or "").lower(), None)
                if pos is None:
                    self.get_logger().warn(f"⚠️ Unknown gripper direction: {cmd.get('direction')}")
                    return
                goal = GripperCommand.Goal()
                goal.command = GripperCommandMsg()
                goal.command.position = pos
                goal.command.max_effort = 1.0
                self.gripper_client.wait_for_server()
                self.gripper_client.send_goal_async(goal)
                self.get_logger().info(f"🦾 Gripper '{cmd['direction']}' executed")
                return

            # ROTATE / MOVE
            if action in ("rotate", "move"):
                direction = cmd.get("direction")
                value = cmd.get("value")
                unit = cmd.get("unit")

                # Continuous ROTATE: value == "keep" → start smooth rotation until 'stop'
                if (action == "rotate") and isinstance(value, str) and value.strip().lower() == "keep":
                    self._start_keep(direction)
                    return

                # One-shot path (unchanged)
                if value is None or unit is None or direction is None:
                    self.get_logger().warn(f"⚠️ Incomplete fields for {action}: {cmd}")
                    return

                traj = JointTrajectory()
                traj.joint_names = ['joint1', 'joint2', 'joint3', 'joint4']
                point = JointTrajectoryPoint()

                radius = L3 * math.cos(self.current_joint3_pos) + L_GRIPPER
                delta = self.get_delta(value, unit, radius)  # convert units → joint delta

                if action == "rotate":
                    if direction == "left":
                        self.current_joint1_pos += delta
                    elif direction == "right":
                        self.current_joint1_pos -= delta
                    elif direction == "up":
                        tgt = self.current_joint3_pos - delta
                        if -math.pi/2 <= tgt <= math.pi/2:
                            self.current_joint3_pos = tgt
                            self.current_joint4_pos = -tgt
                            self.current_z = L2 + L3 * math.sin(tgt)
                    elif direction == "down":
                        tgt = self.current_joint3_pos + delta
                        if -math.pi/2 <= tgt <= math.pi/2:
                            self.current_joint3_pos = tgt
                            self.current_joint4_pos = -tgt
                            self.current_z = L2 + L3 * math.sin(tgt)
                    else:
                        self.get_logger().warn(f"⚠️ Unsupported rotate direction: {direction}")
                        return

                elif action == "move":
                    if direction in ["left", "right"]:
                        self.current_joint1_pos += delta if direction == "left" else -delta

                    elif direction in ["up", "down"]:
                        sign = -1 if direction == "up" else 1
                        target_z = self.current_z + sign * (value / 100.0)  # cm → m
                        success, theta3 = self.inverse_kinematics_z(target_z)
                        if success:
                            self.current_joint3_pos = theta3
                            self.current_joint4_pos = -theta3
                            self.current_z = target_z
                        else:
                            self.get_logger().warn(f"⚠️ IK(Z) out of range: {target_z:.3f} m")
                            return

                    elif direction in ["forward", "backward"]:
                        # Distance units only
                        if unit and str(unit).lower() in ("degree", "deg"):
                            self.get_logger().warn("⚠️ forward/backward accepts distance units only (cm/mm/m/inch).")
                            return

                        l = (L3 + L_GRIPPER)  # effective length
                        step = abs(self.get_delta(value, unit, 1.0))  # [m]
                        if step == 0.0:
                            self.get_logger().warn(f"⚠️ invalid or zero distance: {value} {unit}")
                            return

                        # Current half-angle projection and horizontal reach
                        alpha_cur = 0.5 * (self.current_joint2_pos - self.current_joint3_pos)
                        d_cur = l * math.cos(alpha_cur)

                        # Always compute w.r.t. backward folding (decrease reach)
                        s_back = -step
                        d_new = d_cur + s_back

                        # Clamp for safe acos
                        eps = 1e-9
                        d_new = max(-l + eps, min(l - eps, d_new))

                        a = math.acos(d_new / l)   # 0..π

                        if direction == "forward":
                            self.current_joint2_pos = +a
                            self.current_joint3_pos = -a
                            a_log = -math.degrees(a)
                        else:
                            self.current_joint2_pos = -a
                            self.current_joint3_pos = +a
                            a_log = +math.degrees(a)

                point.positions = [
                    self.current_joint1_pos,
                    self.current_joint2_pos,
                    self.current_joint3_pos,
                    self.current_joint4_pos
                ]
                point.time_from_start.sec = 2
                traj.points.append(point)
                self.arm_pub.publish(traj)
                self.get_logger().info(f"✅ Joint movement completed: {point.positions}")
                return

            # Unknown action
            self.get_logger().warn(f"⚠️ Unknown action: {action}")

        except Exception as e:
            self.get_logger().error(f"❌ process_command error: {e}")

    # ---- continuous rotate helpers ----
    def _start_keep(self, direction: str):
        """Start smooth continuous rotation in the specified direction until 'stop'."""
        d = (direction or "").lower()
        if d not in ("left", "right", "up", "down"):
            self.get_logger().warn(f"⚠️ Unsupported direction for continuous rotate: {direction}")
            return

        # Cancel previous timer if any
        self._stop_keep()

        self._keep_dir = d
        self._keep_timer = self.create_timer(KEEP_DT, self._on_keep_tick)
        self.get_logger().info(f"🔄 Continuous rotate '{d}' at {KEEP_ROTATE_SPEED_DEG_S:.1f} deg/s (say 'stop' to halt).")

    def _stop_keep(self):
        """Stop continuous rotation if running."""
        if self._keep_timer is not None:
            self._keep_timer.cancel()
            self._keep_timer = None
        self._keep_dir = None

    def _on_keep_tick(self):
        """Periodic update for continuous rotation (PWM-like smooth increments)."""
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
            tgt = self.current_joint3_pos - w * KEEP_DT
            if -math.pi / 2 <= tgt <= math.pi / 2:
                self.current_joint3_pos = tgt
                self.current_joint4_pos = -tgt
                self.current_z = L2 + L3 * math.sin(tgt)
        elif d == "down":
            tgt = self.current_joint3_pos + w * KEEP_DT
            if -math.pi / 2 <= tgt <= math.pi / 2:
                self.current_joint3_pos = tgt
                self.current_joint4_pos = -tgt
                self.current_z = L2 + L3 * math.sin(tgt)

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

    # ---- helpers ----
    def get_delta(self, value, unit, radius):
        # Unit normalization is handled by LLM ('degree' or distance units).
        if unit and unit.lower() in ("degree", "deg"):
            return math.radians(float(value))
        elif unit and unit.lower() == "cm":
            # Simple joint conversion (arc-length approx) — tune if needed
            return (float(value) / 100.0) / radius if radius != 0 else 0.0
        elif unit and unit.lower() in ("mm",):
            return ((float(value) / 1000.0) / radius) if radius != 0 else 0.0
        elif unit and unit.lower() in ("m",):
            return (float(value) / radius) if radius != 0 else 0.0
        elif unit and unit.lower() in ("inch", "in"):
            return ((float(value) * 0.0254) / radius) if radius != 0 else 0.0
        return 0.0

    def inverse_kinematics_z(self, z_target):
        delta = z_target - L2
        if abs(delta / L3) > 1.0:
            return False, 0.0
        return True, math.asin(delta / L3)

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
        # Vertical gripper example orientation; adjust per your frames if needed
        pose.pose.orientation.x = 0.0
        pose.pose.orientation.y = 1.0
        pose.pose.orientation.z = 0.0
        pose.pose.orientation.w = 0.0

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

# ---- background spin worker (ADDED) ----
def _spin_worker(node, stop_evt):
    """Background rclpy event loop so timers ('keep') fire while input() blocks."""
    while rclpy.ok() and not stop_evt.is_set():
        rclpy.spin_once(node, timeout_sec=0.05)

def main():
    rclpy.init()
    node = NaturalCommandNode()

    # Start background spin so create_timer callbacks run during input()
    stop_evt = threading.Event()
    spin_thread = threading.Thread(target=_spin_worker, args=(node, stop_evt), daemon=True)
    spin_thread.start()

    print("💬 Enter a command (e.g., 'move to 0.2 0.0 0.1', 'rotate left 10 degree', 'rotate right keep', 'stop', 'gripper open') - Type 'exit' to quit")
    try:
        while rclpy.ok():
            text = input(">>> ").strip()
            if text.lower() == "exit":
                break

            cmd = node.parse_command_with_llm(text)
            if not cmd:
                print("⚠️ Could not parse command (LLM off or invalid).")
                continue

            print("📦 Parsed command:", json.dumps(cmd, ensure_ascii=False))
            node.process_command(cmd)
    except KeyboardInterrupt:
        pass
    finally:
        # Stop background spin and shutdown ROS
        stop_evt.set()
        spin_thread.join(timeout=1.0)
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()