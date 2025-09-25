# natural_command_node.py — Bridge 전용 (LLM 없음, 실제 로봇용)

import math
import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.action import GripperCommand
from control_msgs.msg import GripperCommand as GripperCommandMsg
from rclpy.action import ActionClient

# 로봇 링크 길이 (필요시 실제 로봇 스펙으로 수정)
L2 = 0.128
L3 = 0.124

class NaturalCommandNode(Node):
    def __init__(self):
        super().__init__('natural_command_node')

        # 실제 로봇 토픽 이름 확인 필요!
        self.arm_pub = self.create_publisher(
            JointTrajectory,
            '/arm_controller/joint_trajectory',   # 실제 로봇 bringup 토픽으로 변경
            10
        )
        self.gripper_client = ActionClient(
            self,
            GripperCommand,
            '/gripper_controller/gripper_cmd'    # 실제 토픽 이름 확인 필요
        )

        # 현재 조인트 상태 (간단 추적)
        self.current_joint1 = 0.0
        self.current_joint2 = 0.0
        self.current_joint3 = 0.0
        self.current_joint4 = 0.0

        self.get_logger().info("✅ NaturalCommandNode started (REAL ROBOT MODE)")

    # -------------------------
    # JSON 명령 처리
    # -------------------------
    def process_command(self, cmd: dict):
        try:
            action = cmd.get("action")
            direction = (cmd.get("direction") or "").lower()
            value = cmd.get("value")
            unit = cmd.get("unit")

            if action == "initialize":
                self.reset_pose()
                return

            if action == "gripper":
                self.control_gripper(direction)
                return

            if action == "rotate":
                delta = self.get_delta(value, unit, L3)
                if direction == "left":
                    self.current_joint1 += delta
                elif direction == "right":
                    self.current_joint1 -= delta
                elif direction == "up":
                    self.current_joint3 -= delta
                elif direction == "down":
                    self.current_joint3 += delta
                else:
                    self.get_logger().warn(f"⚠️ Unknown rotate dir: {direction}")
                    return
                self.send_arm_command()
                self.get_logger().info(f"✅ Rotate {direction} {value}{unit}")
                return

            if action == "move":
                step = self.get_delta(value, unit, 1.0)
                if direction == "forward":
                    self.current_joint2 += step
                    self.current_joint3 -= step
                elif direction == "backward":
                    self.current_joint2 -= step
                    self.current_joint3 += step
                else:
                    self.get_logger().warn(f"⚠️ Unknown move dir: {direction}")
                    return
                self.send_arm_command()
                self.get_logger().info(f"✅ Move {direction} {value}{unit}")
                return

            self.get_logger().warn(f"⚠️ Unknown action: {action}")

        except Exception as e:
            self.get_logger().error(f"❌ process_command error: {e}")

    # -------------------------
    # 로봇 제어 함수들
    # -------------------------
    def send_arm_command(self):
        traj = JointTrajectory()
        traj.joint_names = ['joint1', 'joint2', 'joint3', 'joint4']
        pt = JointTrajectoryPoint()
        pt.positions = [
            self.current_joint1,
            self.current_joint2,
            self.current_joint3,
            self.current_joint4
        ]
        pt.time_from_start.sec = 5   # 실제 로봇은 안전하게 5초 정도로
        traj.points.append(pt)
        self.arm_pub.publish(traj)

    def control_gripper(self, direction: str):
        pos_map_deg = {"open": 57.0, "close": 0.0, "reset": 0.0}
        pos_deg = pos_map_deg.get(direction, None)
        if pos_deg is None:
            self.get_logger().warn(f"⚠️ Invalid gripper cmd: {direction}")
            return
        pos_rad = math.radians(pos_deg)
        goal = GripperCommand.Goal()
        goal.command = GripperCommandMsg()
        goal.command.position = pos_rad
        goal.command.max_effort = 1.0
        self.gripper_client.wait_for_server()
        self.gripper_client.send_goal_async(goal)
        self.get_logger().info(f"🦾 Gripper {direction} executed")

    def reset_pose(self):
        self.current_joint1 = 0.0
        self.current_joint2 = 0.0
        self.current_joint3 = 0.0
        self.current_joint4 = 0.0
        self.send_arm_command()
        self.get_logger().info("✅ Robot reset pose executed")

    def get_delta(self, value, unit, radius):
        try:
            if unit in ("degree", "deg"):
                return math.radians(float(value))
            elif unit == "cm":
                return (float(value) / 100.0) / radius
            elif unit == "mm":
                return (float(value) / 1000.0) / radius
            elif unit == "m":
                return float(value) / radius
            elif unit in ("inch", "in"):
                return (float(value) * 0.0254) / radius
        except:
            return 0.0
        return 0.0

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
