# bridge.py
import threading
from flask import Flask, request, jsonify
import rclpy
from omy_f3m_natural_command_node import NaturalCommandNode

app = Flask(__name__)
node = None   # ROS2 노드 인스턴스 저장

@app.route('/cmd', methods=['POST'])
def cmd():
    global node
    data = request.json
    english_cmd = data.get("english", "")

    if not english_cmd:
        return jsonify({"error": "No english command"}), 400

    # ROS2 노드에서 파싱 + 실행
    cmd = node.parse_command_with_llm(english_cmd)
    if not cmd:
        return jsonify({"error": "Parse failed"}), 500
    
    node.get_logger().info(f"🌐 Received via Flask: {english_cmd}")
    node.process_command(cmd)
    return jsonify({"status": "ok", "parsed": cmd})

def ros2_spin():
    global node
    rclpy.init()
    node = NaturalCommandNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

def main():
    # ROS2 실행 (스레드)
    ros_thread = threading.Thread(target=ros2_spin, daemon=True)
    ros_thread.start()

    # Flask 실행
    app.run(host="0.0.0.0", port=6000)

if __name__ == "__main__":
    main()