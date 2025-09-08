import threading
from flask import Flask, request, jsonify
from flask_cors import CORS
import rclpy
from omx_f_natural_command_node import NaturalCommandNode

# Flask 설정
app = Flask(__name__)
CORS(app)

node = None   # ROS2 노드 인스턴스 저장

@app.route('/cmd', methods=['POST'])
def cmd():
    """UI → Bridge: 영어 명령을 받아 ROS2 노드로 전달"""
    global node
    data = request.json
    english_cmd = data.get("english", "")

    if not english_cmd:
        return jsonify({"error": "No english command"}), 400

    if node is None:
        return jsonify({"error": "ROS2 node not ready"}), 500

    try:
        # LLM 파서로 JSON 변환
        cmd = node.parse_command_with_llm(english_cmd)
        if not cmd:
            return jsonify({"error": "Parse failed"}), 500

        node.get_logger().info(f"🌐 Received via Flask: {english_cmd}")
        node.process_command(cmd)

        return jsonify({"status": "ok", "parsed": cmd})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def ros2_spin():
    """ROS2 노드 실행 루프"""
    global node
    rclpy.init()
    node = NaturalCommandNode()

    # spin_once → Flask랑 병렬로 실행
    while rclpy.ok():
        rclpy.spin_once(node, timeout_sec=0.1)

    node.destroy_node()
    rclpy.shutdown()


def main():
    # ROS2 실행 (백그라운드 스레드)
    ros_thread = threading.Thread(target=ros2_spin, daemon=True)
    ros_thread.start()

    # Flask 실행
    print("🚀 Bridge 서버 시작: http://0.0.0.0:6000/cmd")
    app.run(host="0.0.0.0", port=6000)


if __name__ == "__main__":
    main()
