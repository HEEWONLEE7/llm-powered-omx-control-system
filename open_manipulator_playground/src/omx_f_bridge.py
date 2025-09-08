import threading
from flask import Flask, request, jsonify
from flask_cors import CORS
import rclpy
from omx_f_natural_command_node import NaturalCommandNode

# Flask setup
app = Flask(__name__)
CORS(app)

node = None   # Store ROS2 node instance

@app.route('/cmd', methods=['POST'])
def cmd():
    """UI → Bridge: Receive English command and forward to ROS2 node"""
    global node
    data = request.json
    english_cmd = data.get("english", "")

    if not english_cmd:
        return jsonify({"error": "No english command"}), 400

    if node is None:
        return jsonify({"error": "ROS2 node not ready"}), 500

    try:
        # Convert command to JSON using LLM parser
        cmd = node.parse_command_with_llm(english_cmd)
        if not cmd:
            return jsonify({"error": "Parse failed"}), 500

        node.get_logger().info(f"🌐 Received via Flask: {english_cmd}")
        node.process_command(cmd)

        return jsonify({"status": "ok", "parsed": cmd})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def ros2_spin():
    """ROS2 node execution loop"""
    global node
    rclpy.init()
    node = NaturalCommandNode()

    # spin_once → Run in parallel with Flask
    while rclpy.ok():
        rclpy.spin_once(node, timeout_sec=0.1)

    node.destroy_node()
    rclpy.shutdown()


def main():
    # Run ROS2 in background thread
    ros_thread = threading.Thread(target=ros2_spin, daemon=True)
    ros_thread.start()

    # Run Flask
    print("🚀 Bridge server started: http://0.0.0.0:6000/cmd")
    app.run(host="0.0.0.0", port=6000)


if __name__ == "__main__":
    main()
