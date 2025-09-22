import threading
from flask import Flask, request, jsonify
from flask_cors import CORS
import rclpy
from omx_f_natural_command_node_sim import NaturalCommandNode

# --------------------
# Flask setup
# --------------------
app = Flask(__name__)
CORS(app)

node = None   # Store ROS2 node instance


# --------------------
# Routes
# --------------------
@app.route('/cmd', methods=['POST'])
def cmd():
    """Voice server → Bridge: Receive already-parsed JSON and execute"""
    global node
    data = request.json or {}
    parsed = data.get("parsed", None)

    if not parsed:
        return jsonify({"error": "No parsed JSON provided"}), 400

    if node is None:
        return jsonify({"error": "ROS2 node not ready"}), 500

    try:
        node.get_logger().info(f"🌐 Received parsed command: {parsed}")
        node.process_command(parsed)
        return jsonify({"status": "ok", "parsed": parsed})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# --------------------
# ROS2 spin thread
# --------------------
def ros2_spin():
    """ROS2 node execution loop"""
    global node
    rclpy.init()
    node = NaturalCommandNode()

    while rclpy.ok():
        rclpy.spin_once(node, timeout_sec=0.1)

    node.destroy_node()
    rclpy.shutdown()


# --------------------
# Main
# --------------------
def main():
    ros_thread = threading.Thread(target=ros2_spin, daemon=True)
    ros_thread.start()

    print("=" * 60)
    print("🚀 Bridge server started (port 6000)")
    print("🌐 Endpoint: POST http://127.0.0.1:6000/cmd {parsed: {...}}")
    print("=" * 60)
    app.run(host="0.0.0.0", port=6000)


if __name__ == "__main__":
    main()
