import sys
import termios
import tty
import select
import rclpy
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool, String

STEP = 0.01
KEYS = {
    'w': (STEP, 0.0, 0.0), 's': (-STEP, 0.0, 0.0),
    'a': (0.0, STEP, 0.0), 'd': (0.0, -STEP, 0.0),
    'q': (0.0, 0.0, STEP), 'e': (0.0, 0.0, -STEP),
}


def main():
    if not sys.stdin.isatty():
        raise RuntimeError("Use an interactive SSH terminal")
    rclpy.init()
    node = rclpy.create_node('franka_keyboard_control')
    move_pub = node.create_publisher(Twist, '/ee_delta', 10)
    grip_pub = node.create_publisher(Bool, '/gripper_command', 10)
    demo_pub = node.create_publisher(String, '/demo_command', 10)
    old = termios.tcgetattr(sys.stdin.fileno())
    print('W/S X | A/D Y | Q/E Z | O open | C close | R record | P save | N reset | X exit', flush=True)
    try:
        tty.setcbreak(sys.stdin.fileno())
        while rclpy.ok():
            readable, _, _ = select.select([sys.stdin], [], [], 0.1)
            if not readable:
                continue
            key = sys.stdin.read(1).lower()
            if key == 'x':
                break
            if key in KEYS:
                msg = Twist()
                msg.linear.x, msg.linear.y, msg.linear.z = map(float, KEYS[key])
                move_pub.publish(msg)
                print(f'move {key}', flush=True)
            elif key in ('o', 'c'):
                msg = Bool()
                msg.data = key == 'o'
                grip_pub.publish(msg)
                print('gripper', key, flush=True)
            elif key in ('r', 'p', 'n'):
                msg = String()
                msg.data = key
                demo_pub.publish(msg)
                print('demo', key, flush=True)
    finally:
        termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, old)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
