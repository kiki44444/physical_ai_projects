import sys
import termios
import tty
import select
import time
import rclpy
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool

STEP = 0.01  # meters per key press
KEYS = {
    'w': (STEP, 0, 0), 's': (-STEP, 0, 0),
    'a': (0, STEP, 0), 'd': (0, -STEP, 0),
    'q': (0, 0, STEP), 'e': (0, 0, -STEP),
}


def main():
    if not sys.stdin.isatty():
        raise RuntimeError("Run this script in an interactive SSH terminal")
    rclpy.init()
    node = rclpy.create_node('franka_keyboard_control')
    pub_move = node.create_publisher(Twist, '/ee_delta', 10)
    pub_gripper = node.create_publisher(Bool, '/gripper_command', 10)
    old = termios.tcgetattr(sys.stdin.fileno())
    print('W/S: X +/- | A/D: Y +/- | Q/E: Z +/- | O: open | C: close | X: exit', flush=True)
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
                pub_move.publish(msg)
                print(f'key={key}, delta={KEYS[key]}', flush=True)
            elif key in ('o', 'c'):
                msg = Bool()
                msg.data = (key == 'o')
                pub_gripper.publish(msg)
                print('gripper:', 'open' if msg.data else 'close', flush=True)
    finally:
        termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, old)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
