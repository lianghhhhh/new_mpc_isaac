import numpy as np
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import JointState

def compute_angle(qx, qy, qz, qw):
    """Yaw angle from quaternion, inverted to match data collection."""
    siny_cosp = 2 * (qw * qz + qx * qy)
    cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return -yaw  # Flipped angle as per data script

class CarStateNode(Node):
    def __init__(self):
        super().__init__('car_state_node')

        self.joint_subscriber = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )

        self.odom_subscriber = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10
        )

    def joint_state_callback(self, msg):
        # self.get_logger().info(f"Received joint states: {msg}")
        self.latest_joint_state = msg

    def odom_callback(self, msg):
        # self.get_logger().info(f"Received odometry data: {msg}")
        self.latest_odom = msg

    def get_car_state(self):
        """Returns the full 19-dimensional absolute state vector for the LSTM model."""
        if hasattr(self, 'latest_joint_state') and hasattr(self, 'latest_odom'):
            names = self.latest_joint_state.name
            positions = self.latest_joint_state.position
            velocities = self.latest_joint_state.velocity
            
            try:
                fl_idx = names.index('front_left_joint')
                fr_idx = names.index('front_right_joint')
                rl_idx = names.index('rear_left_joint')
                rr_idx = names.index('rear_right_joint')
            except ValueError:
                return None
            
            # Position & Orientation
            pos_x = self.latest_odom.pose.pose.position.x
            pos_y = self.latest_odom.pose.pose.position.y
            qx = self.latest_odom.pose.pose.orientation.x
            qy = self.latest_odom.pose.pose.orientation.y
            qz = self.latest_odom.pose.pose.orientation.z
            qw = self.latest_odom.pose.pose.orientation.w
            car_angle = compute_angle(qx, qy, qz, qw)
            
            # Velocities (flipped to match data script)
            vel_x = -self.latest_odom.twist.twist.linear.x
            vel_y = -self.latest_odom.twist.twist.linear.y
            ang_z = -self.latest_odom.twist.twist.angular.z

            # Assemble the 19-dimensional array
            state_19 = np.array([
                pos_x, pos_y,
                np.sin(car_angle), np.cos(car_angle),
                vel_x, vel_y, ang_z,
                np.sin(positions[fl_idx]), np.cos(positions[fl_idx]),
                np.sin(positions[fr_idx]), np.cos(positions[fr_idx]),
                np.sin(positions[rl_idx]), np.cos(positions[rl_idx]),
                np.sin(positions[rr_idx]), np.cos(positions[rr_idx]),
                velocities[fl_idx], velocities[fr_idx],
                velocities[rl_idx], velocities[rr_idx]
            ], dtype=float)

            return state_19
            
        return None