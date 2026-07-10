import numpy as np
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray

class PathPointsNode(Node):
    def __init__(self):
        super().__init__('path_points_node')
        self.subscription = self.create_subscription(
            Float32MultiArray,
            '/nearest_curve_point',
            self.path_points_callback,
            10)
        self.subscription  # prevent unused variable warning

        self.nearest_points = None  # Will hold the nearest path points
        self.get_logger().info('PathPointsNode has been started.')

    def path_points_callback(self, msg):
        # Store the received path point
        data = msg.data # [idx, x, y, angle]
        if len(data) >= 31:  # Ensure we have enough data for 10 points (10 * 3 + 1 for idx)
            for i in range(0, 10):
                setattr(self, f'point_{i}', NearestPoint(x=data[1 + i*3], y=data[2 + i*3], angle=data[3 + i*3]))
            self.nearest_points = [getattr(self, f'point_{i}') for i in range(10)]
        else:
            self.nearest_points = None
        # self.get_logger().info(f'Received Nearest Path Point: {self.nearest_point}')

class NearestPoint:
    def __init__(self, x=0.0, y=0.0, angle=0.0):
        self.x = x
        self.y = y
        self.angle = np.radians(angle)
        # self.angle = self.wrap_angle(angle)  # Ensure angle is wrapped to [-180, 180]
        # self.angle = -angle  # Store the angle as is, without wrapping

    def wrap_angle(self, angle):
        """Wraps the angle to the range [-180, 180]."""
        angle = angle - 180  # Adjust for the car_state_node's negated yaw
        while angle > 180:
            angle -= 2 * 180
        while angle < -180:
            angle += 2 * 180
        rad_angle = np.radians(angle)  # Convert to radians
        return rad_angle  # Return the wrapped angle in radians