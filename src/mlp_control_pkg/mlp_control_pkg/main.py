import rclpy
from mlp_control_pkg.utils import loadConfig
from rclpy.executors import MultiThreadedExecutor
from mlp_control_pkg.car_state_node import CarStateNode
from mlp_control_pkg.car_control_node import CarControlNode
from mlp_control_pkg.path_points_node import PathPointsNode

def main():
    rclpy.init()

    config = loadConfig()

    car_state_node = CarStateNode()
    path_points_node = PathPointsNode()
    car_control_node = CarControlNode(car_state_node, path_points_node,
                                      config['model_path'], config['N_steps'], config['dt'])

    executor = MultiThreadedExecutor()
    executor.add_node(car_state_node)
    executor.add_node(path_points_node)
    executor.add_node(car_control_node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()
