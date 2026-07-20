import time
import numpy as np
from rclpy.node import Node
from collections import deque
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from mlp_control_pkg.utils import loadModelFunc, createAcadosSolver

class CarControlNode(Node):
    def __init__(self, car_state_node, path_points_node, model_path, N_steps, dt):
        super().__init__('car_control_node')
        self.get_logger().info('Car Control Node has been started.')
        self.effort_pub = self.create_publisher(JointState, '/joint_command', 10)
        self.pred_path_pub = self.create_publisher(Float64MultiArray, "predicted_path", 10)

        self.car_state_node = car_state_node
        self.path_points_node = path_points_node

        self.model_path = model_path
        self.N_steps = N_steps
        self.dt = dt
        self.prev_u = np.zeros((2,))
        self._solving = False
        self._timer_period = 0.05  # 20 Hz
        self._cycle_count = 0

        self.tracking_errors = []   # list of (cross_track, along_track, heading_err)

        # LSTM Sequence Windows
        self.state_history = deque(maxlen=10)
        self.control_history = deque(maxlen=9)

        # MPC setup
        self._model_func, self._lib_dir, self._lib_name = loadModelFunc(self.model_path, self.dt)
        self._acados_solver = createAcadosSolver(self._model_func, self._lib_dir, self._lib_name, self.N_steps, self.dt)
        self.create_timer(self._timer_period, self.find_control_command)

    def _record_tracking_error(self, current_state_19, target_point):
        """Log this cycle's tracking error, in the same path-frame decomposition
        used by the cost function, so it matches what the MPC is optimizing."""
        x_t, y_t, sin_t, cos_t = target_point
        dx = current_state_19[0] - x_t
        dy = current_state_19[1] - y_t

        # CORRECTED for Convention B (Y-forward):
        along_track = dx * sin_t + dy * cos_t    # along path direction
        cross_track = -dx * cos_t + dy * sin_t   # perpendicular to path

        car_heading = np.arctan2(current_state_19[2], current_state_19[3])
        path_heading = np.arctan2(sin_t, cos_t)
        heading_err = np.arctan2(np.sin(car_heading - path_heading),
                                np.cos(car_heading - path_heading))  # wrapped

        self.tracking_errors.append((cross_track, along_track, heading_err))


    def report_tracking_error(self):
        """Call this once when shutting down to summarize tracking performance."""
        if not self.tracking_errors:
            self.get_logger().warn('No tracking error samples recorded.')
            return

        arr = np.array(self.tracking_errors)  # shape (N, 3)
        cross, along, heading = arr[:, 0], arr[:, 1], arr[:, 2]

        def stats(name, values):
            return (f'{name}: mean={np.mean(values):+.4f}  '
                    f'mean_abs={np.mean(np.abs(values)):.4f}  '
                    f'rmse={np.sqrt(np.mean(values**2)):.4f}  '
                    f'max_abs={np.max(np.abs(values)):.4f}')

        self.get_logger().info('===== Tracking Error Summary =====')
        self.get_logger().info(f'Samples: {len(self.tracking_errors)}')
        self.get_logger().info(stats('Cross-track (m)', cross))
        self.get_logger().info(stats('Along-track (m)', along))
        self.get_logger().info(stats('Heading (rad)', heading))
        self.get_logger().info('===================================')

        # optional: dump raw samples for offline plotting
        try:
            np.savetxt('/tmp/mpc_tracking_errors.csv', arr,
                        delimiter=',', header='cross_track,along_track,heading_err',
                        comments='')
            self.get_logger().info('Saved raw samples to /tmp/mpc_tracking_errors.csv')
        except Exception as e:
            self.get_logger().warn(f'Could not save CSV: {e}')

    def _absolute_path_angles(self, nearest_points, car_angle):
        """Convert each point's car-relative heading diff into an ABSOLUTE
        path heading, in the same convention as the rest of the pipeline.
        (See the long comment in _build_reference_path for why it's
        car_angle - diff rather than car_angle + diff.)
        """
        return [car_angle - point.angle for point in nearest_points]

    def _curvature_ahead(self, nearest_points, car_angle, min_arc_length=0.05):
        """Estimate how sharply the path ahead curves.

        Deliberately uses ALL of `nearest_points`, not just the first
        N_steps+1 that go into the solver -- if path_points_node ever
        publishes more points than the dynamics horizon needs, this
        function automatically gets a longer preview than the MPC stages
        themselves, which is the whole point (see build_reference_path).

        Returns an average turn-rate along the path ahead, in rad/m.
        ~0 for a straight road, large for a tight turn. Robust to
        duplicate/near-duplicate points (common near the end of a
        slow-moving path buffer) by skipping near-zero-length segments
        instead of dividing by them.
        """
        if nearest_points is None or len(nearest_points) < 2:
            return 0.0

        abs_angles = self._absolute_path_angles(nearest_points, car_angle)

        total_heading_change = 0.0
        total_arc_length = 0.0
        for i in range(1, len(nearest_points)):
            dx = nearest_points[i].x - nearest_points[i - 1].x
            dy = nearest_points[i].y - nearest_points[i - 1].y
            seg_len = np.hypot(dx, dy)
            if seg_len < 1e-6:
                continue  # stale/duplicate point, don't let it distort the average

            dtheta = abs_angles[i] - abs_angles[i - 1]
            dtheta = np.arctan2(np.sin(dtheta), np.cos(dtheta))  # wrap to [-pi, pi]

            total_heading_change += abs(dtheta)
            total_arc_length += seg_len

        if total_arc_length < min_arc_length:
            return 0.0  # not enough distance covered to trust the estimate

        return total_heading_change / total_arc_length

    def _build_reference_path(self, nearest_points, horizon_size, car_angle):
        # NearestPoint.angle is now the heading DIFFERENCE between the car and
        # the path at that point (not the path's own absolute tangent angle
        # anymore). The rest of the MPC (cost function target_path vs the
        # car's absolute sin/cos heading) still needs an ABSOLUTE reference
        # heading, in the SAME angle convention as the rest of the state
        # pipeline, so we reconstruct it here.
        #
        # IMPORTANT: the diff is computed upstream (temp_curve.py, in the
        # Isaac Sim graph) as `path_angle_true - car_angle_true` using the
        # car's TRUE yaw straight off the rotation matrix. But everywhere
        # else in this pipeline (car_state_node.compute_angle, training data,
        # current_state_19) the car's angle is the NEGATED yaw
        # (S = -T, "Flipped angle as per data script"). Working through the
        # algebra: T_path = diff + T_car = diff - S_car, so
        # S_path = -T_path = S_car - diff.
        # i.e. the target must be car_angle MINUS the diff, not plus.
        abs_angles = self._absolute_path_angles(nearest_points, car_angle)
 
        path = np.array([
            [point.x, point.y, np.sin(abs_angle), np.cos(abs_angle)]
            for point, abs_angle in zip(nearest_points, abs_angles)
        ], dtype=float)

        if path.shape[0] >= horizon_size:
            return path[:horizon_size]

        if len(nearest_points) >= 2:
            step_x = nearest_points[-1].x - nearest_points[-2].x
            step_y = nearest_points[-1].y - nearest_points[-2].y
            step = np.array([step_x, step_y, 0.0, 0.0], dtype=float)

            if np.linalg.norm([step_x, step_y]) < 1e-6:
                step = np.array([
                    np.cos(abs_angles[-1]) * 0.1,
                    np.sin(abs_angles[-1]) * 0.1,
                    0.0, 0.0
                ], dtype=float)
        else:
            step = np.array([
                np.cos(abs_angles[-1]) * 0.1,
                np.sin(abs_angles[-1]) * 0.1,
                0.0, 0.0
            ], dtype=float)

        while path.shape[0] < horizon_size:
            path = np.vstack([path, path[-1] + step])

        self.get_logger().info(f'Built reference path: {path}')
        return path

    def _speed_ref(self, curvature, v_max=1.5, k=2.0):
        """Map curvature (rad/m) to a target speed (m/s). Tune v_max/k to taste."""
        return v_max / (1.0 + k * curvature)

    def publish_control_command(self, control_input):
        self.get_logger().info(f'Publishing control command: {control_input}')

        msg = JointState()
        msg.name = ['front_left_joint', 'front_right_joint', 'rear_left_joint', 'rear_right_joint']
        
        # Raw assignment: The neural network was trained on raw efforts and implicitly knows the mapping!
        # msg.effort = control_input.tolist()
        msg.effort = [control_input[0], control_input[1], control_input[0], control_input[1]]
        self.effort_pub.publish(msg)

    def publish_predicted_path(self, predicted_path):
        msg = Float64MultiArray()
        pos = []
        vals = predicted_path.flatten().tolist()
        # The solver state is [x, y, sin(yaw), cos(yaw), vel_left, vel_right].
        # Only publish pose information to the path topic.
        for i in range(0, len(vals), 6):
            if i + 3 >= len(vals):
                break
            x, y, sin, cos = vals[i], vals[i+1], vals[i+2], vals[i+3]
            angle = np.arctan2(sin, cos)
            pos.append(x)
            pos.append(y)
            pos.append(angle)
            
        msg.data = pos[:9]  # Limit to first 9 states (3 poses) for visualization
        self.get_logger().info(f'Publishing predicted path: {msg.data}')
        self.pred_path_pub.publish(msg)

    def find_control_command(self):
        if self._solving:
            return

        self._solving = True
        cycle_start = time.perf_counter()

        current_state_19 = self.car_state_node.get_car_state()
        if current_state_19 is None:
            self.get_logger().warn("Current car state is not available. Skipping control computation.")
            self._solving = False
            return

        nearest_points = self.path_points_node.nearest_points
        if nearest_points is None or len(nearest_points) == 0:
            self.get_logger().warn("No nearest path points available. Skipping control computation.")
            self._solving = False
            return

        # Initialize sequence history on first frame
        if len(self.state_history) == 0:
            for _ in range(10):
                self.state_history.append(current_state_19)
            for _ in range(9):
                self.control_history.append(np.zeros(2))
        else:
            self.state_history.append(current_state_19)

        self.get_logger().info(f'Current state (19D): {current_state_19}')
        self.get_logger().info(f'nearest_points: {[ (p.x, p.y, p.angle) for p in nearest_points ]}')

        car_angle = np.arctan2(current_state_19[2], current_state_19[3])
        curvature = self._curvature_ahead(nearest_points, car_angle)
        v_ref = self._speed_ref(curvature)
        path_points = self._build_reference_path(nearest_points, self.N_steps + 1, car_angle)
        self._record_tracking_error(current_state_19, path_points[0])
        
        # Assemble the 106-dimensional augmented state vector for x0
        augmented_x0 = np.concatenate([
            np.concatenate(list(self.state_history)),  # 10 * 7 = 70 states
            np.concatenate(list(self.control_history)) # 9 * 2 = 18 controls
        ])
        
        self._acados_solver.set(0, "lbx", augmented_x0)
        self._acados_solver.set(0, "ubx", augmented_x0)
        
        # Warm start
        if self._cycle_count == 0:
            for t in range(self.N_steps + 1):
                self._acados_solver.set(t, "x", augmented_x0)
            for t in range(self.N_steps):
                self._acados_solver.set(t, "u", self.prev_u)
        else:
            # RTI needs a coherent warm start across the WHOLE horizon, not
            # just stage 0, or successive solves can disagree wildly on
            # direction and chatter between cycles. Shift last cycle's
            # solution forward by one step (repeat the final stage to fill
            # the new last slot).
            for t in range(self.N_steps):
                self._acados_solver.set(t, "x", self._prev_x_traj[t + 1])
                u_idx = min(t + 1, self.N_steps - 1)
                self._acados_solver.set(t, "u", self._prev_u_traj[u_idx])
            self._acados_solver.set(self.N_steps, "x", self._prev_x_traj[self.N_steps])

        # Set target path parameters across the horizon (Only 4 variables now!)
        for t in range(self.N_steps + 1):
            self._acados_solver.set(t, "p", np.append(path_points[t, :], v_ref))

        try:
            status = self._acados_solver.solve()
            if status != 0:
                self.get_logger().error(f'Acados solver failed with status {status}')
                self._solving = False
                return

            control_input = self._acados_solver.get(0, "u") # get optimal control at time 0
            self.prev_u = control_input  # store for warm start
            self.control_history.append(control_input) # Shift the historic control window

            # Save the full trajectory so next cycle can warm-start from a
            # coherent shifted version of it (see Warm start block above).
            self._prev_x_traj = [self._acados_solver.get(t, "x") for t in range(self.N_steps + 1)]
            self._prev_u_traj = [self._acados_solver.get(t, "u") for t in range(self.N_steps)]

            # Extract predicted paths (Must extract from indices 171:190 of augmented state)
            predict_path = []
            for t in range(self.N_steps + 1):
                latest_aug = self._acados_solver.get(t, "x")
                latest_19 = latest_aug[171:190] # The current physical state in the window
                predict_path.append(latest_19[:6])
            predict_path = np.array(predict_path)
            
            self.publish_control_command(control_input)
            self.publish_predicted_path(predict_path)
            self._cycle_count += 1

            elapsed_ms = (time.perf_counter() - cycle_start) * 1000.0
            if elapsed_ms > 50.0:
                self.get_logger().warn(f'MPC cycle over budget: {elapsed_ms:.1f} ms')

        except Exception as e:
            self.get_logger().error(f'MPC solver failed: {e}')
        finally:
            self._solving = False