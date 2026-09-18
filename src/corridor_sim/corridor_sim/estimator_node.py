#!/usr/bin/env python3

import os
import csv
import math
import copy

import numpy as np

import rclpy
from rclpy.node import Node

from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu, PointCloud2
from geometry_msgs.msg import PoseStamped

from sensor_msgs_py import point_cloud2

from scipy.spatial import cKDTree


def normalize_angle(a):
    return math.atan2(math.sin(a), math.cos(a))


def pose_to_xyyaw(msg):
    x = msg.pose.pose.position.x
    y = msg.pose.pose.position.y

    q = msg.pose.pose.orientation

    # yaw from quaternion
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)

    yaw = math.atan2(siny_cosp, cosy_cosp)

    return np.array([x, y, yaw])


class EstimatorNode(Node):

    def __init__(self):

        super().__init__('task2_estimator')

        ## parameters 

        self.declare_parameter('sigma_v', 0.02)
        self.declare_parameter('sigma_g', 0.01)
        self.declare_parameter('sigma_b', 0.001)

        self.declare_parameter(
            'degeneracy_threshold',
            1.0e-3
        )

        self.declare_parameter(
            'log_dir',
            os.path.expanduser(
                '~/corridor_ws/task2_logs'
            )
        )

        self.sigma_v = float(
            self.get_parameter('sigma_v').value
        )

        self.sigma_g = float(
            self.get_parameter('sigma_g').value
        )

        self.sigma_b = float(
            self.get_parameter('sigma_b').value
        )

        self.degeneracy_threshold = float(
            self.get_parameter(
                'degeneracy_threshold'
            ).value
        )

        self.log_dir = os.path.expanduser(
            self.get_parameter('log_dir').value
        )

        os.makedirs(
            self.log_dir,
            exist_ok=True
        )

        ## EKF state parameters

        self.x = np.zeros(4)

        self.P = np.diag([
            0.01,
            0.01,
            0.01,
            0.001
        ])

        self.last_odom_time = None
        self.last_imu_time = None

        self.latest_v = 0.0
        self.latest_gyro_z = 0.0

        ## Scan matching

        self.previous_scan = None
        self.previous_scan_time = None

        self.previous_estimate = None

        self.degenerate = False

        ## ROS publishers

        self.pose_pub = self.create_publisher(
            PoseStamped,
            '/fused_pose',
            10
        )

        ## ROS subscribers

        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom_noisy',
            self.odom_callback,
            20
        )

        self.imu_sub = self.create_subscription(
            Imu,
            '/imu/noisy',
            self.imu_callback,
            50
        )

        self.scan_sub = self.create_subscription(
            PointCloud2,
            '/points',
            self.scan_callback,
            5
        )

        ## Logging

        self.csv_path = os.path.join(
            self.log_dir,
            'task2_estimator.csv'
        )

        self.warning_path = os.path.join(
            self.log_dir,
            'degeneracy_warnings.log'
        )

        self.csv_file = open(
            self.csv_path,
            'w',
            newline=''
        )

        self.csv_writer = csv.writer(
            self.csv_file
        )

        self.csv_writer.writerow([
            'time',
            'x',
            'y',
            'yaw',
            'bias_z',
            'P_xx',
            'P_yy',
            'P_yaw',
            'P_bias',
            'info_eig_0',
            'info_eig_1',
            'info_eig_2',
            'min_info_eigenvalue'
        ])

        self.warning_file = open(
            self.warning_path,
            'w'
        )

        self.get_logger().info(
            'Task 2 estimator started'
        )

    ## Odometry
    def odom_callback(self, msg):

        t = (
            msg.header.stamp.sec
            +
            msg.header.stamp.nanosec * 1e-9
        )

        v = msg.twist.twist.linear.x

        self.latest_v = v

        if self.last_odom_time is None:

            self.last_odom_time = t

            return

        dt = t - self.last_odom_time

        self.last_odom_time = t

        if dt <= 0.0 or dt > 1.0:
            return

        self.predict(
            v,
            self.latest_gyro_z,
            dt
        )

    ## IMU
    def imu_callback(self, msg):

        self.latest_gyro_z = (
            msg.angular_velocity.z
        )

    ## EKF Prediction

    def predict(self, v, omega, dt):

        px = self.x[0]
        py = self.x[1]
        yaw = self.x[2]
        bias = self.x[3]

        # Correct gyro measurement using estimated bias
        omega_corrected = omega - bias

        ## State propagation

        self.x[0] += (
            v * math.cos(yaw) * dt
        )

        self.x[1] += (
            v * math.sin(yaw) * dt
        )

        self.x[2] += (
            omega_corrected * dt
        )

        self.x[2] = normalize_angle(
            self.x[2]
        )

        # bias remains unchanged during prediction

        ## State Jacobian

        F = np.eye(4)

        F[0, 2] = (
            -v * math.sin(yaw) * dt
        )

        F[1, 2] = (
            v * math.cos(yaw) * dt
        )

        F[2, 3] = -dt

        ## Process noise
        #
        # Inputs:
        # velocity noise
        # gyro noise
        # bias random walk

        G = np.zeros((4, 3))

        G[0, 0] = (
            math.cos(yaw) * dt
        )

        G[1, 0] = (
            math.sin(yaw) * dt
        )

        G[2, 1] = dt

        G[3, 2] = math.sqrt(dt)

        Q = np.diag([
            self.sigma_v ** 2,
            self.sigma_g ** 2,
            self.sigma_b ** 2
        ])

        self.P = (
            F @ self.P @ F.T
            +
            G @ Q @ G.T
        )

    ## Point cloud processing

    def extract_scan(self, msg):

        points = []

        for p in point_cloud2.read_points(
            msg,
            field_names=('x', 'y', 'z'),
            skip_nans=True
        ):

            x, y, z = p

            # Ignore points too close to robot
            r = math.sqrt(
                x*x + y*y
            )

            if r < 0.2:
                continue

            # Keep points in useful horizontal range
            if r > 20.0:
                continue

            points.append([
                x,
                y
            ])

        if len(points) < 50:
            return None

        points = np.asarray(
            points,
            dtype=np.float64
        )

        # Downsample for CPU operation
        if len(points) > 1000:

            indices = np.linspace(
                0,
                len(points) - 1,
                1000
            ).astype(int)

            points = points[indices]

        return points

    ## Scan matching

    def scan_callback(self, msg):

        scan = self.extract_scan(msg)

        if scan is None:
            return

        current_time = (
            msg.header.stamp.sec
            +
            msg.header.stamp.nanosec * 1e-9
        )

        if self.previous_scan is None:

            self.previous_scan = scan
            self.previous_scan_time = current_time

            self.previous_estimate = (
                self.x.copy()
            )

            return

        result = self.icp_point_to_plane(
            self.previous_scan,
            scan
        )

        if result is None:

            self.previous_scan = scan
            return

        relative_pose, H, residual = result

        ## Pose to global

        previous_pose = (
            self.previous_estimate
        )

        px = previous_pose[0]
        py = previous_pose[1]
        yaw = previous_pose[2]

        dx = relative_pose[0]
        dy = relative_pose[1]
        dyaw = relative_pose[2]

        c = math.cos(yaw)
        s = math.sin(yaw)

        global_dx = (
            c * dx - s * dy
        )

        global_dy = (
            s * dx + c * dy
        )

        z = np.array([
            px + global_dx,
            py + global_dy,
            normalize_angle(
                yaw + dyaw
            )
        ])

        ## Information matrix eigenvalues

        eigvals = np.linalg.eigvalsh(
            H
        )

        eigvals = np.sort(
            np.maximum(eigvals, 0.0)
        )

        min_eig = eigvals[0]

        ## Degeneracy detection

        currently_degenerate = (
            min_eig
            <
            self.degeneracy_threshold
        )

        if (
            currently_degenerate
            and not self.degenerate
        ):

            warning = (
                f'LOCALIZATION_DEGENERACY_WARNING '
                f't={current_time:.3f} '
                f'min_eigenvalue={min_eig:.6e}'
            )

            self.get_logger().warn(
                warning
            )

            self.warning_file.write(
                warning + '\n'
            )

            self.warning_file.flush()

        self.degenerate = (
            currently_degenerate
        )

        ## Information -> covariance
        

        H_reg = H + np.eye(3) * 1e-6

        try:

            R_relative = np.linalg.inv(
                H_reg
            )

        except np.linalg.LinAlgError:

            R_relative = (
                np.eye(3) * 1.0
            )

        # Scale covariance according to residual
        if len(residual) > 0:

            variance = np.mean(
                residual ** 2
            )

            R_relative *= max(
                variance,
                1e-4
            )

        ## EKF measurement
        #
        # z = [x, y, yaw]
        

        Hk = np.zeros(
            (3, 4)
        )

        Hk[0, 0] = 1.0
        Hk[1, 1] = 1.0
        Hk[2, 2] = 1.0

        innovation = (
            z
            -
            self.x[:3]
        )

        innovation[2] = normalize_angle(
            innovation[2]
        )

        S = (
            Hk @ self.P @ Hk.T
            +
            R_relative
        )

        try:

            K = (
                self.P
                @ Hk.T
                @ np.linalg.inv(S)
            )

        except np.linalg.LinAlgError:

            K = np.zeros(
                (4, 3)
            )

        self.x = (
            self.x
            +
            K @ innovation
        )

        self.x[2] = normalize_angle(
            self.x[2]
        )

        I = np.eye(4)

        self.P = (
            (I - K @ Hk)
            @ self.P
        )

        # Numerical symmetry
        self.P = (
            self.P + self.P.T
        ) * 0.5

        ## Publish
        

        self.publish_pose(
            msg,
            self.x
        )

        ## Logging

        self.csv_writer.writerow([
            current_time,
            self.x[0],
            self.x[1],
            self.x[2],
            self.x[3],
            self.P[0, 0],
            self.P[1, 1],
            self.P[2, 2],
            self.P[3, 3],
            eigvals[0],
            eigvals[1],
            eigvals[2],
            min_eig
        ])

        self.csv_file.flush()

        self.previous_scan = scan
        self.previous_scan_time = current_time
        self.previous_estimate = self.x.copy()

    ## POINT-TO-PLANE ICP
    

    def icp_point_to_plane(
        self,
        target,
        source
    ):

        # Initial transform
        theta = 0.0
        tx = 0.0
        ty = 0.0

        max_iterations = 8

        final_H = None
        final_residual = None

        for _ in range(max_iterations):

            c = math.cos(theta)
            s = math.sin(theta)

            R = np.array([
                [c, -s],
                [s,  c]
            ])

            transformed = (
                source @ R.T
                +
                np.array([tx, ty])
            )

            tree = cKDTree(
                target
            )

            distances, indices = tree.query(
                transformed,
                k=1
            )

            valid = distances < 0.5

            if np.count_nonzero(valid) < 30:
                return None

            src = transformed[valid]
            tgt = target[indices[valid]]

            ## PCA based normals
            normals = []

            for p in tgt:

                _, nn = tree.query(
                    p,
                    k=8
                )

                neighbors = target[nn]

                centered = (
                    neighbors
                    -
                    np.mean(
                        neighbors,
                        axis=0
                    )
                )

                covariance = (
                    centered.T
                    @ centered
                )

                eigvals, eigvecs = (
                    np.linalg.eigh(
                        covariance
                    )
                )

                normal = eigvecs[:, 0]

                normal /= (
                    np.linalg.norm(
                        normal
                    ) + 1e-12
                )

                normals.append(normal)

            normals = np.asarray(
                normals
            )

            ## Point to plane residuals and Jacobian

            diff = src - tgt

            residual = np.sum(
                normals * diff,
                axis=1
            )

            J = np.zeros(
                (len(src), 3)
            )

            J[:, 0] = normals[:, 0]
            J[:, 1] = normals[:, 1]

            J[:, 2] = (
                normals[:, 0] * (-src[:, 1])
                +
                normals[:, 1] * src[:, 0]
            )

            H = J.T @ J

            b = J.T @ residual

            try:

                delta = np.linalg.solve(
                    H + np.eye(3) * 1e-6,
                    -b
                )

            except np.linalg.LinAlgError:

                break

            tx += delta[0]
            ty += delta[1]
            theta += delta[2]

            if np.linalg.norm(
                delta
            ) < 1e-5:

                break

            final_H = H
            final_residual = residual

        if final_H is None:
            return None

        # Transform source -> target.
        # Robot motion is inverse of this transform.

        c = math.cos(theta)
        s = math.sin(theta)

        dx = -(
            c * tx + s * ty
        )

        dy = (
            s * tx - c * ty
        )

        dtheta = -theta

        relative_pose = np.array([
            dx,
            dy,
            dtheta
        ])

        return (
            relative_pose,
            final_H,
            final_residual
        )

    ## Publish the pose
    def publish_pose(
        self,
        reference_msg,
        state
    ):

        msg = PoseStamped()

        msg.header = (
            reference_msg.header
        )

        msg.header.frame_id = (
            'base_footprint'
        )

        msg.pose.position.x = float(
            state[0]
        )

        msg.pose.position.y = float(
            state[1]
        )

        msg.pose.position.z = 0.0

        yaw = state[2]

        msg.pose.orientation.x = 0.0
        msg.pose.orientation.y = 0.0
        msg.pose.orientation.z = math.sin(
            yaw / 2.0
        )
        msg.pose.orientation.w = math.cos(
            yaw / 2.0
        )

        self.pose_pub.publish(
            msg
        )

    # =============================================================
    # SHUTDOWN
    # =============================================================

    def destroy_node(self):

        try:
            self.csv_file.close()
            self.warning_file.close()
        except Exception:
            pass

        super().destroy_node()


def main(args=None):

    rclpy.init(args=args)

    node = EstimatorNode()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
