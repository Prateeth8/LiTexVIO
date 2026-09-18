#!/usr/bin/env python3

import csv
import math
import os
import numpy as np

import rclpy
from rclpy.node import Node

from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu


class Task2Fusion(Node):

    def __init__(self):
        super().__init__('task2_fusion')

        self.sigma_v = 0.02
        self.sigma_g = 0.01
        self.sigma_b = 0.001

        self.x = np.zeros(4)
        self.P = np.diag([0.01, 0.01, 0.01, 0.001])

        self.initialized = False
        self.last_time = None
        self.latest_gyro = 0.0

        self.noisy_odom_initialized = False
        self.noisy_x = 0.0
        self.noisy_y = 0.0
        self.noisy_yaw = 0.0
        self.noisy_last_time = None

        self.log_dir = os.path.expanduser('~/corridor_ws/task2_logs')
        os.makedirs(self.log_dir, exist_ok=True)

        self.odom_file = open(os.path.join(self.log_dir, 'odom.csv'), 'w', newline='')
        self.noisy_file = open(os.path.join(self.log_dir, 'odom_noisy.csv'), 'w', newline='')
        self.ekf_file = open(os.path.join(self.log_dir, 'ekf.csv'), 'w', newline='')
        self.cov_file = open(os.path.join(self.log_dir, 'pose_covariance.csv'), 'w', newline='')

        self.odom_writer = csv.writer(self.odom_file)
        self.noisy_writer = csv.writer(self.noisy_file)
        self.ekf_writer = csv.writer(self.ekf_file)
        self.cov_writer = csv.writer(self.cov_file)

        self.odom_writer.writerow(['time', 'x', 'y', 'yaw'])
        self.noisy_writer.writerow(['time', 'x', 'y', 'yaw', 'velocity'])
        self.ekf_writer.writerow(['time', 'x', 'y', 'yaw', 'gyro_bias'])

        covariance_header = ['time']
        for i in range(4):
            for j in range(4):
                covariance_header.append(f'P_{i}{j}')
        self.cov_writer.writerow(covariance_header)

        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            20
        )
        self.noisy_odom_sub = self.create_subscription(
            Odometry,
            '/odom_noisy',
            self.noisy_odom_callback,
            20
        )
        self.imu_sub = self.create_subscription(
            Imu,
            '/imu/noisy',
            self.imu_callback,
            50
        )

        self.get_logger().info('Task 2 fusion node started')

    def quaternion_to_yaw(self, q):
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def imu_callback(self, msg):
        self.latest_gyro = msg.angular_velocity.z

    def odom_callback(self, msg):
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        yaw = self.quaternion_to_yaw(msg.pose.pose.orientation)

        self.odom_writer.writerow([t, x, y, yaw])
        self.odom_file.flush()

        if not self.initialized:
            self.x[0] = x
            self.x[1] = y
            self.x[2] = yaw
            self.x[3] = 0.0
            self.last_time = t
            self.initialized = True
            return

    def noisy_odom_callback(self, msg):
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        v = msg.twist.twist.linear.x

        if not self.noisy_odom_initialized:
            self.noisy_x = self.x[0]
            self.noisy_y = self.x[1]
            self.noisy_yaw = self.x[2]
            self.noisy_last_time = t
            self.noisy_odom_initialized = True
            return

        dt = t - self.noisy_last_time
        self.noisy_last_time = t

        if dt <= 0.0 or dt > 1.0:
            return

        self.noisy_x += v * math.cos(self.noisy_yaw) * dt
        self.noisy_y += v * math.sin(self.noisy_yaw) * dt

        self.noisy_writer.writerow([t, self.noisy_x, self.noisy_y, self.noisy_yaw, v])
        self.noisy_file.flush()

        if not self.initialized:
            return

        if self.last_time is None:
            self.last_time = t
            return

        dt_ekf = t - self.last_time
        self.last_time = t

        if dt_ekf <= 0.0 or dt_ekf > 1.0:
            return

        self.predict(v, self.latest_gyro, dt_ekf)

        self.ekf_writer.writerow([t, self.x[0], self.x[1], self.x[2], self.x[3]])
        self.ekf_file.flush()

        covariance_row = [t]
        for i in range(4):
            for j in range(4):
                covariance_row.append(self.P[i, j])

        self.cov_writer.writerow(covariance_row)
        self.cov_file.flush()

    def predict(self, v, omega, dt):
        yaw = self.x[2]
        bias = self.x[3]

        omega_corrected = omega - bias

        self.x[0] += v * math.cos(yaw) * dt
        self.x[1] += v * math.sin(yaw) * dt
        self.x[2] += omega_corrected * dt
        self.x[2] = math.atan2(math.sin(self.x[2]), math.cos(self.x[2]))

        F = np.eye(4)
        F[0, 2] = -v * math.sin(yaw) * dt
        F[1, 2] = v * math.cos(yaw) * dt
        F[2, 3] = -dt

        G = np.zeros((4, 3))
        G[0, 0] = math.cos(yaw) * dt
        G[1, 0] = math.sin(yaw) * dt
        G[2, 1] = dt
        G[3, 2] = math.sqrt(dt)

        Q = np.diag([self.sigma_v ** 2, self.sigma_g ** 2, self.sigma_b ** 2])

        self.P = F @ self.P @ F.T + G @ Q @ G.T
        self.P = (self.P + self.P.T) / 2.0

    def destroy_node(self):
        self.odom_file.close()
        self.noisy_file.close()
        self.ekf_file.close()
        self.cov_file.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = Task2Fusion()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()