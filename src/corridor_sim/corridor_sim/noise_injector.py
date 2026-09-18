#!/usr/bin/env python3

import copy
import math
import numpy as np

import rclpy
from rclpy.node import Node

from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu


class NoiseInjector(Node):

    def __init__(self):
        super().__init__('noise_injector')

        self.declare_parameter('sigma_v', 0.02)
        self.declare_parameter('sigma_g', 0.01)
        self.declare_parameter('sigma_b', 0.001)

        self.declare_parameter('slip_scale', 0.70)
        self.declare_parameter('slip_x_min', 10.0)
        self.declare_parameter('slip_x_max', 15.0)

        self.sigma_v = self.get_parameter('sigma_v').value
        self.sigma_g = self.get_parameter('sigma_g').value
        self.sigma_b = self.get_parameter('sigma_b').value

        self.slip_scale = self.get_parameter('slip_scale').value
        self.slip_x_min = self.get_parameter('slip_x_min').value
        self.slip_x_max = self.get_parameter('slip_x_max').value

        self.gyro_bias = np.zeros(3)
        self.last_imu_time = None
        self.previous_slip_state = False

        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10
        )
        self.imu_sub = self.create_subscription(
            Imu,
            '/imu/data',
            self.imu_callback,
            50
        )

        self.odom_pub = self.create_publisher(
            Odometry,
            '/odom_noisy',
            10
        )
        self.imu_pub = self.create_publisher(
            Imu,
            '/imu/noisy',
            50
        )

        self.get_logger().info('Noise injector started')
        self.get_logger().info(f'Odometry noise sigma_v = {self.sigma_v}')
        self.get_logger().info(f'Gyro noise sigma_g = {self.sigma_g}')
        self.get_logger().info(f'Gyro bias random walk sigma_b = {self.sigma_b}')
        self.get_logger().info(
            f'Slip region: X = [{self.slip_x_min}, {self.slip_x_max}] m, scale = {self.slip_scale}'
        )

    def odom_callback(self, msg):
        noisy_msg = copy.deepcopy(msg)

        x = msg.pose.pose.position.x
        v_true = msg.twist.twist.linear.x

        if self.slip_x_min <= x <= self.slip_x_max:
            scale = self.slip_scale
            slipping = True
        else:
            scale = 1.0
            slipping = False

        white_noise = np.random.normal(0.0, self.sigma_v)
        v_recorded = scale * v_true + white_noise

        noisy_msg.twist.twist.linear.x = float(v_recorded)
        noisy_msg.twist.covariance[0] = self.sigma_v ** 2

        if slipping and not self.previous_slip_state:
            self.get_logger().warn(f'SLIP ACTIVE: X={x:.3f} m, scale={scale:.2f}')
        elif not slipping and self.previous_slip_state:
            self.get_logger().info(f'SLIP ENDED: X={x:.3f} m')

        self.previous_slip_state = slipping
        self.odom_pub.publish(noisy_msg)

    def imu_callback(self, msg):
        noisy_msg = copy.deepcopy(msg)

        current_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        if self.last_imu_time is None:
            dt = 0.01
        else:
            dt = current_time - self.last_imu_time
            if dt <= 0.0 or dt > 1.0:
                dt = 0.01

        self.last_imu_time = current_time

        bias_increment = self.sigma_b * math.sqrt(dt) * np.random.randn(3)
        self.gyro_bias += bias_increment

        gyro_noise = self.sigma_g * np.random.randn(3)

        noisy_msg.angular_velocity.x = float(msg.angular_velocity.x + self.gyro_bias[0] + gyro_noise[0])
        noisy_msg.angular_velocity.y = float(msg.angular_velocity.y + self.gyro_bias[1] + gyro_noise[1])
        noisy_msg.angular_velocity.z = float(msg.angular_velocity.z + self.gyro_bias[2] + gyro_noise[2])

        noisy_msg.angular_velocity_covariance[0] = self.sigma_g ** 2
        noisy_msg.angular_velocity_covariance[4] = self.sigma_g ** 2
        noisy_msg.angular_velocity_covariance[8] = self.sigma_g ** 2

        self.imu_pub.publish(noisy_msg)


def main(args=None):
    rclpy.init(args=args)
    node = NoiseInjector()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()