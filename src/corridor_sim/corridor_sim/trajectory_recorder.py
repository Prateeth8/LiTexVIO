#!/usr/bin/env python3

import csv
import math
import os

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry, Path


class TrajectoryRecorder(Node):

    def __init__(self):
        super().__init__('trajectory_recorder')

        self.gt_x = None
        self.gt_y = None
        self.gt_yaw = None
        self.gt_last_time = None

        self.noisy_x = None
        self.noisy_y = None
        self.noisy_yaw = 0.0
        self.noisy_last_time = None

        self.gt_sub = self.create_subscription(
            Odometry,
            '/odom',
            self.gt_callback,
            20
        )
        self.noisy_sub = self.create_subscription(
            Odometry,
            '/odom_noisy',
            self.noisy_callback,
            20
        )

        self.gt_path_pub = self.create_publisher(Path, '/ground_truth_path', 10)
        self.noisy_path_pub = self.create_publisher(Path, '/noisy_odom_path', 10)

        self.gt_path = Path()
        self.gt_path.header.frame_id = 'odom'

        self.noisy_path = Path()
        self.noisy_path.header.frame_id = 'odom'

        self.log_dir = os.path.expanduser('~/corridor_ws/task2_logs')
        os.makedirs(self.log_dir, exist_ok=True)

        self.gt_file = open(os.path.join(self.log_dir, 'ground_truth.csv'), 'w', newline='')
        self.noisy_file = open(os.path.join(self.log_dir, 'noisy_odometry.csv'), 'w', newline='')

        self.gt_writer = csv.writer(self.gt_file)
        self.noisy_writer = csv.writer(self.noisy_file)

        self.gt_writer.writerow(['time', 'x', 'y', 'yaw'])
        self.noisy_writer.writerow(['time', 'x', 'y', 'yaw', 'velocity'])

        self.get_logger().info('Trajectory recorder started')

    def quaternion_to_yaw(self, q):
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def gt_callback(self, msg):
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        yaw = self.quaternion_to_yaw(msg.pose.pose.orientation)

        self.gt_x = x
        self.gt_y = y
        self.gt_yaw = yaw

        self.gt_writer.writerow([t, x, y, yaw])
        self.gt_file.flush()

        pose = PoseStamped()
        pose.header = msg.header
        pose.header.frame_id = 'odom'
        pose.pose = msg.pose.pose

        self.gt_path.header.stamp = msg.header.stamp
        self.gt_path.poses.append(pose)
        self.gt_path_pub.publish(self.gt_path)

    def noisy_callback(self, msg):
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        velocity = msg.twist.twist.linear.x

        if self.noisy_last_time is None:
            self.noisy_last_time = t
            if self.gt_x is not None:
                self.noisy_x = self.gt_x
                self.noisy_y = self.gt_y
            else:
                self.noisy_x = 0.0
                self.noisy_y = 0.0
            return

        dt = t - self.noisy_last_time
        self.noisy_last_time = t

        if dt <= 0.0 or dt > 1.0:
            return

        self.noisy_x += velocity * math.cos(self.noisy_yaw) * dt
        self.noisy_y += velocity * math.sin(self.noisy_yaw) * dt

        self.noisy_writer.writerow([t, self.noisy_x, self.noisy_y, self.noisy_yaw, velocity])
        self.noisy_file.flush()

        pose = PoseStamped()
        pose.header.stamp = msg.header.stamp
        pose.header.frame_id = 'odom'
        pose.pose.position.x = self.noisy_x
        pose.pose.position.y = self.noisy_y
        pose.pose.position.z = 0.0
        pose.pose.orientation.w = 1.0

        self.noisy_path.header.stamp = msg.header.stamp
        self.noisy_path.poses.append(pose)
        self.noisy_path_pub.publish(self.noisy_path)

    def destroy_node(self):
        self.gt_file.close()
        self.noisy_file.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = TrajectoryRecorder()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()