#!/usr/bin/env python3

import os
import csv
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2


class PointCloudRecorder(Node):

    def __init__(self):
        super().__init__('pointcloud_recorder')

        self.output_dir = os.path.expanduser(
            '~/corridor_ws/task2_data/B_noisy_add'
        )

        os.makedirs(self.output_dir, exist_ok=True)

        self.scan_count = 0

        self.timestamp_file = os.path.join(
            self.output_dir,
            'timestamps.csv'
        )

        self.csv_file = open(
            self.timestamp_file,
            'w',
            newline=''
        )

        self.csv_writer = csv.writer(self.csv_file)

        self.csv_writer.writerow([
            'scan_id',
            'sec',
            'nanosec'
        ])

        self.subscription = self.create_subscription(
            PointCloud2,
            '/points',
            self.pointcloud_callback,
            10
        )

        self.get_logger().info(
            f'Recording point clouds to: {self.output_dir}'
        )

    def pointcloud_callback(self, msg):

        points_structured = point_cloud2.read_points(
            msg,
            field_names=('x', 'y', 'z'),
            skip_nans=True
        )

        points = np.column_stack((
            points_structured['x'],
            points_structured['y'],
            points_structured['z']
        )).astype(np.float32)

        if points.shape[0] == 0:
            self.get_logger().warn(
                'Received empty point cloud'
            )
            return

        self.scan_count += 1

        filename = os.path.join(
            self.output_dir,
            f'scan_{self.scan_count:06d}.npy'
        )

        np.save(filename, points)

        self.csv_writer.writerow([
            self.scan_count,
            msg.header.stamp.sec,
            msg.header.stamp.nanosec
        ])

        self.csv_file.flush()

        if self.scan_count % 10 == 0:
            self.get_logger().info(
                f'Saved scan {self.scan_count}: '
                f'{points.shape[0]} points'
            )

    def destroy_node(self):

        self.csv_file.close()
        super().destroy_node()


def main(args=None):

    rclpy.init(args=args)

    node = PointCloudRecorder()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
