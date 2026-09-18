#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, PointCloud2
from laser_geometry import LaserProjection

class ScanToPointCloud(Node):
    def __init__(self):
        super().__init__('scan_to_cloud')
        self.projector = LaserProjection()
        self.sub = self.create_subscription(
            LaserScan, '/scan/points', self.scan_cb, 10)
        self.pub = self.create_publisher(PointCloud2, '/lidar/pointcloud', 10)

    def scan_cb(self, msg):
        cloud_out = self.projector.projectLaser(msg)
        self.pub.publish(cloud_out)

def main(args=None):
    rclpy.init(args=args)
    node = ScanToPointCloud()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
