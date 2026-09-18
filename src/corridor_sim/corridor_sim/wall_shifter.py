#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64

class WallShifter(Node):
    def __init__(self):
        super().__init__('wall_shifter')
        self.publisher = self.create_publisher(Float64, '/model/shifting_wall/joint_cmd', 10)
        self.shifted = False
        
        # Runs precisely every 15.0 seconds
        self.timer = self.create_timer(15.0, self.shift_callback)
        self.get_logger().info('Wall Shifter active. Trigger interval: 15 seconds.')

    def shift_callback(self):
        self.shifted = not self.shifted
        # Toggles 1.5m inwards and back to 0.0m
        target_position = 1.5 if self.shifted else 0.0
        
        msg = Float64()
        msg.data = target_position
        self.publisher.publish(msg)
        self.get_logger().info(f'[T=15s Event] Shifting wall lateral offset: {target_position} m')

def main(args=None):
    rclpy.init(args=args)
    node = WallShifter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
