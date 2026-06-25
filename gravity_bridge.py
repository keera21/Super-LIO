import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, Imu
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import numpy as np

from livox_ros_driver2.msg import CustomMsg, CustomPoint

class NativeLivoxBridge(Node):
    def __init__(self):
        super().__init__('native_livox_bridge')
        
        qos_subscribe = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=100)
        qos_publish = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=100)
        
        self.sub_lidar = self.create_subscription(PointCloud2, '/livox/lidar', self.lidar_cb, qos_subscribe)
        self.sub_imu = self.create_subscription(Imu, '/imu', self.imu_cb, qos_subscribe)
        
        self.pub_lidar = self.create_publisher(CustomMsg, '/livox/lidar_final', qos_publish)
        self.pub_imu = self.create_publisher(Imu, '/imu_final', qos_publish)
        
        self.lidar_count = 0
        self.startup_frames = 0
        
        self.get_logger().info("[BRIDGE] Production Mode Active. Warmup Protocol running...")

    def imu_cb(self, msg):
        # Simply pass the IMU data through as fast as possible in complete silence
        self.pub_imu.publish(msg)

    def lidar_cb(self, msg):
        try:
            self.startup_frames += 1
            if self.startup_frames <= 5:
                self.get_logger().info(f"[BRIDGE] Dropping warmup frame {self.startup_frames}/5 to safely fill IMU buffer...")
                return

            raw_bytes = np.frombuffer(msg.data, dtype=np.uint8).reshape(-1, msg.point_step)
            x = raw_bytes[:, 0:4].copy().view(np.float32).flatten()
            y = raw_bytes[:, 4:8].copy().view(np.float32).flatten()
            z = raw_bytes[:, 8:12].copy().view(np.float32).flatten()

            out_msg = CustomMsg()
            out_msg.header = msg.header
            out_msg.header.frame_id = "livox_frame"
            out_msg.timebase = int(msg.header.stamp.sec * 1e9 + msg.header.stamp.nanosec)
            out_msg.point_num = len(x)
            out_msg.lidar_id = 1

            points = []
            for i in range(len(x)):
                pt = CustomPoint()
                pt.offset_time = int(i * 10) 
                pt.x = float(x[i])
                pt.y = float(y[i])
                pt.z = float(z[i])
                pt.reflectivity = 100
                pt.tag = 0
                pt.line = int(i % 4)
                points.append(pt)

            out_msg.points = points
            self.pub_lidar.publish(out_msg)
            
            self.lidar_count += 1
            if self.lidar_count % 10 == 0:
                self.get_logger().info(f"[BRIDGE] Pushed {self.lidar_count} Dense NATIVE Livox Scans.")
                
        except Exception as e:
            self.get_logger().error(f"Format Error: {e}")

def main():
    rclpy.init()
    node = NativeLivoxBridge()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
