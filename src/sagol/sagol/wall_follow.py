import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
import math

class WallFollow(Node):

    def __init__(self):
        super().__init__('wall_follow_node')

        # Create ROS subscribers and publishers
        self.publisher_ = self.create_publisher(AckermannDriveStamped, '/drive', 10)
        self.subscription_ = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)

        # PID CONTROL PARAMS
        self.kp = 3
        self.ki = 0
        self.kd = 0.1
        self.servo_offset = 0.0
        self.prev_error = 0.0
        self.error = 0.0
        self.integral = 0.0
        self.start_t = -1
        self.curr_t = 0.0
        self.prev_t = 0.0

    def get_range(self, scan_msg, angle):
        """
        Simple helper to return the corresponding range measurement at a given angle. Make sure you take care of NaNs and infs.

        Args:
            scan_msg: single range array from the LiDAR
            angle: between angle_min and angle_max of the LiDAR in radians

        Returns:
            range: range measurement in meters at the given angle
        """
        assert angle >= scan_msg.angle_min and angle <= scan_msg.angle_max  # Angle must be within range
        i = int((angle - scan_msg.angle_min) / scan_msg.angle_increment)  # index i of closest angle
        if math.isnan(scan_msg.ranges[i]) or scan_msg.ranges[i] > scan_msg.range_max:
            return scan_msg.range_max  # In case of NaNs and infinity, just return the maximum of the scan message
        return scan_msg.ranges[i]

    def to_radians(self, theta):
        return math.pi * theta / 180.0

    def to_degrees(self, theta):
        return theta * 180.0 / math.pi

    def get_error(self, scan_msg, dist):
        """
        Calculates the error to the wall. Follow the wall to the left (going counter clockwise in the Levine loop). You potentially will need to use get_range()

        Args:
            scan_msg: single range array from the LiDAR
            dist: desired distance to the wall

        Returns:
            error: calculated error
        """
        a = self.get_range(scan_msg, self.to_radians(-50.0))
        b = self.get_range(scan_msg, self.to_radians(-90.0))  # 0 degrees is in front of the card.
        theta = self.to_radians(40.0)  # 90.0 - 50.0 = 40.0 degrees
        alpha = math.atan((a * math.cos(theta) - b) / (a * math.sin(theta)))
        D_t = b * math.cos(alpha)

        self.prev_error = self.error
        self.error = dist - D_t
        self.integral += self.error
        self.prev_t = self.curr_t
        self.curr_t = scan_msg.header.stamp.nanosec * 10e-9 + scan_msg.header.stamp.sec
        if self.start_t == 0.0:
            self.start_t = self.curr_t

    def pid_control(self):
        """
        Based on the calculated error, publish vehicle control

        Args:
            error: calculated error
            velocity: desired velocity

        Returns:
            None
        """
        angle = 0.0
        # Use kp, ki & kd to implement a PID controller
        if self.prev_t == 0.0:
            return
        angle = self.kp * self.error + self.ki * self.integral * (self.curr_t - self.start_t) + self.kd * (
                    self.error - self.prev_error) / (self.curr_t - self.prev_t)

        drive_msg = AckermannDriveStamped()
        # Fill in drive message and publish
        drive_msg.drive.steering_angle = angle

        # We go slower if we need to a large steering angle correction
        if abs(drive_msg.drive.steering_angle) >= self.to_radians(0) and abs(
                drive_msg.drive.steering_angle) < self.to_radians(10):
            drive_msg.drive.speed = 1.5
        elif abs(drive_msg.drive.steering_angle) >= self.to_radians(10) and abs(
                drive_msg.drive.steering_angle) < self.to_radians(20):
            drive_msg.drive.speed = 1.0
        else:
            drive_msg.drive.speed = 0.5
        self.publisher_.publish(drive_msg)

    def scan_callback(self, scan_msg):
        """
        Callback function for LaserScan messages. Calculate the error and publish the drive message in this function.

        Args:
            scan_msg: Incoming LaserScan message

        Returns:
            None
        """
        self.get_error(scan_msg, 1)
        self.pid_control()


def main(args=None):
    rclpy.init(args=args)
    wall_follow = WallFollow()
    rclpy.spin(wall_follow)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
