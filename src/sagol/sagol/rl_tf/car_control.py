import rclpy
from rclpy.node import Node
from ackermann_msgs.msg import AckermannDriveStamped
#from rclpy.exceptions import ROSException

from threading import Thread
import time
import argparse

try:
    from geometry_msgs.msg import PoseStamped
    from geometry_msgs.msg import PoseWithCovarianceStamped
except ImportError:
    pass

try:
    from .sensors import Sensors
except ImportError:
    from .sensors import Sensors

PUBLISHER_WAIT = 0.005

MAX_SPEED_REDUCTION = 4.5
STEERING_SPEED_REDUCTION = 4.5
BACKWARD_SPEED_REDUCTION = 4.5
LIGHTLY_STEERING_REDUCTION = 2.4
BACKWARD_SECONDS = 1.5

MAX_SPEED_REDUCTION_SIM = 1
STEERING_SPEED_REDUCTION_SIM = 1.4
BACKWARD_SPEED_REDUCTION_SIM = 3
LIGHTLY_STEERING_REDUCTION_SIM = 2.4
BACKWARD_SECONDS_SIM = 1.5
USE_RESET_INSTEAD_OF_BACKWARDS_SIM = True

MIN_SPEED_REDUCTION = 5

class Drive(Node):
    def __init__(self, sensors, is_simulator=False):
        super().__init__('drive_node')
        self.is_simulator = is_simulator
        if not is_simulator:
            topic = "/vesc/high_level/ackermann_cmd_mux/input/nav_0"
            max_steering = 0.34
            self.max_speed_reduction = MAX_SPEED_REDUCTION
            self.steering_speed_reduction = STEERING_SPEED_REDUCTION
            self.backward_speed_reduction = BACKWARD_SPEED_REDUCTION
            self.lightly_steering_reduction = LIGHTLY_STEERING_REDUCTION
            self.backward_seconds = BACKWARD_SECONDS
        else:
            topic = "/drive"
            max_steering = 0.4189
            self.max_speed_reduction = MAX_SPEED_REDUCTION_SIM
            self.steering_speed_reduction = STEERING_SPEED_REDUCTION_SIM
            self.backward_speed_reduction = BACKWARD_SPEED_REDUCTION_SIM
            self.lightly_steering_reduction = LIGHTLY_STEERING_REDUCTION_SIM
            self.backward_seconds = BACKWARD_SECONDS_SIM
            #self.reset_publisher = self.create_publisher(PoseStamped, "/pose", 10)
            self.reset_publisher = self.create_publisher(PoseWithCovarianceStamped, "/initialpose", 10)
        self.max_speed = self.declare_parameter("max_speed", 5).value
        self.max_steering = self.declare_parameter("max_steering", max_steering).value
        self.drive_publisher = self.create_publisher(AckermannDriveStamped, topic, 10)
        self.sensors = sensors
        self.stop()
        process = Thread(target=self.drive_command_runner)
        process.daemon = True
        process.start()
        self.get_logger().info(f"max_speed: {self.max_speed}, max_steering: {self.max_steering}")

    def forward(self):
        self.send_drive_command(self.max_speed/self.max_speed_reduction, 0)
    
    def backward(self):
        self.send_drive_command(-self.max_speed/self.backward_speed_reduction, 0)
    
    def stop(self):
        self.send_drive_command(0, 0)
    
    def right(self):
        self.send_drive_command(self.max_speed/self.steering_speed_reduction, -self.max_steering)

    def left(self):
        self.send_drive_command(self.max_speed/self.steering_speed_reduction, self.max_steering)

    def lightly_right(self):
        self.send_drive_command(self.max_speed/self.steering_speed_reduction, -self.max_steering/self.lightly_steering_reduction)

    def lightly_left(self):
        self.send_drive_command(self.max_speed/self.steering_speed_reduction, self.max_steering/self.lightly_steering_reduction)

    def slowdown(self):
        speed = self.last_speed/2 if self.last_speed/2 > self.max_speed/MIN_SPEED_REDUCTION else self.max_speed/MIN_SPEED_REDUCTION
        self.send_drive_command(speed, self.last_angle)

    def send_drive_command(self, speed, steering_angle):
        self.last_angle = steering_angle
        self.last_speed = speed
        ack_msg = AckermannDriveStamped()
        ack_msg.drive.speed = float(speed)
        ack_msg.drive.steering_angle = float(steering_angle)
        self.ack_msg = ack_msg

    def drive_command_runner(self):
        while rclpy.ok():
            try:
                self.drive_publisher.publish(self.ack_msg)
            except:
                if str(e) == "publish() to a closed topic":
                    pass
                else:
                    raise e
            time.sleep(PUBLISHER_WAIT)

    def backward_until_obstacle(self):
        if USE_RESET_INSTEAD_OF_BACKWARDS_SIM and self.is_simulator:
            self.reset_simulator()
        else:
            self.backward()
            start = time.time()
            while not self.sensors.back_obstacle() and time.time() - start < self.backward_seconds:
                time.sleep(0.01)
            self.stop()
            time.sleep(0.1)

    def reset_simulator(self):
        if self.is_simulator:
            # position 1
            initpose = PoseWithCovarianceStamped()
            '''
            initpose.pose.pose.position.x = 21.51589584350586
            initpose.pose.pose.position.y = 7.113001823425293
            initpose.pose.pose.position.z = 0.0
            initpose.pose.pose.orientation.x = 0.0
            initpose.pose.pose.orientation.y = 0.0
            initpose.pose.pose.orientation.z = 0.37604736511721754
            initpose.pose.pose.orientation.w = 0.9266004420398245


            # position 2
            initpose.pose.pose.position.x = -24.91632080078125
            initpose.pose.pose.position.y = 23.890209197998047
            initpose.pose.pose.orientation.z = -0.969707340250551
            initpose.pose.pose.orientation.w = 0.2442696752857425


            # position 3
            initpose.pose.pose.position.x = -17.376888275146484
            initpose.pose.pose.position.y = 47.93435287475586
            initpose.pose.pose.orientation.z = -0.011258415959253535
            initpose.pose.pose.orientation.w = 0.9999366220266604
            '''

            # position 1 in shanghai
            initpose.pose.pose.position.x = 36.55048751831055
            initpose.pose.pose.position.y = -12.759111404418945
            initpose.pose.pose.orientation.z = -0.48088700042375426
            initpose.pose.pose.orientation.w = 0.8767825801322949

            # position 2 in shanghai
            initpose.pose.pose.position.x = -44.2680778503418
            initpose.pose.pose.position.y = 8.5191068649292
            initpose.pose.pose.orientation.z = -0.42817192613476024
            initpose.pose.pose.orientation.w = 0.9036972953760841

            # position 3 in shanghai
            initpose.pose.pose.position.x = 43.47073745727539
            initpose.pose.pose.position.y = -14.217448234558105
            initpose.pose.pose.orientation.z = 0.8841909693849728
            initpose.pose.pose.orientation.w = 0.46712560372780054

            self.reset_publisher.publish(initpose)

def main(args=None):
    rclpy.init(args=args)
    parser = argparse.ArgumentParser()
    parser.add_argument("--simulator", action='store_true', help="to set the use of the simulator")
    args = parser.parse_args()
    sensors = Sensors()
    drive = Drive(sensors, args.simulator)
    rclpy.spin(drive)
    drive.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
