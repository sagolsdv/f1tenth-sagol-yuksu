import gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
import numpy as np
from gym.envs.registration import register
import rclpy
import time
from rclpy.node import Node
from std_msgs.msg import Int32
from std_msgs.msg import Float32
from sensor_msgs.msg import LaserScan
from threading import Thread
import argparse
import sys
from sensor_msgs.msg import Image, Imu, JointState, LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
import torch
from std_msgs.msg import Float32, Int32
from geometry_msgs.msg import Point
from tf2_msgs.msg import TFMessage
from stable_baselines3.common.env_checker import check_env


class SagolCar(Node):
    def __init__(self, is_autodrive=False):
        super().__init__('sagol_car')
        self.is_autodrive = is_autodrive
        self.last_angle = 0.0
        self.last_speed = 0.0
        self.max_speed = 1.0
        self.max_steering = 0.5236
        self.backward_speed = -0.1


        self.params = {'mu': 1.0489, 'C_Sf': 4.718, 'C_Sr': 5.4562, 'lf': 0.15875, 'lr': 0.17145, 'h': 0.074, 'm': 3.74, 'I': 0.04712, 's_min': -0.4189, 's_max': 0.4189, 'sv_min': -3.2, 'sv_max': 3.2, 'v_switch': 7.319, 'a_max': 9.51, 'v_min':-5.0, 'v_max': 20.0, 'width': 0.31, 'length': 0.58}


        # Subscriptions
        self.subs_sensors = [
            ('/autodrive/f1tenth_1/front_camera', Image),
            ('/autodrive/f1tenth_1/imu', Imu),
            ('/autodrive/f1tenth_1/left_encoder', JointState),
            ('/autodrive/f1tenth_1/lidar', LaserScan),
            ('/autodrive/f1tenth_1/right_encoder', JointState),
            ('/autodrive/f1tenth_1/steering', Float32),
            ('/autodrive/f1tenth_1/throttle', Float32)
        ]

        self.subs_restricted = [
            ('/autodrive/f1tenth_1/best_lap_time', Float32),
            ('/autodrive/f1tenth_1/collision_count', Int32),
            ('/autodrive/f1tenth_1/ips', Point),
            ('/autodrive/f1tenth_1/lap_count', Int32),
            ('/autodrive/f1tenth_1/lap_time', Float32),
            ('/autodrive/f1tenth_1/last_lap_time', Float32),
            ('/autodrive/f1tenth_1/speed', Float32),
            ('/tf', TFMessage)
        ]

        self.cache = {}

        for topic, msg_type in self.subs_sensors:
            self.create_subscription(msg_type, topic, self.generic_callback_factory(topic), 10)

        for topic, msg_type in self.subs_restricted:
            self.create_subscription(msg_type, topic, self.generic_callback_factory(topic), 10)

        # Publishers
        self.pubs = {
            'steering': self.create_publisher(Float32, '/autodrive/f1tenth_1/steering_command', 10),
            'throttle': self.create_publisher(Float32, '/autodrive/f1tenth_1/throttle_command', 10),
            'drive': self.create_publisher(AckermannDriveStamped, '/drive', 10),
        }
        self.get_logger().info('SagolCar node has been started.')

    def generic_callback_factory(self, topic):
        def generic_callback(msg):
            self.cache[topic] = msg
            #if topic == '/autodrive/f1tenth_1/collision_count':
            #    self.get_logger().info(f'Received message on restricted topic: {topic} with data: {msg}')
        return generic_callback

    def get_collision_count(self):
        return self.cache.get('/autodrive/f1tenth_1/collision_count', Int32()).data

    def get_scan_data(self):
        ranges = self.cache.get('/autodrive/f1tenth_1/lidar', LaserScan()).ranges
        if self.is_autodrive:
            return np.array(ranges[:-1])
        return np.array(ranges)

    def publish_steering(self, steering_angle):
        msg = Float32()
        msg.data = steering_angle
        self.pubs['steering'].publish(msg)
        #self.get_logger().info(f'Steering angle published: {steering_angle}')

    def publish_throttle(self, throttle_value):
        msg = Float32()
        msg.data = throttle_value
        self.pubs['throttle'].publish(msg)
        #self.get_logger().info(f'Throttle command published: {throttle_value}')

    def publish_drive(self, speed, steering_angle):
        ack_msg = AckermannDriveStamped()
        ack_msg.drive.speed = float(speed)
        ack_msg.drive.steering_angle = float(steering_angle)
        self.pubs['drive'].publish(ack_msg)
        self.get_logger().info(f'Drive command published: {speed}, {steering_angle}')

    def perform_action(self, action):
        steering, speed = action
        reward = speed
        if abs(steering) < 0.1:
            reward *= 1.5
        elif abs(steering) < 0.2:
            reward *= 1.3
        elif abs(steering) < 0.3:
            reward *= 1.2
        elif abs(steering) < 0.5:
            reward *= 1.1
        self.send_drive_command(speed, steering)
        return reward
        '''
        reward = 0.0
        if action == 0:
            self.forward()
            reward = 1.0
        elif action == 1:
            self.speedup()
            reward = 0.5
        elif action == 2:
            self.slowdown()
            reward = 0.2
        elif action == 3:
            self.stop()
        elif action == 4:
            self.right()
            reward = 0.3
        elif action == 5:
            self.left()
            reward = 0.3
        elif action == 6:
            self.slightly_right()
            reward = 0.4
        elif action == 7:
            self.slightly_left()
            reward = 0.4
        else:
            raise ValueError(f'Invalid action: {action}')
        #elif action == 8:
        #    self.backward()
        return reward
        '''

    '''
    def forward(self):
        self.send_drive_command(self.max_speed, 0.0)

    def speedup(self):
        new_speed = self.last_speed * 1.1
        if new_speed > self.max_speed:
            new_speed = self.max_speed
        self.send_drive_command(new_speed, 0.0)

    def slowdown(self):
        new_speed = self.last_speed - (self.last_speed * 0.1)
        if new_speed < 0.0:
            new_speed = 0.0
        self.send_drive_command(new_speed, 0.0)

    def stop(self):
        self.send_drive_command(0.0, 0.0)

    def right(self):
        new_speed = self.last_speed - (self.last_speed * 0.3)
        if new_speed < 0.0:
            new_speed = 0.05
        self.send_drive_command(new_speed, -self.max_steering)

    def left(self):
        new_speed = self.last_speed - (self.last_speed * 0.3)
        if new_speed < 0.0:
            new_speed = 0.05
        self.send_drive_command(new_speed, self.max_steering)

    def slightly_right(self):
        new_speed = self.last_speed - (self.last_speed * 0.1)
        if new_speed < 0.0:
            new_speed = 0.05
        new_angle = self.last_angle - (self.last_angle * 0.1)
        if new_angle < -self.max_steering:
            new_angle = -self.max_steering
        self.send_drive_command(new_speed, new_angle)

    def slightly_left(self):
        new_speed = self.last_speed - (self.last_speed * 0.1)
        if new_speed < 0.0:
            new_speed = 0.05
        new_angle = self.last_angle + (self.last_angle * 0.1)
        if new_angle > self.max_steering:
            new_angle = self.max_steering
        self.send_drive_command(new_speed, new_angle)
    '''

    def backward(self):
        self.send_drive_command(self.backward_speed, 0.0)

    def send_drive_command(self, speed, steering_angle):
        self.last_angle = steering_angle
        self.last_speed = speed
        if self.is_autodrive:
            self.publish_steering(steering_angle.item())
            self.publish_throttle(speed.item())
        else:
            self.publish_drive(speed.item(), steering_angle.item())

        time.sleep(0.005) # wait for publisher


def convert_range(value, input_range, output_range):
    # converts value(s) from range to another range
    # ranges ---> [min, max]
    (in_min, in_max), (out_min, out_max) = input_range, output_range
    in_range = in_max - in_min
    out_range = out_max - out_min
    print('convert_range called', type(value), type(in_min))
    print(value, in_min)
    return (((value - in_min) * out_range) / in_range) + out_min

class SagolSimEnv(gym.Env):
    def __init__(self, sagol_car_node=None):
        super(SagolSimEnv, self).__init__()
        self.sagol_car_node = sagol_car_node

        # normalised action space, steer and speed
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0, 0.0]),
            high=np.array([1.0, 1.0]),
            dtype=np.float32)

        # normalised observations, just take the lidar scans
        self.observation_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(1080,),
            dtype=np.float32)

        self.reward = 0.0

        self.lidar_min = 0
        self.lidar_max = 10


    def un_normalise_actions(self, actions):
        # convert actions from range [-1, 1] to normal steering/speed range
        steer = convert_range(actions[0], [-1, 1], [self.s_min, self.s_max])
        speed = convert_range(actions[1], [-1, 1], [self.v_min, self.v_max])
        return np.array([steer, speed], dtype=np.float)

    def normalise_observations(self, observations):
        # convert observations from normal lidar distances range to range [-1, 1]
        a= convert_range(observations, [self.lidar_min, self.lidar_max], [-1, 1])
        print('normalize', type(a), a)
        return a


    def reset(self):
        print('reset called')
        state = self.sagol_car_node.get_scan_data()
        self.reward = 0.0
        self.collision_count = self.sagol_car_node.get_collision_count()
        observations = {
            'scans': self.sagol_car_node.get_scan_data(),
            #'collisions': self.collision_count
        }

        return self.normalise_observations(observations['scans']), {}

    def step(self, action):
        print(f'step called with action: {action}')
        self.reward += self.sagol_car_node.perform_action(action)
        collision = self.sagol_car_node.get_collision_count()
        if self.collision_count < collision:
            self.collision_count = collision
            done = True
        else:
            done = False

        '''
        observations = {'ego_idx': self.ego_idx,
            'scans': [],
            'poses_x': [],
            'poses_y': [],
            'poses_theta': [],
            'linear_vels_x': [],
            'linear_vels_y': [],
            'ang_vels_z': [],
            'collisions': self.collisions}
        '''
        observations = {
            'scans': self.sagol_car_node.get_scan_data(),
            #'collisions': self.collision_count
        }
        print(f'State: {observations}, Reward: {self.reward}, Done: {done}, collision: {collision}')
        print('2')
        print(observations['scans'])
        return self.normalise_observations(observations['scans'][0]), self.reward, done, {}

    def render(self, mode='human'):
        print('render called')
        pass  # Example render method

    def close(self):
        print('close called')
        pass  # Example close method

# Register the custom environment
register(
    id='SagolSim-v0',
    entry_point='sb3_driver:SagolSimEnv',
)

def train_and_evaluate(sagol_car_node):
    # Create the environment
    env = gym.make('SagolSim-v0', sagol_car_node=sagol_car_node)

    # it will check your custom environment and output additional warnings if needed
    #check_env(env)

    # Check if GPU is available and set the device accordingly
    device = 'cuda' if torch.cuda.is_available() else 'cpu'


    # Save a checkpoint every 1000 steps
    checkpoint_callback = CheckpointCallback(
        save_freq=1000,
        save_path="./logs/",
        name_prefix="rl_model",
        save_replay_buffer=True,
        save_vecnormalize=True,
        verbose=2,
    )

    eval_callback = EvalCallback(env, best_model_save_path="./logs/",
                                 log_path="./logs/", eval_freq=500,
                                 deterministic=True, render=False,
                                 verbose=1)

    # Instantiate the agent with the specified device
    model = PPO('MlpPolicy', env, verbose=1, device=device)

    # Train the agent
    model.learn(total_timesteps=10000,
                callback=checkpoint_callback,
                progress_bar=False)

    # Evaluate the agent
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
    print(f"Mean reward: {mean_reward} +/- {std_reward}")

    # Test the trained agent
    vec_env = model.get_env()
    obs = vec_env.reset()
    for _ in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = env.step(action)
        vec_env.render("human")
        if dones:
            obs = env.reset()

    env.close()

def main():
    parser = argparse.ArgumentParser(description='SagolCar ROS2 Node')
    parser.add_argument('-d', '--debug', action='store_true', help='Enable debug logging')
    parser.add_argument('-a', '--autodrive', action='store_true', help='Enable Autodrive simulation mode')
    args = parser.parse_args()

    # Initialize ROS 2
    rclpy.init()

    # Set logging level
    if args.debug:
       rclpy.logging.set_logger_level('sagol_car', rclpy.logging.LoggingSeverity.DEBUG)

    # Create the SagolCar node
    sagol_car_node = SagolCar(is_autodrive=args.autodrive)

    # Start the training and evaluation in a separate thread
    def start_training():
        print("Training thread start")
        train_thread = Thread(target=train_and_evaluate, args=(sagol_car_node,))
        train_thread.start()

    # Create a timer to start training after 10 seconds
    sagol_car_node.create_timer(3.0, start_training)

    rclpy.spin(sagol_car_node)

    sagol_car_node.destroy_node()

    # Shutdown ROS 2
    rclpy.shutdown()

if __name__ == "__main__":
    main()
