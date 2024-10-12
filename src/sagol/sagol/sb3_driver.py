#import gym
import glob
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
import numpy as np
import math
#from gym.envs.registration import register
from gymnasium.envs.registration import register
from collections import OrderedDict
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
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped
import torch
from std_msgs.msg import Float32, Int32
from geometry_msgs.msg import Point
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import PoseWithCovarianceStamped
from tf2_msgs.msg import TFMessage
from stable_baselines3.common.env_checker import check_env

TTC_THRESHOLD_REAL_CAR = 0.35
EUCLIDEAN_THRESHOLD_REAL_CAR = 0.35
ONLY_EXTERNAL_BARRIER = False
EXTERNAL_BARRIER_THRESHOLD = 2.73

class SagolCar(Node):
    def __init__(self):
        super().__init__('sagol_car')
        self.last_angle = 0.0
        self.last_speed = 0.0
        self.max_speed = 1.0
        self.max_steering = 0.5236
        self.backward_speed = -1

        self.ttc_treshold = TTC_THRESHOLD_REAL_CAR
        self.euclidean_treshold = EUCLIDEAN_THRESHOLD_REAL_CAR
        self.emergency_brake = False
        self.collision_count = 0
        self.safety = True
        self.backward_seconds = 1.5


        # Subscriptions
        self.subs_sensors = [
            #('/imu', Imu),
            ('/ego_racecar/odom', Odometry),
            ('/scan', LaserScan),
        ]

        self.cache = {}

        for topic, msg_type in self.subs_sensors:
            self.create_subscription(msg_type, topic, self.generic_callback_factory(topic), 10)

        # Publishers
        self.pubs = {
            'drive': self.create_publisher(AckermannDriveStamped, '/drive', 10),
            'reset': self.create_publisher(PoseWithCovarianceStamped, '/initialpose', 10),
        }
        self.get_logger().info('SagolCar node has been started.')

        self._tid = {}
        self._tid['drive'] = self.create_timer(0.05, self.timer_callback)

    def single_shot(self, timeout, callback, *args):
        tid = int(time.time()*1000)
        def _callback():
            timer = self._tid.pop(tid)
            timer.cancel()
            callback(*args)
        self._tid[tid] = self.create_timer(1.0, _callback)

    def timer_callback(self):
        self.publish_drive(self.last_speed, self.last_angle)

    def get_car_state(self):
        ranges = self.get_lidar_range()
        if not ranges:
            return []
        current_data = list(ranges)
        if self.add_velocity:
            current_data.append(self.check_car_linear_velocity())
        return current_data

    def reset_pose(self):
        initpose = PoseWithCovarianceStamped()
        initpose.pose.pose.position.x = -2.831418514251709
        initpose.pose.pose.position.y = -0.7414035797119141
        initpose.pose.pose.orientation.z = 0.10800853768286314
        initpose.pose.pose.orientation.w = 0.9941499664475222

        initpose.pose.pose.position.x = -8.107040405273438
        initpose.pose.pose.position.y = 25.138076782226562
        initpose.pose.pose.orientation.z = -0.9999683243274612
        initpose.pose.pose.orientation.w = 0.007959292790800734

        self.pubs['reset'].publish(initpose)

    def generic_callback_factory(self, topic):
        def generic_callback(msg):
            self.cache[topic] = msg
            if topic == '/scan':
                if self.safety:
                    self.check_emergency_brake(msg)
        return generic_callback

    def check_car_linear_velocity(self):
        msg = self.cache.get('/ego_racecar/odom', Odometry())
        if msg.twist.twist.linear.x == 0 and msg.twist.twist.linear.x == 0:
            return 0
        return math.sqrt(msg.twist.twist.linear.x ** 2 + msg.twist.twist.linear.y ** 2)

    def get_lidar_range(self):
        return self.cache.get('/scan', LaserScan()).ranges

    def check_emergency_brake(self, msg):
        if not self.safety:
            return False

        acceleration = self.check_car_linear_velocity()
        lidar_data = self.cache.get('/scan', LaserScan())
        if not lidar_data.ranges:
            return False
        if acceleration > 0:
            for i in range(len(lidar_data.ranges)):
                angle = lidar_data.angle_min + i * lidar_data.angle_increment
                proj_velocity = acceleration * math.cos(angle)
                if proj_velocity != 0:
                    ttc = lidar_data.ranges[i] / proj_velocity
                    if ttc < self.ttc_treshold and ttc >= 0:
                        self.collision_count += 1
                        self.emergency_brake = True
                        break

        if min(lidar_data.ranges) < self.euclidean_treshold:
            self.emergency_brake = True

        if ONLY_EXTERNAL_BARRIER and min(lidar_data.ranges) > EXTERNAL_BARRIER_THRESHOLD:
            self.emergency_brake = True

        if self.emergency_brake:
            self.collision_count += 1
            #print('emergency brake', self.collision_count)
            self.stop_drive()
        return self.emergency_brake

    def disable_safety(self):
        self.safety = False

    def unlock_brake(self):
        self.emergency_brake = False

    def check_back_obstacle(self):
        return False

    def backward_until_obstacle(self):
        #print('[backward until obstacle]')
        self.reset_pose()
        return
        self.backward()
        start = time.time()
        while not self.check_back_obstacle() and time.time() - start < self.backward_seconds:
            time.sleep(0.01)
        self.stop_drive()
        time.sleep(0.1)
        #print('[backward until obstacle done]')

    def enable_safety(self):
        self.safety = True

    def get_collision_count(self):
        return self.collision_count

    def get_scan_data(self):
        ranges = self.cache.get('/scan', LaserScan()).ranges
        return np.array(ranges)

    def get_reduced_scan_data(self):
        data = self.get_scan_data()
        max_distance_norm = 20
        cut_by = 10
        reduce_by = 27

        data_avg = []
        for i in range(0, len(data), reduce_by):
            filtered = list(filter(lambda x:  x <= max_distance_norm, data[i:i + reduce_by]))
            if len(filtered) == 0:
                data_avg.append(max_distance_norm)
            else:
                data_avg.append(sum(filtered)/len(filtered))
        data = data_avg

        data = data[cut_by:-cut_by]
        if max_distance_norm > 1:
            data = [x / max_distance_norm for x in data]
        return data

    def publish_drive(self, speed, steering_angle):
        ack_msg = AckermannDriveStamped()
        ack_msg.drive.speed = float(speed)
        ack_msg.drive.steering_angle = float(steering_angle)
        self.pubs['drive'].publish(ack_msg)
        #self.get_logger().info(f'Drive command published: {speed}, {steering_angle}')

    def perform_action(self, action):
        #print(action)
        steering, speed = action
        reward = 0
        if abs(steering) < 0.1:
            reward += 0.08
        elif abs(steering) < 0.2:
            reward += 0.05
        elif abs(steering) < 0.3:
            reward += 0.05
        elif abs(steering) < 0.5:
            reward += 0.01

        speed += 1.0
        speed /= 2
        if speed < 0.1:
            reward -= 0.01
        speed *= 5
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
            self.stop_drive()
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

    def stop_drive(self):
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

    def backward(self):
        self.send_drive_command(self.backward_speed, 0.0)

    def send_drive_command(self, speed, steering_angle):
        self.last_angle = steering_angle
        self.last_speed = speed


def convert_range(value, input_range, output_range):
    # converts value(s) from range to another range
    # ranges ---> [min, max]
    (in_min, in_max), (out_min, out_max) = input_range, output_range
    in_range = in_max - in_min
    out_range = out_max - out_min
    #print('convert_range called', type(value), type(in_min))
    #print(value, in_min)
    return (((value - in_min) * out_range) / in_range) + out_min

class SagolSimEnv(gym.Env):
    def __init__(self, sagol_car_node=None):
        super(SagolSimEnv, self).__init__()
        self.sagol_car_node = sagol_car_node

        # Define observation space
        self.pose = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float64)
        self.acceleration = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float64)
        self.lidar = gym.spaces.Box(low=0, high=np.inf, shape=(1080,), dtype=np.float64)

        self.observation_space = gym.spaces.Dict({
            'acceleration': self.acceleration,
            'lidar': self.lidar,
            'pose': self.pose,
        })
        self.observation_space = self.lidar

        # Define action space
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float64)
        self.state_data = []


        self.reward = 0.0

        self.lidar_min = 0
        self.lidar_max = 10
        self.stop_count = 0

    def process_state(self):
        data = self.sagol_car_node.get_reduced_scan_data()
        newstate = (data, self.sagol_car_node.check_car_linear_velocity())
        self.state_data = [newstate,]+self.state_data[:1]

    def get_state(self):
        state = self.state_data
        if len(state) < 2:
            return {}
        lidar_state = [state[0][0], state[1][0]]
        acc_state = [state[0][1], state[1][1]]
        return {'scan': np.asarray(lidar_state).reshape((len(lidar_state[0]), 2)), 'accel': np.asarray(acc_state)}


    def reset(self, seed=None):
        print('reset called')
        self.process_state()
        self.reward = 0.0
        self.collision_count = self.sagol_car_node.get_collision_count()

        pose = np.zeros(6)
        acceleration = np.zeros(6)
        lidar = np.zeros(1080)

        observations = {
            'acceleration': acceleration,
            'lidar': lidar,
            'pose': pose,
        }
        observations = lidar

        return observations, self.get_state()

    def step(self, action):
        #print(f'[step called with action: {action} - start]', id(self))
        self.reward += self.sagol_car_node.perform_action(action)
        if self.sagol_car_node.emergency_brake:
            done = True
            self.reward -= 1.0
            print('collision detected, reward', self.reward)


            self.collision_count = self.sagol_car_node.get_collision_count()
            self.sagol_car_node.disable_safety()
            time.sleep(0.6)
            self.sagol_car_node.backward_until_obstacle()
            time.sleep(0.4)
            self.sagol_car_node.enable_safety()
            self.sagol_car_node.unlock_brake()
            # if you select right/left from stop state, the real car turn the servo without moving..
            self.sagol_car_node.forward()
            time.sleep(0.4)

        else:
            done = False

            self.reward += self.sagol_car_node.check_car_linear_velocity() * 0.3 * 0.09
            self.reward += min(list(self.sagol_car_node.get_lidar_range())) * 0.01

            if self.sagol_car_node.last_speed == 0:
                self.stop_count += 1
                if self.stop_count > 10:
                    done = True
                    self.reward -= 2.0

        pose = np.zeros(6)
        acceleration = np.zeros(6)
        
        observations = {
            'acceleration': acceleration,
            'lidar': self.sagol_car_node.get_scan_data(),
            'pose': pose,
        }
        observations = self.sagol_car_node.get_scan_data()
        #print(f'State: {observations}, Reward: {self.reward}, Done: {done}, collision: {self.collision_count}')
        truncated = False
        #print(f'[step called with action: {action} - end]', id(self))
        self.process_state()
        return observations, self.reward, done, truncated, self.get_state()

    def render(self, mode='human'):
        print('render called')

    def close(self):
        print('close called')

# Register the custom environment
#register(
#    id='SagolSim-v0',
#    entry_point=SagolSimEnv,
#)

def train_and_evaluate(sagol_car_node, model_file=None):
    # Create the environment
    #env = gym.make('SagolSim-v0', sagol_car_node=sagol_car_node)
    env = SagolSimEnv(sagol_car_node)

    # it will check your custom environment and output additional warnings if needed
    check_env(env)

    # Check if GPU is available and set the device accordingly
    device = 'cuda' if torch.cuda.is_available() else 'cpu'


    # Save a checkpoint every 1000 steps
    checkpoint_callback = CheckpointCallback(
        save_freq=1000,
        save_path="./models/",
        name_prefix="rl_model",
        save_replay_buffer=True,
        save_vecnormalize=True,
        verbose=3,
    )

    eval_callback = EvalCallback(env, best_model_save_path="./bestmodels/",
                                 eval_freq=5000,
                                 deterministic=True, render=True,
                                 verbose=1)

    # Instantiate the agent with the specified device
    model = PPO('MlpPolicy', env, verbose=1, device=device)
    if model_file:
        print('load', model_file)
        model = PPO.load(model_file, env=env, device=device)
    print(dir(model))
    print(help(model.load))
    print(help(PPO))

    # Train the agent
    model.learn(total_timesteps=200000,
                callback=checkpoint_callback,
                progress_bar=False)

    # Evaluate the agent
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
    print(f"Mean reward: {mean_reward} +/- {std_reward}")

    # Test the trained agent
    obs, _states = env.reset()
    for _ in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        print(action, _states)
        obs, rewards, dones, truncated, info = env.step(action)
        print('reward', rewards)
        env.render("human")
        if dones:
            obs, info = env.reset()

    env.close()

def evaluate(sagol_car_node, model):
    # Create the environment
    #env = gym.make('SagolSim-v0', sagol_car_node=sagol_car_node)
    env = SagolSimEnv(sagol_car_node)

    # Instantiate the agent with the specified device
    print(f"loading model: {model}")

    model = PPO.load(model, env=env)
    obs, state = env.reset()

    while True:
        action, _states = model.predict(obs, state, deterministic=True)
        print(action, _states)
        if len(action)==1:
            action = action[0]
        obs, rewards, dones, truncated, _state = env.step(action)
        #print('state', _state, 'reward', rewards)
        if dones:
            obs, info = env.reset()
            print('reward', rewards)


def main():
    parser = argparse.ArgumentParser(description='SagolCar ROS2 Node')
    parser.add_argument('-d', '--debug', action='store_true', help='Enable debug logging')
    parser.add_argument('-l', '--load', help='load model')
    parser.add_argument('mode', default="train", help='mode (train or evaluate)')
    args = parser.parse_args()

    # Initialize ROS 2
    rclpy.init()

    # Set logging level
    if args.debug:
       rclpy.logging.set_logger_level('sagol_car', rclpy.logging.LoggingSeverity.DEBUG)

    # Create the SagolCar node
    sagol_car_node = SagolCar()

    # Start the training and evaluation in a separate thread
    def start_training(model):
        print("Training thread start")
        train_thread = Thread(target=train_and_evaluate, args=(sagol_car_node, model))
        train_thread.start()

    def evaluate_model(model):
        print("Evaluation thread start")
        train_thread = Thread(target=evaluate, args=(sagol_car_node, model))
        train_thread.start()

    if args.mode == "evaluate":
        sagol_car_node.single_shot(1.0, evaluate_model, args.load)
    else:
        sagol_car_node.single_shot(1.0, start_training, args.load)

    rclpy.spin(sagol_car_node)

    sagol_car_node.destroy_node()

    # Shutdown ROS 2
    rclpy.shutdown()

if __name__ == "__main__":
    main()