"""
Author: Ge Li, ge.li@tuebingen.mpg.de

This script shall work with the policy gradient script, offering input states and rewards to policy gradient and receive output actions.

This script will link to the vrep simulation, where the ball observation and landing info are generated and the actions should be executed by the robot.

Using:
zmq to communicate with C++ code.

"""

import argparse
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import distance
import os
import math
import zmq
import time
import json
from policy_gradient import PolicyGradient
import datetime


class TrttPort:
    def __init__(self, args):
        self.host = args.host
        self.port = args.port

        if args.train == 'true':
            self.on_train = True
        elif args.train == 'false':
            self.on_train = False
        else:
            raise argparse.ArgumentTypeError(
                'arg "train": boolean value expected.')

        self.batch_num = args.batch_num
        self.reuse_num = args.reuse_num
        self.ep_num = args.ep_num
        self.learning_rate = args.lr
        self.hidden_layer_number = args.hl
        self.hidden_neural_number = args.hn
        self.save_num = args.save_num
        self.save_dir_file = args.save_dir_file
        self.restore_dir_file = args.restore_dir_file
        self.loss_dir_file = args.loss_dir_file
        self.policyGradient = None
        self.save_iterator = 0
        self.current_ball_state = None
        self.current_action = None
        self.current_reward = None

        self.current_landing_info = None
        self.current_ball_racket_dist_info = None

        self.context = None
        self.socket = None

        self.openSocket()
        self.sampling_from_json()
        # self.mainLoop()
        # self.exhaust_sampling()
        self.closeSocket()

    def openSocket(self):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind("tcp://*:8181")

    def closeSocket(self):
        pass

    def generateReward(self, landing_info, distance_info):
        reward = 0

        """
            Second level of reward represents landing position
        """
        target_coordinate = [0.35, -2.93, -0.99]
        # compute Euclidean distance in x and y coordinate space
        distance_to_target = distance.euclidean(
            landing_info[0:2], target_coordinate[0:2])
        #print("\ndistance to target: ", distance_to_target)
        # print("\n")
        if distance_to_target <= 3.0:
            reward += -1 * distance_to_target + 3
        else:
            reward += 0

        return reward

    def recordTrainingData(self):
        pass

    def mainLoop(self):
        self.policyGradient = PolicyGradient(on_train=self.on_train,
                                             ball_state_dimension=6, hidden_layer_dimension=self.hidden_neural_number, learning_rate=self.learning_rate, output_graph=False, restore_dir_file=self.restore_dir_file,
                                             batch_num=self.batch_num, reuse_num=self.reuse_num)

        """
        If Tensorflow-gpu is used, the first action will take quite long time to compute
        In case of Tensorflow-gpu, do one dummy action generation to activate the NN
        """
        self.dummy_ball_state = np.zeros((6)).tolist()
        dummy_action = self.policyGradient.generate_action(
            self.dummy_ball_state)

        for episode_counter in range(self.ep_num):
            """
                Start of new episode
            """
            print("\n\n\n============================================")
            print("\n------ Episode: {}\n".format(episode_counter+1))

            print("------------------------------------")

            """
                Get ball observation from Vrep sim
            """
            print("\n--- Waiting for ball observation...")
            ball_obs_json = self.socket.recv_json()

            print("--- Ball observation received!\n")
            self.current_ball_state = ball_obs_json["ball_obs"]

            ball_velocity_magnitude = np.linalg.norm(
                self.current_ball_state[3:6])
            #print("mag: ", ball_velocity_magnitude)

            energy = self.current_ball_state[2] * 9.81 * \
                0.0027 + 0.5*0.0027 * pow(ball_velocity_magnitude, 2)

            #print("energy: ", energy)

            print("------------------------------------")
            """
                Get action parameters from Policy Gradient
            """
            print("\n--- Computing hitting parameters...")
            #t2 = datetime.datetime.now()
            self.current_action = self.policyGradient.generate_action(
                self.current_ball_state).tolist()
            print("--- Hitting parameters computed!\n\n")

            print("====>           T: {:.6f}\n====>    delta_t0: {:.6f}".format(
                self.current_action[0], self.current_action[1]))

            #t3 = datetime.datetime.now()
            # print(t3-t2)
            # print("\n\n")
            action_json = {
                "T": self.current_action[0], "delta_t0": self.current_action[1]}

            # Try a fixed action
            action_json = {"T": 0.3631, "delta_t0": 0.88}
            self.socket.send_json(action_json)
            print("\n")
            print("--- Action exported!\n")
            print("------------------------------------")

            """
                Get ball reward info from Vrep sim
            """
            print("\n--- Waiting for ball reward info...")
            reward_info_json = self.socket.recv_json()
            print("--- Ball reward info received!\n")

            self.current_landing_info = reward_info_json["landing_point"]
            self.current_ball_racket_dist_info = reward_info_json["min_distance"]

            """
                Generate reward
            """
            # print("Transfer landing info into reward.")
            self.current_reward = self.generateReward(
                self.current_landing_info, self.current_ball_racket_dist_info)
            print("====>      Reward: {:.3f}\n".format(self.current_reward))
            self.policyGradient.store_episode(
                self.current_ball_state, self.current_action, self.current_reward)
            print("------------------------------------")
            """
                Update Policy
            """
            print("\n--- Updating hitting policy...")
            #t5 = datetime.datetime.now()

            if self.save_num is not 0:
                self.save_iterator += 1
                self.save_iterator %= self.save_num
                if self.save_iterator is 0:
                    self.policyGradient.learn(
                        save=True, save_dir_file=self.save_dir_file)
                else:
                    self.policyGradient.learn()
            else:
                self.policyGradient.learn()
            #t6 = datetime.datetime.now()
            # print(t6-t5)
            policy_updated_json = {"policy_ready": True}
            self.socket.send_json(policy_updated_json)
            print("--- Policy updated!")
        self.policyGradient.print_loss(self.loss_dir_file)

    def exhaust_sampling(self):
        data_list = list()
        print("start of exhaust sampling")
        for T in np.arange(0.45, 0.450001, 0.005):
            for delta_t0 in np.arange(0.86, 0.8600001, 0.01):
                for repeat in range(100):
                    data = dict()
                    print("\n====>    delta_t0: ", delta_t0,
                          "\n====>           T: ", T)
                    ball_obs_json = self.socket.recv_json()

                    action_json = {"T": T, "delta_t0": delta_t0}
                    self.socket.send_json(action_json)

                    reward_info_json = self.socket.recv_json()
                    current_landing_info = reward_info_json["landing_point"]
                    current_ball_racket_dist_info = reward_info_json["min_distance"]

                    policy_updated_json = {"policy_ready": True}
                    self.socket.send_json(policy_updated_json)
                    data["T"] = T
                    data["delta_t0"] = delta_t0
                    data["landing_info"] = current_landing_info
                    data["current_ball_racket_dist_info"] = current_ball_racket_dist_info
                    data_list.append(data)
        with open("/tmp/data_list.json", 'w') as outfile:
            json.dump(data_list, outfile)

    def sampling_from_json(self):
        counter = 0
        while True:
            sampling_file = rel_path('../config/sampling.json')
            with open(sampling_file) as j_file:
                j_obj = json.load(j_file)
                T = j_obj["T"]
                delta_t0 = j_obj["delta_t0"]
                #w = j_obj["w"]
                if counter % 2 == 0:
                    w = -1.5
                else:
                    w = -2.0
                #w  = -1.8
                counter += 1
            print("\n====>    delta_t0: ", delta_t0,
                  "\n====>           T: ", T, "\n====>           w: ", w,)
            ball_obs_json = self.socket.recv_json()

            action_json = {"T": T, "delta_t0": delta_t0, "w": w}
            self.socket.send_json(action_json)

            reward_info_json = self.socket.recv_json()
            current_landing_info = reward_info_json["landing_point"]
            current_ball_racket_dist_info = reward_info_json["min_distance"]

            policy_updated_json = {"policy_ready": True}
            self.socket.send_json(policy_updated_json)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    def rel_path(fname):
        return os.path.join(os.path.dirname(__file__), fname)

    parser.add_argument('--host', default='localhost',
                        help="Name of the host to connect to")

    parser.add_argument('--port', default=8181,
                        help="Port of the host to connect to")

    parser.add_argument('--train', default='true',
                        help="Indicate whether train the policy or not ('true' or 'false')")

    parser.add_argument('--lr', default=0.0001, type=float,
                        help="Learning rate.")

    parser.add_argument('--hl', default=2, type=int,
                        help="Number of hidden layers in Neural Network.")

    parser.add_argument('--hn', default=20, type=int,
                        help="Number of neueal in each hidden layer")

    parser.add_argument('--batch_num', type=int, default=10,
                        help="Number of episodes to train the policy once")

    parser.add_argument('--reuse_num', type=int, default=5,
                        help="Number of old batches to be reused in learning")

    parser.add_argument('--ep_num', type=int, default=10000,
                        help="Number of total episodes to sample, e.g. 10000.")
    parser.add_argument('--save_num', type=int, default=0,
                        help="Save the Neural network parameters for every N episode, e.g. 200")
    parser.add_argument('--save_dir_file', default='/tmp/RL_NN_parameters',
                        help="Dir and file name (without .ckpt suffix) where the parameters shall be stored, e.g. /tmp/parameters. Time stamp will be added automatically.")

    parser.add_argument('--restore_dir_file', default=None,
                        help="Dir and file name where the parameters shall be load, e.g. /tmp/parameters")

    parser.add_argument('--loss_dir_file', default=None,
                        help="Dir and file name where the loss data shall be stored, e.g. /tmp/loss")

    args = parser.parse_args()

    pg = TrttPort(args)
