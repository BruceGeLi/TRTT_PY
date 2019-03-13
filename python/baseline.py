import numpy as np
import os
import math
import time
from policy_gradient import PolicyGradient
from scipy.spatial import distance
import random

class Baseline:
    def __init__(self):
        self.policy_gradient = PolicyGradient(
            ball_state_dimension=6, hidden_layer_dimension=20, learning_rate=0.001, queue_length=1)
        self.main_loop()

    def get_reward(self, target, current):
        generate_error = random.randint(1, 11)
        if generate_error == 1:
            reward = -0.0074
        else:
            reward = -pow(distance.euclidean(target, current), 2) * 1
        #reward = -pow(distance.euclidean(target, current), 2) * 1
        return reward

    def main_loop(self):
        for episode in range(10000):
            print("\nEpisode:", episode+1)
            ball_obs = [0.294038474559784, -3.09185361862183, -0.347362071275711, -
                        1.4689177274704, 4.52377796173096, 1.27400755882263]
            current_action = self.policy_gradient.generate_action(ball_obs)

            target_action=[0.45,0.83]            
            reward= self.get_reward(target_action, current_action)
            self.policy_gradient.store_episode(ball_obs, current_action, reward)
            self.policy_gradient.learn()
        self.policy_gradient.print_loss()

if __name__ == "__main__":
    bl = Baseline()