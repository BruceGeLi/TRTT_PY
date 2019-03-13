"""
Author: Ge Li, ge.li@tuebingen.mpg.de

This scipt shall use Policy Gradient of Reinforcement Learning to establish a policy which map the incoming ball trajectory parameters to the robot's hitting parameters.

Using:
Tensorflow

"""
import tensorflow as tf
import numpy as np
import time
from datetime import datetime
import pathlib
import matplotlib.pyplot as plt
import math
import json


# set random seed
np.random.seed(int(time.time()))
tf.set_random_seed(int(time.time()))


class PolicyGradient:
    def __init__(self, on_train=True, ball_state_dimension=6, hidden_layer_dimension=20, learning_rate=0.0001, output_graph=False, restore_dir_file=None, queue_length=10):

        # dimension of ball trajectory parameters, 6
        self.input_dimension = ball_state_dimension

        self.hidden_layer_dimension = hidden_layer_dimension

        self.learning_rate = learning_rate

        self.restore_dir_file = restore_dir_file
        self.on_train = on_train

        # Queue length indicates how many number of samples are used to train the policy each time
        # This can help reduce the variance of the reward function
        self.queue_length = queue_length

        # Define queues for learning with multiple samplings to reduce variance
        self.ball_state_queue = list()
        self.T_queue = list()
        self.delta_t0_queue = list()
        self.reward_queue = list()

        self.build_net()
        self.sess = tf.Session()
        self.loss_list = list()

        if output_graph is True:
            tf.summary.FileWriter("/tmp/graph", self.sess.graph)

        self.saver = tf.train.Saver()

        if self.restore_dir_file is None:
            self.sess.run(tf.global_variables_initializer())

        else:
            path = pathlib.Path(self.restore_dir_file)
            path_file_suffix = str(path.resolve(
            ).parent) + str(path.resolve().anchor) + str(path.resolve().stem) + ".ckpt"
            self.saver.restore(self.sess, path_file_suffix)
            print("\n\nTensorflow restored parameters from: ",
                  path_file_suffix)

    def build_net(self):
        with tf.name_scope("Inputs"):
            """
                 Place holders for Neural Network's computation
            """
            # Ball state as NN's input
            self.ball_state = tf.placeholder(
                tf.float32, [None, self.input_dimension], name="ball_state")

            # NN's outputs are mean and variance of action's normal distribution
            # With these distribution in hand, we can generate action
            # These action are used here to compute loss function
            self.T = tf.placeholder(
                tf.float32, [None, 1], name="T")
            self.delta_t0 = tf.placeholder(
                tf.float32, [None, 1], name="delta_t0")

            # Reward to do Policy Gradient
            # Together with loss function, do back propagation of NN
            self.reward = tf.placeholder(
                tf.float32, [None, 1], name="reward")

        with tf.name_scope("Neural_Network"):
            # Build hidden layer
            # Consume ball state as input
            self.hidden_layer1 = tf.layers.dense(
                inputs=self.ball_state,
                units=self.hidden_layer_dimension,
                activation=tf.nn.tanh,
                kernel_initializer=tf.random_normal_initializer(
                    mean=0, stddev=0.3),
                name="Hidden_layer1"
            )

            self.hidden_layer2 = tf.layers.dense(
                inputs=self.hidden_layer1,
                units=self.hidden_layer_dimension,
                activation=tf.nn.tanh,
                kernel_initializer=tf.random_normal_initializer(
                    mean=0, stddev=0.3),
                name="Hidden_layer2"
            )

            # Build output layer for T mean raw, bounded[0, 1]
            self.T_mean_raw = tf.layers.dense(
                inputs=self.hidden_layer2,
                units=1,
                activation=tf.nn.sigmoid,
                kernel_initializer=tf.random_normal_initializer(
                    mean=0, stddev=0.3),
                name="T_mean_raw"
            )

            # Build output layer for T standard deviation raw, bounded[0, 1]
            self.T_dev_raw = tf.layers.dense(
                inputs=self.hidden_layer2,
                units=1,
                activation=tf.nn.sigmoid,
                kernel_initializer=tf.random_normal_initializer(
                    mean=0, stddev=0.3),
                name="T_dev_raw"
            )

            # Build output layer for delta t0 mean raw, bounded[0, 1]
            self.delta_t0_mean_raw = tf.layers.dense(
                inputs=self.hidden_layer2,
                units=1,
                activation=tf.nn.sigmoid,
                kernel_initializer=tf.random_normal_initializer(
                    mean=0, stddev=0.3),
                name="delta_t0_mean_raw"
            )

            # Build output layer for delta t0 standard deviation raw, bounded[0, 1]
            self.delta_t0_dev_raw = tf.layers.dense(
                inputs=self.hidden_layer2,
                units=1,
                activation=tf.nn.sigmoid,
                kernel_initializer=tf.random_normal_initializer(
                    mean=0, stddev=0.3),
                name="delta_t0_dev_raw"
            )

        with tf.name_scope("Normal_distribution"):
            # bound T mean from 0.3 to 0.5
            T_mean_weight = tf.fill([1, 1], 0.2)
            T_mean_bias = tf.fill([1], 0.3)

            # bound T dev from 0.00 to 0.02
            T_dev_weight = tf.fill([1, 1], 0.02)
            T_dev_bias = tf.fill([1], 0.00)

            # bound delta_t0 mean from 0.8 to 0.9
            delta_t0_mean_weight = tf.fill([1, 1], 0.1)
            delta_t0_mean_bias = tf.fill([1], 0.8)

            # bound delta_t0 dev from 0.00 to 0.01
            delta_t0_dev_weight = tf.fill([1, 1], 0.01)
            delta_t0_dev_bias = tf.fill([1], 0.00)

            self.T_mean = tf.nn.xw_plus_b(
                x=self.T_mean_raw, weights=T_mean_weight, biases=T_mean_bias)

            self.T_dev = tf.nn.xw_plus_b(
                x=self.T_dev_raw, weights=T_dev_weight, biases=T_dev_bias)

            self.delta_t0_mean = tf.nn.xw_plus_b(
                x=self.delta_t0_mean_raw, weights=delta_t0_mean_weight, biases=delta_t0_mean_bias)

            self.delta_t0_dev = tf.nn.xw_plus_b(
                x=self.delta_t0_dev_raw, weights=delta_t0_dev_weight, biases=delta_t0_dev_bias)

            # Declare normal distribution of T and t0
            self.T_dist = tf.distributions.Normal(
                loc=tf.reshape(self.T_mean, [-1]), scale=tf.reshape(self.T_dev, [-1]))
            self.delta_t0_dist = tf.distributions.Normal(
                loc=tf.reshape(self.delta_t0_mean, [-1]), scale=tf.reshape(self.delta_t0_dev, [-1]))

            # Sample an action from distribution
            self.T_sample = self.T_dist.sample()
            self.delta_t0_sample = self.delta_t0_dist.sample()
            #print("sample shape: ", tf.shape(self.sample))

        # Define logarithm of of these two probability distributions
        # Consume action which are executed by the robot as input
        with tf.name_scope("Loss"):
            # log_prob is 2 dim vector
            self.T_log_prob = self.T_dist.log_prob(tf.reshape(self.T, [-1]))
            self.delta_t0_log_prob = self.delta_t0_dist.log_prob(
                tf.reshape(self.delta_t0, [-1]))

            # reduce mean value along vector dimension
            self.loss = tf.reduce_mean([self.T_log_prob,
                                         self.delta_t0_log_prob] * -tf.reshape(self.reward, [-1]))

        with tf.name_scope("Train"):
            # optimizer
            self.train_optimizer = tf.train.AdamOptimizer(
                self.learning_rate).minimize(self.loss)

    def generate_action(self, ball_state):
        if self.on_train is True:
            T = self.sess.run(self.T_sample, feed_dict={
                self.ball_state: [ball_state]})
            delta_t0 = self.sess.run(self.delta_t0_sample, feed_dict={
                self.ball_state: [ball_state]})
            action = np.reshape([T, delta_t0], [-1])
            return action
        else:
            T = self.sess.run(self.T_mean, feed_dict={
                self.ball_state: [ball_state]})
            delta_t0 = self.sess.run(self.delta_t0_mean, feed_dict={
                self.ball_state: [ball_state]})
            action = np.reshape([T, delta_t0], [-1])
            return action

    def store_episode(self, state, action, reward):
        # Append new episode to queue
        self.ball_state_queue.append(state)
        self.T_queue.append([action[0]])
        self.delta_t0_queue.append([action[1]])
        self.reward_queue.append([reward])

    def learn(self, save=False, save_dir_file="/tmp/RL_NN_parameters"):
        # todo: normalize reward function
        if self.on_train is True:
            if len(self.ball_state_queue) is self.queue_length:
                [_, T_log_prob, delta_t0_log_prob, loss] = self.sess.run([self.train_optimizer, self.T_log_prob, self.delta_t0_log_prob, self.loss], feed_dict={
                    self.ball_state: self.ball_state_queue,
                    self.T: self.T_queue,
                    self.delta_t0: self.delta_t0_queue,
                    self.reward: self.reward_queue
                })
                print("\nT_log_prob", T_log_prob)
                print()
                print("\nloss:", loss)
                self.loss_list.append(loss.item())
                if save is True:
                    now = datetime.now()

                    path = pathlib.Path(save_dir_file)

                    save_file_suffix = str(path.resolve().parent) + str(path.resolve().anchor) + str(path.stem) + "_" + "{:04d}".format(now.year) + "{:02d}".format(
                        now.month) + "{:02d}".format(now.day) + "_" + "{:02d}".format(now.hour) + "{:02d}".format(now.minute) + "{:02d}".format(now.second) + ".ckpt"

                    save_path = self.saver.save(self.sess, save_file_suffix)
                    print("Tensorflow saved parameters into: ", save_path)            
            else:
                print("Episode queue is not full, do not update policy")

        else:
            print("NN is not learning! Deterministic policy is used. ")

        T_mean = self.sess.run(self.T_mean, feed_dict={
            self.ball_state: [self.ball_state_queue[-1]]})
        T_dev = self.sess.run(self.T_dev, feed_dict={
            self.ball_state: [self.ball_state_queue[-1]]})
        delta_t0_mean = self.sess.run(self.delta_t0_mean, feed_dict={
            self.ball_state: [self.ball_state_queue[-1]]})
        delta_t0_dev = self.sess.run(self.delta_t0_dev, feed_dict={
            self.ball_state: [self.ball_state_queue[-1]]})

        T_mean = np.reshape(T_mean, [-1])
        T_dev = np.reshape(T_dev, [-1])
        delta_t0_mean = np.reshape(delta_t0_mean, [-1])
        delta_t0_dev = np.reshape(delta_t0_dev, [-1])

        print("\n       T_mean: {:.3f}".format(T_mean[0]))
        print("        T_dev: {:.3f}".format(T_dev[0]))
        print("delta_t0_mean: {:.3f}".format(delta_t0_mean[0]))
        print(" delta_t0_dev: {:.3f}\n".format(delta_t0_dev[0]))
        
        if len(self.ball_state_queue) is self.queue_length: 
            self.ball_state_queue.clear()
            self.T_queue.clear()
            self.delta_t0_queue.clear()
            self.reward_queue.clear()

    def print_loss(self, loss_dir_file=None):
        if self.on_train is True:
            compress_list = list()
            episodes_list = list()
            list_length = len(self.loss_list)
            point_to_show = 30
            compress_rate = math.ceil(list_length / point_to_show)
            for counter in range(0, list_length, compress_rate):
                if counter + compress_rate < list_length:
                    compress_list.append(
                        sum(self.loss_list[counter: counter + compress_rate])/compress_rate)
                    episodes_list.append(self.queue_length * counter + math.floor(self.queue_length * compress_rate / 2.0))
                else:
                    compress_list.append(
                        sum(self.loss_list[counter:])/len(self.loss_list[counter:]))
                    episodes_list.append(
                        self.queue_length * counter + math.floor(self.queue_length * len(self.loss_list[counter:])/2.0))

            #print("raw loss", self.loss_list)
            #print("after compress:", compress_list)
            plt.plot(episodes_list, compress_list)
            plt.title("loss function")
            plt.xlabel("episodes")
            plt.ylabel("loss")

            if loss_dir_file is not None:
                path = pathlib.Path(loss_dir_file)
                save_file_suffix = str(
                    path.resolve().parent) + str(path.resolve().anchor) + str(path.stem) + ".json"
                with open(save_file_suffix, 'w') as outfile:
                    json.dump({"loss_list":self.loss_list, "queue_length": self.queue_length}, outfile)

            plt.show()
        else:
            print("NN is not learning! Deterministic policy is used. No loss available")
