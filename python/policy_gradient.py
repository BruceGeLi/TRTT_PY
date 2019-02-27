"""
Author: Ge Li, ge.li@tuebingen.mpg.de

This scipt shall use Policy Gradient of Reinforcement Learning to establish a policy which map the incoming ball trajectory parameters to the robot's hitting parameters.

Using:
Tensorflow

"""
import tensorflow as tf
import numpy as np
import time

# set random seed
np.random.seed(int(time.time()))
tf.set_random_seed(int(time.time()))


class PolicyGradient:
    def __init__(self, state_dimension, action_dimension, hidden_layer_dimension=20, learning_rate=0.01, output_graph=False):

        # dimension of ball trajectory parameters, 6
        self.input_dimension = state_dimension

        # dimension of robot action parameters,2
        self.output_dimension = action_dimension

        self.hidden_layer_dimension = hidden_layer_dimension

        self.learning_rate = learning_rate

        self.current_state = list()
        self.current_action = list()
        self.current_reward = None

        self.build_net()
        self.sess = tf.Session()

        if output_graph is True:
            # tf.summary.FileWriter("")
            pass

        self.sess.run(tf.global_variables_initializer())

    def build_net(self):
        with tf.name_scope("Inputs"):
            """
                 Place holders for Neural Network's computation
            """
            # Ball states as NN's input
            self.ball_states = tf.placeholder(
                tf.float32, [1, self.input_dimension], name="ball_states")

            # NN's outputs are mean and variance of action's normal distribution
            # With these distribution in hand, we can generate actions
            # These actions are used here to compute loss function
            self.actions = tf.placeholder(
                tf.float32, [1, self.output_dimension], name="actions")

            # Reward to do Policy Gradient
            # Together with loss function, do back propagation of NN
            self.reward = tf.placeholder(
                tf.float32, [1, 1], name="reward")

        with tf.name_scope("Neural Network"):
            # Build hidden layer
            # Consume ball states as input
            self.hidden_layer = tf.layers.dense(
                inputs=self.ball_states,
                units=self.hidden_layer_dimension,
                activation=tf.nn.tanh,
                kernel_initializer=tf.random_normal_initializer(
                    mean=0, stddev=0.3),
                bias_initializer=tf.constant_initializer(0.1),
                name="Hidden_layer"
            )

            # Build output layer for T mean
            self.T_mean = tf.layers.dense(
                inputs=self.hidden_layer,
                units=1,
                activation=tf.nn.sigmoid,
                kernel_initializer=tf.random_normal_initializer(
                    mean=0, stddev=0.3),
                bias_initializer=tf.constant_initializer(0.1),
                name="T_mean"
            )

            # Build output layer for T standard deviation
            self.T_dev = tf.layers.dense(
                inputs=self.hidden_layer,
                units=1,
                activation=tf.nn.sigmoid,
                kernel_initializer=tf.random_normal_initializer(
                    mean=0, stddev=0.3),
                bias_initializer=tf.constant_initializer(0.1),
                name="T_dev"
            )

            # Build output layer for delta t0 mean
            self.delta_t0_mean = tf.layers.dense(
                inputs=self.hidden_layer,
                units=1,
                activation=tf.nn.sigmoid,
                kernel_initializer=tf.random_normal_initializer(
                    mean=0, stddev=0.3),
                bias_initializer=tf.constant_initializer(0.1),
                name="delta_t0_mean"
            )

            # Build output layer for delta t0 standard deviation
            self.delta_t0_dev = tf.layers.dense(
                inputs=self.hidden_layer,
                units=1,
                activation=tf.nn.sigmoid,
                kernel_initializer=tf.random_normal_initializer(
                    mean=0, stddev=0.3),
                bias_initializer=tf.constant_initializer(0.1),
                name="delta_t0_dev"
            )

        # Declare normal distrubution of T and t0 independently
        self.T_norm_dist = tf.distributions.Normal(self.T_mean, self.T_dev)
        self.delta_t0_norm_dist = tf.distributions.Normal(
            self.delta_t0_mean, self.delta_t0_dev)

        # Define logarithm of of these two probability distributions
        # With log(M*N) = log(M) + log(N)
        # Consume action which are executed by the robot as input
        with tf.name_scope("Loss"):
            log_prob = self.T_norm_dist.log_prob(tf.slice(self.actions, [0, 0], [1, 1])) + self.delta_t0_norm_dist.log_prob(tf.slice(self.actions, [0, 1], [1, 1]))
            loss =  tf.reduce_mean(-log_prob * self.reward)

        with tf.name_scope("Train"):
            # optimizor
            self.train_optimizer = tf.train.GradientDescentOptimizer(
                self.learning_rate).minimize(loss)

    def choose_action(self, input_states):
        parameters = self.sess.run(self.output_layer, feed_dict={
                                   self.input_states: input_states[np.newaxis, :]})
        current_action = list()
        for mean, sigma in zip(parameters[0::2], parameters[1::2]):
            sub_action = np.random.normal(mean, sigma)
            current_action.append(sub_action)
        return current_action

    def store_transition(self, state, action, reward):
        self.current_state = state
        self.current_action = action
        self.current_reward = reward

    def learn(self):
        # todo: normalize reward function

        self.sess.run(self.train_optimizer, feed_dict={
            self.input_states: np.array(self.current_state),
            self.output_parameters: np.array(self.current_action),
            self.action_value: self.current_reward
        })

        self.current_state = list()
        self.current_action = list()
        self.current_reward = None
