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
    def __init__(self, ball_state_dimension, action_dimension, hidden_layer_dimension=20, learning_rate=0.01, output_graph=False):

        # dimension of ball trajectory parameters, 6
        self.input_dimension = ball_state_dimension

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
            # Ball state as NN's input
            self.ball_state = tf.placeholder(
                tf.float32, [None, self.input_dimension], name="ball_state")

            # NN's outputs are mean and variance of action's normal distribution
            # With these distribution in hand, we can generate action
            # These action are used here to compute loss function
            self.action = tf.placeholder(
                tf.float32, [None, self.output_dimension], name="action")

            # Reward to do Policy Gradient
            # Together with loss function, do back propagation of NN
            self.reward = tf.placeholder(
                tf.float32, [None, 1], name="reward")

        with tf.name_scope("Neural_Network"):
            # Build hidden layer
            # Consume ball state as input
            self.hidden_layer = tf.layers.dense(
                inputs=self.ball_state,
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

        # Declare normal distrubution of T and t0
        self.dist = tf.distributions.Normal(
            loc=[self.T_mean, self.delta_t0_mean], scale=[self.T_dev, self.delta_t0_dev])

        # Define logarithm of of these two probability distributions
        # Consume action which are executed by the robot as input
        with tf.name_scope("Loss"):
            # log_prob is 2 dim vector
            log_prob = self.dist.log_prob(self.action)
            # reduce mean value along vector dimension
            loss = tf.reduce_mean(-log_prob * self.reward)

        with tf.name_scope("Train"):
            # optimizor
            self.train_optimizer = tf.train.GradientDescentOptimizer(
                self.learning_rate).minimize(loss)

    def generate_action(self, ball_state):
        action = self.sess.run(self.dist.sample(), feed_dict={
                               self.ball_state: [ball_state]})
        action = np.reshape(action, [-1])
        return action

    def store_transition(self, state, action, reward):
        self.current_ball_state = state
        self.current_action = action
        self.current_reward = reward

    def learn(self):
        # todo: normalize reward function

        self.sess.run(self.train_optimizer, feed_dict={
            self.ball_state: [self.current_ball_state],
            self.action: [self.current_action],
            self.reward: [[self.current_reward]]
        })

        self.current_state = None
        self.current_action = None
        self.current_reward = None
