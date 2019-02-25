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
    def __init__(self, input_layer_size, output_layer_size, hidden_layer_size=20, learning_rate=0.01, output_graph=False):
        self.input_layer_size = input_layer_size
        self.output_layer_size = output_layer_size
        self.hidden_layer_size = hidden_layer_size
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
            self.input_states = tf.placeholder(
                tf.float32, [1, self.input_layer_size], name="input_states")
            self.output_parameters = tf.placeholder(
                tf.float32, [1, self.output_layer_size], name="output_parameters")
            self.action_value = tf.placeholder(
                tf.float32, [1, 1], name="action_value")

        self.hidden_layer = tf.layers.dense(
            inputs=self.input_states,
            units=self.hidden_layer_size,
            activation=tf.nn.tanh,
            kernel_initializer=tf.random_normal_initializer(
                mean=0, stddev=0.3),
            bias_initializer=tf.constant_initializer(0.1),
            name="Hidden_layer"
        )

        self.output_layer = tf.layers.dense(
            inputs=self.hidden_layer,
            units=self.output_layer_size,
            activation=tf.nn.sigmoid,
            kernel_initializer=tf.random_normal_initializer(
                mean=0, stddev=0.3),
            bias_initializer=tf.constant_initializer(0.1),
            name="Output_layer"
        )

        with tf.name_scope("Loss"):
            # to maximize total reward (log_p * R) is to minimize -(log_p * R), and the tf only have minimize(loss)
            neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=self.output_layer, labels=self.output_parameters)   # this is negative log of chosen action
            # or in this way:
            # neg_log_prob = tf.reduce_sum(-tf.log(self.all_act_prob)*tf.one_hot(self.tf_acts, self.n_actions), axis=1)
            # reward guided loss
            loss = tf.reduce_mean(neg_log_prob * self.output_parameters)

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
