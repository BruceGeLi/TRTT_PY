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
    def __init__(self, ball_state_dimension, action_dimension, hidden_layer_dimension=20, learning_rate=0.01, output_graph=False, restore_dir_file=None):

        # dimension of ball trajectory parameters, 6
        self.input_dimension = ball_state_dimension

        # dimension of robot action parameters,2
        self.output_dimension = action_dimension

        self.hidden_layer_dimension = hidden_layer_dimension

        self.learning_rate = learning_rate

        self.restore_dir_file = restore_dir_file

        self.current_state = list()
        self.current_action = list()
        self.current_reward = None

        self.build_net()
        self.sess = tf.Session()

        if output_graph is True:
            # tf.summary.FileWriter("")
            pass

        self.saver = tf.train.Saver()

        if self.restore_dir_file is None:
            self.sess.run(tf.global_variables_initializer())
        else:
            self.saver.restore(self.sess, self.restore_dir_file)
            

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

            # Build output layer for T mean raw, bounded[0, 1]
            self.T_mean_raw = tf.layers.dense(
                inputs=self.hidden_layer,
                units=1,
                activation=tf.nn.sigmoid,
                kernel_initializer=tf.random_normal_initializer(
                    mean=0, stddev=0.3),
                bias_initializer=tf.constant_initializer(0.1),
                name="T_mean"
            )

            # Build output layer for T standard deviation raw, bounded[0, 1]
            self.T_dev_raw = tf.layers.dense(
                inputs=self.hidden_layer,
                units=1,
                activation=tf.nn.sigmoid,
                kernel_initializer=tf.random_normal_initializer(
                    mean=0, stddev=0.3),
                bias_initializer=tf.constant_initializer(0.1),
                name="T_dev"
            )

            # Build output layer for delta t0 mean raw, bounded[0, 1]
            self.delta_t0_mean_raw = tf.layers.dense(
                inputs=self.hidden_layer,
                units=1,
                activation=tf.nn.sigmoid,
                kernel_initializer=tf.random_normal_initializer(
                    mean=0, stddev=0.3),
                bias_initializer=tf.constant_initializer(0.1),
                name="delta_t0_mean"
            )

            # Build output layer for delta t0 standard deviation raw, bounded[0, 1]
            self.delta_t0_dev_raw = tf.layers.dense(
                inputs=self.hidden_layer,
                units=1,
                activation=tf.nn.sigmoid,
                kernel_initializer=tf.random_normal_initializer(
                    mean=0, stddev=0.3),
                bias_initializer=tf.constant_initializer(0.1),
                name="delta_t0_dev"
            )

        # bound T mean from 0.3 to 0.5
        T_mean_weight = tf.fill([1, 1], 0.2)
        T_mean_bias = tf.fill([1], 0.3)

        # bound T dev from 0.0 to 0.1
        T_dev_weight = tf.fill([1, 1], 0.1)
        T_dev_bias = tf.fill([1], 0.0)

        # bound delta_t0 mean from 0.8 to 0.9
        delta_t0_mean_weight = tf.fill([1, 1], 0.1)
        delta_t0_mean_bias = tf.fill([1], 0.8)

        # bound delta_t0 dev from 0.0 to 0.1
        delta_t0_dev_weight = tf.fill([1, 1], 0.1)
        delta_t0_dev_bias = tf.fill([1], 0.0)

        self.T_mean = tf.nn.xw_plus_b(
            x=self.T_mean_raw, weights=T_mean_weight, biases=T_mean_bias)

        self.T_dev = tf.nn.xw_plus_b(
            x=self.T_dev_raw, weights=T_dev_weight, biases=T_dev_bias)

        self.delta_t0_mean = tf.nn.xw_plus_b(
            x=self.delta_t0_mean_raw, weights=delta_t0_mean_weight, biases=delta_t0_mean_bias)

        self.delta_t0_dev = tf.nn.xw_plus_b(
            x=self.delta_t0_dev_raw, weights=delta_t0_dev_weight, biases=delta_t0_dev_bias)

        # Declare normal distrubution of T and t0
        # self.dist = tf.distributions.Normal(
        #    loc=[self.T_mean, self.delta_t0_mean], scale=[self.T_dev, self.delta_t0_dev])
        self.T_dist = tf.distributions.Normal(
            loc=self.T_mean, scale=self.T_dev)

        self.delta_t0_dist = tf.distributions.Normal(
            loc=self.delta_t0_mean, scale=self.delta_t0_dev)

        self.sample = [self.T_dist.sample(), self.delta_t0_dist.sample()]
        print("sample shape: ", tf.shape(self.sample))
        # Define logarithm of of these two probability distributions
        # Consume action which are executed by the robot as input
        with tf.name_scope("Loss"):
            # log_prob is 2 dim vector

            #self.log_prob = self.dist.log_prob(self.action)
            self.log_prob = [self.T_dist.log_prob(
                [self.action[0][0]]), self.delta_t0_dist.log_prob([self.action[0][1]])]

            print("log prob shape:", tf.shape(self.log_prob))

            # reduce mean value along vector dimension
            self.loss = tf.reduce_mean(self.log_prob * -self.reward)

        with tf.name_scope("Train"):
            # optimizor
            # self.train_optimizer = tf.train.GradientDescentOptimizer(
            #    self.learning_rate).minimize(loss)
            self.train_optimizer = tf.train.AdamOptimizer(
                self.learning_rate).minimize(self.loss)

        

    def generate_action(self, ball_state):
        action = self.sess.run(self.sample, feed_dict={
                               self.ball_state: [ball_state]})
        action = np.reshape(action, [-1])
        return action

    def store_transition(self, state, action, reward):
        self.current_ball_state = state
        self.current_action = action
        self.current_reward = reward

    def learn(self, save=False, save_dir_file="/tmp/RL_NN_parameters.ckpt"):
        # todo: normalize reward function

        [_, log_prob, loss] = self.sess.run([self.train_optimizer, self.log_prob, self.loss], feed_dict={
            self.ball_state: [self.current_ball_state],
            self.action: [self.current_action],
            self.reward: [[self.current_reward]]
        })
        print("log prob:", log_prob, "  loss:", loss)
        T_mean = self.sess.run(self.T_mean, feed_dict={
                               self.ball_state: [self.current_ball_state]})
        T_dev = self.sess.run(self.T_dev, feed_dict={
                              self.ball_state: [self.current_ball_state]})
        delta_t0_mean = self.sess.run(self.delta_t0_mean, feed_dict={
                                      self.ball_state: [self.current_ball_state]})
        delta_t0_dev = self.sess.run(self.delta_t0_dev, feed_dict={
                                     self.ball_state: [self.current_ball_state]})

        T_mean = np.reshape(T_mean, [-1])
        T_dev = np.reshape(T_dev, [-1])
        delta_t0_mean = np.reshape(delta_t0_mean, [-1])
        delta_t0_dev = np.reshape(delta_t0_dev, [-1])

        print("\n       T_mean: {:.3f}".format(T_mean[0]))
        print("        T_dev: {:.3f}".format(T_dev[0]))
        print("delta_t0_mean: {:.3f}".format(delta_t0_mean[0]))
        print(" delta_t0_dev: {:.3f}\n".format(delta_t0_dev[0]))

        self.current_state = None
        self.current_action = None
        self.current_reward = None

        if save is True:
            save_path = self.saver.save(self.sess, save_dir_file)
            print("Model saved in path: {}".format(save_path))
