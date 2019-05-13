"""
This is a test script to understand tensorflow
The most annoying thing is the shape!
"""

import tensorflow as tf
import numpy as np
import pathlib
from sklearn.preprocessing import PolynomialFeatures
tfd = tf.distributions

# Define a batch of two scalar valued Normals.
# The first has mean 1 and standard deviation 11, the second 2 and 22.
dist = tfd.Normal(loc=[1, 2.], scale=[11, 22.])

# Evaluate the pdf of the first distribution on 0, and the second on 1.5,
# returning a length two tensor.
prob = dist.prob([0, 1.5])

# Get 3 samples, returning a 3 x 2 tensor.
example = dist.sample([3])


T_mean_raw = [[1.] ,[0.9]]


T_mean_weight = tf.fill([1, 1], 0.2)
T_mean_bias = tf.fill([1], 0.3)

T_mean = tf.nn.xw_plus_b(
            x=T_mean_raw, weights=T_mean_weight, biases=T_mean_bias)

vector = tf.fill([2,5], 0.2)
reward = tf.constant([0.1,0.2,0.3,0.4,0.5])
loss = vector * -1

sess = tf.Session()

result = sess.run(loss)
#print(result)

n1 = np.array([[1,2,3],[4,5,6]])
n2 = np.array(n1)
n3 = n1
a1 = np.array([1,2,3])
n4 = np.array([1,2,3,4,5,6])
#print(np.matmul(np.expand_dims(a1, axis=1), np.expand_dims(a1, axis=0)))

action_sample = np.random.multivariate_normal(
                    [1, 2], [[1,0],[0,1]])

#print(action_sample)

x1 = 3* np.identity(3)
x2 = 2 * np.identity(3)
x3 = np.ones(2)+1
x4 = np.ones(2)
x5 = np.array([[1,2],[3,4]])

X = np.arange(6).reshape(3, 2)
print(np.reshape(X, -1))
Y = np.expand_dims(np.array([2,3]), axis=0)

s1 = str("abc")
s1 += "bhd" + "hb"

print(s1) 