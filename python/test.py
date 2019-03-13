"""
This is a test script to understand tensorflow
The most annoying thing is the shape!
"""

import tensorflow as tf

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
print(result)
