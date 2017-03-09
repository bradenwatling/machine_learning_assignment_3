import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # Load the two-class notMNIST dataset
    data = np.load('data2D.npy')

    x_in = tf.placeholder(tf.float32)

    K = 3
    B = data.shape[0]
    D = data.shape[1]
    learning_rate = 0.01

    # Create a [K, D] variable to hold the means initialized with a Gaussian
    means = tf.Variable(tf.random_normal([K, D]), dtype=tf.float32)

    # Make x a [B, 1, D] tensor
    x = tf.expand_dims(x_in, 1)

    # Take square of x - means and sum over outer dimension
    # [B, 1, D] - [K, D] = [B, K, D] 
    squared_diff = tf.reduce_sum(tf.square(x - means), -1)

    # Choose the minimum over the K dimension
    min_squared_diff = tf.reduce_min(squared_diff, -1)

    # Sum up the min_squared_diff for input points
    loss = tf.reduce_sum(min_squared_diff)

    optimizer = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.99, epsilon=1e-5).minimize(loss)

    sess = tf.InteractiveSession()
    init = tf.global_variables_initializer()
    sess.run(init)

    for j in range(200):
        sess.run([optimizer], feed_dict={ x_in:data })

    x = data[:, [0]]
    y = data[:, [1]]

    estimated_means = means.eval(sess)
    means_x = estimated_means[:, [0]]
    means_y = estimated_means[:, [1]]

    plt.scatter(x, y, c='b')
    plt.scatter(means_x, means_y, c='r')
    plt.show()
