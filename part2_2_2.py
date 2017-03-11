import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math

if __name__ == '__main__':
    # Load the two-class notMNIST dataset
    data = np.load('data2D.npy')

    tf.set_random_seed(521)

    x_in = tf.placeholder(tf.float32)

    for K in range(3, 6):
        B = data.shape[0]
        D = data.shape[1]
        learning_rate = 0.05

        # Create a [K, D] variable to hold the means initialized with a Gaussian
        means = tf.Variable(tf.random_normal([K, D]), dtype=tf.float32)
        # Create a [K] variable to hold the standard deviations
        variance = tf.Variable(tf.zeros([K], dtype=tf.float32))
        # Constrain the variance to [0, inf)
        variance = tf.exp(variance)
        # Create a [K] variable to hold the mixing coefficients
        mixes = tf.Variable(tf.ones([K], dtype=tf.float32) / K)
        # Constrain the mixing coefficients to sum_k(pi_k) = 1 and take the log
        log_pi = tf.nn.log_softmax(mixes)

        # Make x a [B, 1, D] tensor
        x = tf.expand_dims(x_in, 1)

        # Take square of x - means and sum over outer dimension
        # [B, 1, D] - [K, D] = [B, K, D]
        # Summing over the outer dimension gives [B, K]
        squared_diff = tf.reduce_sum(tf.square(x - means), -1)

        # 1 / (2 * pi * sigma_k ^ 2) ^ (D / 2)
        # Shape [K]
        coefficient = tf.pow(2 * math.pi * variance, -D / 2.)

        # -1 / (2 * sigma_k ^ 2) * (x - mu_k) ^ T * (x - mu_k) 
        # Shape [B, K]
        exponent = -squared_diff / (2 * variance)

        # Calculate the likelihood P(x|z=k)=N(x|mu_k, sigma_k^2)
        # Shape [B, K]
        marginal_log_likelihood = tf.reduce_logsumexp(log_pi + tf.log(coefficient) + exponent, -1)

        loss = -tf.reduce_sum(marginal_log_likelihood)

        optimizer = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.99, epsilon=1e-5).minimize(loss)

        sess = tf.InteractiveSession()
        init = tf.global_variables_initializer()
        sess.run(init)

        losses = []
        for j in range(200):
            _, current_loss = sess.run([optimizer, loss], feed_dict={ x_in:data })
            losses.append(current_loss)

        print("Mixes: " + str(sess.run(log_pi, feed_dict={ x_in:data })))
        print("Means: " + str(means.eval(sess)))
        print("variance: " + str(sess.run(variance, feed_dict={ x_in:data })))

        print('Final loss: ' + str(losses[-1]))

        x = data[:, [0]]
        y = data[:, [1]]
        assignments = sess.run(tf.argmax(marginal_log_likelihood, -1), feed_dict={ x_in:data })

        print(assignments)
        print(assignments.shape)

        estimated_means = means.eval(sess)
        means_x = estimated_means[:, [0]]
        means_y = estimated_means[:, [1]]

        plt.figure()
        plt.scatter(x, y, c=assignments, s=20)
        plt.scatter(means_x, means_y, c='r', s=40)
        plt.title('2D Scatter Plot of Cluster Assignments k=' + str(K))

        plt.figure()
        plt.plot(losses)
        plt.title('Loss vs. Number of Updates k=' + str(K))

    plt.show()