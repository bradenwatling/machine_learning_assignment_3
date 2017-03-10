import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # Load the two-class notMNIST dataset
    data = np.load('data2D.npy')

    tf.set_random_seed(521)

    x_in = tf.placeholder(tf.float32)

    for K in range(1, 6):
        B = data.shape[0]
        D = data.shape[1]
        learning_rate = 0.05

        # Create a [K, D] variable to hold the means initialized with a Gaussian
        means = tf.Variable(tf.random_normal([K, D]), dtype=tf.float32)
        # Create a [K] variable to hold the standard deviations
        stddevs = tf.Variable(tf.random_normal([K], dtype=tf.float32))
        # Create a [K] variable to hold the mixing coefficients
        mixes = tf.Variable(tf.random_normal([K], dtype=tf.float32))

        # Make x a [B, 1, D] tensor
        x = tf.expand_dims(x_in, 1)



        optimizer = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.99, epsilon=1e-5).minimize(loss)

        sess = tf.InteractiveSession()
        init = tf.global_variables_initializer()
        sess.run(init)

        losses = []
        for j in range(200):
            _, current_loss = sess.run([optimizer, loss], feed_dict={ x_in:data })
            losses.append(current_loss)

        print('Final loss: ' + str(losses[-1]))

        cluster_ids, counts = sess.run([clusters, cluster_counts], feed_dict={ x_in:data })
        for i in range(cluster_ids.size):
            print('Cluster ' + str(cluster_ids[i]) + ': ' + str(100. * counts[i] / B) + '%')

        x = data[:, [0]]
        y = data[:, [1]]
        assignments = sess.run([cluster_assignments], feed_dict={ x_in:data })[0]

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
