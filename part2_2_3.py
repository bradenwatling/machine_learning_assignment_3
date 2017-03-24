import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math

if __name__ == '__main__':
    # Load the two-class notMNIST dataset
    data = np.load('data2D.npy')

    tf.set_random_seed(521)
    np.random.seed(521)

    x_in = tf.placeholder(tf.float32)

    B = data.shape[0]
    D = data.shape[1]
    learning_rate = 0.05

    randIndx = np.arange(len(data))
    np.random.shuffle(randIndx)
    data = data[randIndx]
    validIndex = int(math.ceil(B * 2 / 3))
    trainData = data[:validIndex]
    validData = data[validIndex:]

    for K in range(1, 6):
        # Create a [K, D] variable to hold the means initialized with a Gaussian
        means = tf.Variable(tf.random_normal([K, D]), dtype=tf.float32)

        # Create a [K] variable to hold the variances
        log_variance = tf.Variable(tf.zeros([K], dtype=tf.float32))
        # Constrain the variance to [0, inf)
        variance = tf.exp(log_variance)

        # Create a [K] variable to hold the mixing coefficients
        mixes = tf.Variable(tf.ones([K], dtype=tf.float32) / K)
        # Constrain the mixing coefficients to sum_k(pi_k) = 1 and take the log
        log_pi = tf.nn.log_softmax(mixes)

        # Make x a [B, 1, D] tensor
        x = tf.expand_dims(x_in, 1)

        # Compute (x - u_k)^T * (x - u_k)
        # [B, 1, D] - [K, D] = [B, K, D]
        # Summing over the outer dimension gives [B, K]
        squared_diff = tf.reduce_sum(tf.square(x - means), -1)

        # The log of the coefficient of the Gaussian
        # 1 / (2 * pi * sigma_k ^ 2) ^ (D / 2)
        # Use the following form to prevent overflow
        # Shape [K]
        log_coefficient = -D / 2. * (tf.log(2 * math.pi) + log_variance)

        # The exponent of the Gaussian
        # -1 / (2 * sigma_k ^ 2) * (x - mu_k) ^ T * (x - mu_k) 
        # Shape [B, K]
        exponent = -squared_diff / (2 * variance)

        # Calculate the log likelihood log(P(x|z=k))
        # Shape [B, K]
        log_likelihood = log_coefficient + exponent

        # Calculate the mixed log likelihood log(P(x|z=k) * P(z=k)) = log(pi_k) + log_likelihood 
        # Shape [B, K]
        mixed_log_likelihood = log_pi + log_likelihood

        # Calculate the log posterior log(P(z=k|x)) using Baye's rule
        # Keep the outer dimension so we can broadcast across the outer dimension of mixed_log_likelihood
        # The posterior distribution will be used to determine which cluster a given point belongs to
        # Shape [B, K]
        log_posterior = mixed_log_likelihood - tf.reduce_logsumexp(mixed_log_likelihood, -1, True)

        # Calculate the marginal log likelihood log(P(X))
        # Shape [B]
        marginal_log_likelihood = tf.reduce_logsumexp(mixed_log_likelihood, -1)

        # Sum over all data points to determine the loss
        # Shape [1]
        loss = -tf.reduce_sum(marginal_log_likelihood)

        # Determine the assignments to each cluster
        cluster_assignments = tf.argmax(log_posterior, -1)
        # Determine the count for each cluster
        clusters, _, cluster_counts = tf.unique_with_counts(cluster_assignments)

        optimizer = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.99, epsilon=1e-5).minimize(loss)

        sess = tf.InteractiveSession()
        init = tf.global_variables_initializer()
        sess.run(init)

        losses = []
        for j in range(200):
            _, current_loss = sess.run([optimizer, loss], feed_dict={ x_in:trainData })
            losses.append(current_loss)

        print
        print('K=' + str(K))

        validLoss = sess.run([loss], feed_dict={ x_in:validData })[0]
        print('Validation loss for k=' + str(K) + ': ' + str(validLoss))

        cluster_ids, counts = sess.run([clusters, cluster_counts], feed_dict={ x_in:data })
        for i in range(cluster_ids.size):
            print('Cluster ' + str(cluster_ids[i]) + ': ' + str(100. * counts[i] / B) + '%')

        x = trainData[:, [0]]
        y = trainData[:, [1]]
        assignments = sess.run(cluster_assignments, feed_dict={ x_in:trainData })

        estimated_means = means.eval(sess)
        means_x = estimated_means[:, [0]]
        means_y = estimated_means[:, [1]]

        plt.figure()
        plt.scatter(x, y, c=assignments, s=20)
        plt.scatter(means_x, means_y, c='r', s=40)
        plt.title('2D Scatter Plot of Training Cluster Assignments k=' + str(K))

        x = validData[:, [0]]
        y = validData[:, [1]]
        assignments = sess.run(cluster_assignments, feed_dict={ x_in:validData })

        plt.figure()
        plt.scatter(x, y, c=assignments, s=20)
        plt.scatter(means_x, means_y, c='r', s=40)
        plt.title('2D Scatter Plot of Validation Cluster Assignments k=' + str(K))

        plt.figure()
        plt.plot(losses)
        plt.title('Loss vs. Number of Updates k=' + str(K))

    plt.show()
