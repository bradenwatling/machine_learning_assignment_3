import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math

def visualize(w, rows):
    w = np.reshape(w, [rows, 8, 8])

    plt.figure()

    for i in range(rows):
        plot = plt.subplot(1, rows, i + 1)
        plt.axis('off')
        # From piazza
        plt.imshow(w[i], cmap=plt.cm.gray, vmin=0.5 * w.min(), vmax=0.5 * w.max())
    # Remove spacing between the images
    plt.subplots_adjust(wspace=0, hspace=0)

if __name__ == '__main__':
    # Load the tinymnist dataset
    with np.load("tinymnist.npz") as data:
        trainData, trainTarget = data['x'], data['y']
        testData, testTarget = data['x_test'], data['y_test']
        validData, validTarget = data['x_valid'], data['y_valid']

        tf.set_random_seed(521)

        x = tf.placeholder(tf.float32)
        y = tf.placeholder(tf.float32)

        for K in range(4, 5):
            D = trainData.shape[1]
            learning_rate = 0.05

            # Variables
            average = tf.reduce_mean(x, 0)
            mean = tf.Variable(tf.random_normal([D], dtype=tf.float32))
            # The mean must be between 0 and 1
            mean = tf.sigmoid(mean)
            psi = tf.Variable(tf.random_normal([D], dtype=tf.float32))
            # Psi must be a diagonal matrix with positive eigenvalues
            psi = tf.diag(tf.exp(psi))
            W = tf.Variable(tf.random_normal([D, K], dtype=tf.float32))

            # Calculate psi + WW^T.
            # This is a Tensor of shape [D, D]
            matrix = psi + tf.matmul(W, tf.transpose(W))
            # Calculate the log determinant of this matrix using Cholesky decomposition
            log_det = 2.0 * tf.reduce_sum(tf.log(tf.diag_part(tf.cholesky(matrix))))

            # Calculate the (x - mean)^T * (matrix) * (x - mean) term
            # This is a Tensor of shape [B]
            log_exp = -tf.reduce_sum((x - mean) * tf.reduce_sum(tf.matrix_inverse(matrix) * tf.expand_dims(x - mean, 1), -1), -1) / 2

            # Calculate the log marginal likelihood from each term
            # This is a Tensor of shape [B]
            log_marginal_likelihood = tf.reduce_sum(-D / 2 * tf.log(2 * math.pi) - log_det / 2 + log_exp)

            loss = -log_marginal_likelihood

            optimizer = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.99, epsilon=1e-5).minimize(loss)

            sess = tf.InteractiveSession()
            init = tf.global_variables_initializer()
            sess.run(init)

            losses = []
            for j in range(200):
                print(j)
                _, current_loss = sess.run([optimizer, loss], feed_dict={ x:trainData, y:trainTarget })
                losses.append(current_loss)

            print("Training log marginal likelihood: " + str(sess.run(log_marginal_likelihood, feed_dict={ x:trainData, y:trainTarget })))
            print("Testing log marginal likelihood: " + str(sess.run(log_marginal_likelihood, feed_dict={ x:testData, y:testTarget })))
            print("Validation log marginal likelihood: " + str(sess.run(log_marginal_likelihood, feed_dict={ x:validData, y:validTarget })))

            plt.figure()
            plt.plot(losses)
            plt.title('Loss vs. Number of Updates k=' + str(K))

            visualize(sess.run(tf.transpose(W)), K)
            visualize(sess.run(mean, feed_dict={ x:trainData, y:trainTarget }), 1)
            visualize(sess.run(average, feed_dict={ x:trainData, y:trainTarget }), 1)
            plt.show()
