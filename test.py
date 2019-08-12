import tensorflow as tf
import numpy as np
import random
from tensorflow.examples.tutorials.mnist import input_data

# initialize free parameters
n_input = 28
n_hidden_0 = 20
n_hidden_1 = 40
n_hidden_2 = 200
n_output = 10
lambda_term = 0.1
learning_rate = 0.03
dropout_0 = 0.5
conv_ker_0 = (5,5,1)
conv_ker_1 = (5,5,20)

# Download, extract and load MNIST dataset
data = input_data.read_data_sets('MNIST_data', one_hot=True)
train_reshaped = data.train.images.reshape(-1, 28, 28, 1)
x_test = data.test.images.reshape(-1, 28, 28, 1)
y_test = data.test.labels

train_data = [(x, y) for x, y in zip(train_reshaped, data.train.labels)]

#initialize input and output placeholders
x = tf.placeholder(dtype = tf.float32, shape = [None, n_input, n_input, 1], name = "x")
y_true = tf.placeholder(dtype = tf.float32, shape = [None, n_output], name = "labels")

#initialize weights and biases
weights = [0] * 5
bias = [0] * 5
z = [0] * 5
act = [0] * 5

weights[0] = tf.Variable(tf.random_normal([conv_ker_0[0], conv_ker_0[1], conv_ker_0[2], n_hidden_0], stddev = np.sqrt(1.0 / n_hidden_0) ), name = "weights_0" )
bias[0] = tf.Variable(tf.random_normal([n_hidden_0], stddev = np.sqrt(1.0 / n_hidden_0)))

weights[1] = tf.Variable(tf.random_normal([conv_ker_1[0], conv_ker_1[1], conv_ker_1[2], n_hidden_1], stddev = np.sqrt(1.0 / n_hidden_1)), name = "weights_1")
bias[1] = tf.Variable(tf.random_normal([n_hidden_1], stddev = np.sqrt(1.0 / n_hidden_1)))

weights[2] = tf.Variable(tf.random_normal([n_hidden_1 * 4 * 4, n_hidden_2], stddev = np.sqrt(1.0 / n_hidden_2)), name = "weights_2" )
bias[2] = tf.Variable(tf.random_normal([n_hidden_2], stddev = np.sqrt(1.0 / n_hidden_2)))

weights[3] = tf.Variable(tf.random_normal([n_hidden_2, n_output], stddev = np.sqrt(1.0 / n_output)), name = "weights_3")
bias[3] = tf.Variable(tf.random_normal([n_output], stddev = np.sqrt(1.0 / n_output)))

#conv2d layer 1
z[1] = tf.nn.bias_add( tf.nn.conv2d(input = x, filter = weights[0], strides = [1, 1, 1, 1], padding = 'VALID'), bias[0])
act[1] = tf.nn.relu(z[1])
act_1_pooled = tf.nn.max_pool(act[1], ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'VALID')

#conv2d layer 1
z[2] = tf.nn.bias_add( tf.nn.conv2d(input = act_1_pooled, filter = weights[1], strides = [1, 1, 1, 1], padding = 'VALID'), bias[1])
act[2] = tf.nn.relu(z[2])
act_2_pooled = tf.nn.max_pool(act[2], ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'VALID')
flattened_2 = tf.reshape(act_2_pooled, [-1, act_2_pooled.shape[1] * act_2_pooled.shape[2] * act_2_pooled.shape[3] ])

#FC layer
z[3] = tf.matmul(flattened_2, weights[2]) + bias[2]
act[3] = tf.nn.relu(z[3])

#output layer
dropout_rate = tf.placeholder(dtype = tf.float32)
dropout = tf.nn.dropout(x = act[3], rate = dropout_rate)
z[4] = tf.matmul(dropout, weights[3]) + bias[3]
y_pred = tf.nn.softmax(z[4])

#apply L2 regularization
regulizer = tf.nn.l2_loss(weights[0]) + tf.nn.l2_loss(weights[1]) + tf.nn.l2_loss(weights[2]) + tf.nn.l2_loss(weights[3])

#make loss function
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits = z[4], labels = y_true)
loss = tf.reduce_mean(cross_entropy + lambda_term / data.train.images.shape[0] * regulizer)

#apply backprop
train_step = tf.train.GradientDescentOptimizer(learning_rate = learning_rate).minimize(loss)

#make predictions
y_pred_cls = tf.argmax(y_pred, dimension = 1)
y_true_cls = tf.argmax(y_true, dimension = 1)
correct_pred = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))



with tf.Session() as sess:
    num_epochs = 60
    batch_sz = 10
    sess.run(tf.global_variables_initializer())

    for i in range(num_epochs):
        random.shuffle(train_data)
        x_train = [x for x,y in train_data[:]]
        y_train = [y for x,y in train_data[:]]

        for j in range( int ( len(train_data) / batch_sz)):
            x_batch = x_train[j * batch_sz : (j + 1) * batch_sz]
            y_batch = y_train[j * batch_sz : (j + 1) * batch_sz]
            feed_dict = {x: x_batch, y_true: y_batch, dropout_rate: dropout_0}
            sess.run(train_step, feed_dict)
        feed_dict = {x: x_test, y_true: y_test, dropout_rate: 0.0}

        print("Epoch %s complete" %(i+1))
        loss_val_train = sess.run(loss, {x: x_train, y_true: y_train, dropout_rate: 0.0})
        loss_val_test = sess.run(loss, feed_dict)
        print("Train loss: %s, Test loss: %s" %(loss_val_train, loss_val_test))
        print("Train accuracy %s, Test accuracy: %s" %(accuracy.eval({x: x_train, y_true: y_train, dropout_rate: 0.0}), accuracy.eval(feed_dict)))
        print()
