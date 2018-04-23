import tensorflow as tf
import os
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def learn_single_function():
    x_data = np.random.rand(100).astype(np.float32)
    y_data = x_data * 0.1 + 0.3
    weights = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
    biases = tf.Variable(tf.zeros([1]))
    y = weights * x_data + biases
    loss = tf.reduce_mean(tf.square(y - y_data))
    optimizer = tf.train.GradientDescentOptimizer(0.5)
    train = optimizer.minimize(loss)
    init = tf.global_variables_initializer()
    # principles above
    sess = tf.Session()
    sess.run(init)
    for step in range(150):
        sess.run(train)
        if step % 20 == 0:
            print(step, sess.run(weights), sess.run(biases))

def add_layer(inputs, in_size, out_size, activation_function = None):
    weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    wx_plus_b = tf.matmul(inputs, weights) + biases
    if activation_function is None:
        outputs = wx_plus_b
    else:
        outputs = activation_function(wx_plus_b)
    return outputs

def main(argv):
    x_data = np.linspace(-1,1,300)[:,np.newaxis]
    noise = np.random.normal(0, 0.05, x_data.shape)
    y_data = np.square(x_data) - 0.5 + noise

    xs = tf.placeholder(tf.float32, [None, 1])
    ys = tf.placeholder(tf.float32, [None, 1])

    l1 = add_layer(xs, 1, 10,
                   activation_function=tf.nn.relu)
    predition = add_layer(l1, 10, 1, activation_function=None)
    loss = tf.reduce_mean(
        tf.reduce_sum(tf.square(ys - predition),
                      reduction_indices = [1]))
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)

    for i in range(1000):
        sess.run(train_step, feed_dict={xs:x_data, ys:y_data})
        if i % 50 == 0:
            print(sess.run(loss, feed_dict={xs: x_data, ys:y_data}))

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)


