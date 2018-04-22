import tensorflow as tf
import os
import numpy as np
import argparse
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=100, type=int, help='batch size')
parser.add_argument('--train_steps', default=1000, type=int,
                    help='number of training steps')

def input_evaluation_set():
    x_data = np.random.rand(100).astype(np.float32)
    y_data = x_data * 0.1 + 0.3
    features = {'x':x_data}
    labels = y_data
    return features, labels

def my_model(features, labels, mode, params):
    weights = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
    biases = tf.Variable(tf.zeros([1]))
    #y = weights * 1.0 + biases

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

def main(argv):
    args = parser.parse_args(argv[1:])
    # Fetch the data
    x_data = np.random.rand(100).astype(np.float32)
    y_data = x_data * 0.1 + 0.3


    # Feature columns describe how to use the input.
    #my_feature_columns = []
    #for key in train_x.keys():
    #    my_feature_columns.append(tf.feature_column.numeric_column(key=key))

    # Build 2 hidden layer DNN with 10, 10 units respectively.
    #classifier = tf.estimator.Estimator(
    #    model_fn=my_model,
    #    params={
    #        'feature_columns': my_feature_columns,
    #        # Two hidden layers of 10 nodes each.
    #        'hidden_units': [10, 10],
    #        # The model must choose between 3 classes.
    #        'n_classes': 3,
    #    })

    # Train the Model.
    #classifier.train(
    #    input_fn=lambda:iris_data.train_input_fn(train_x, train_y, args.batch_size),
    #    steps=args.train_steps)

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)


