from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
import tensorflow as tf
import models.iris_data as iris_data

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=100, type=int,
                    help='batch size')
parser.add_argument('--train_steps', default=1000, type=int,
                    help='number of training steps')

def my_model(features, labels, mode, params):
    ''' DNN with three hidden layers '''
    # Create three fully connected layers each layer having a dropout
    # probability of 0.1
    net = tf.feature_column.input_layer(features,
                                        params['feature_columns'])
    for units in params['hidden_units']:
        net = tf.layers.dense(net, units=units,
                              activation=tf.nn.relu)

def train_input_fn(features, labels, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices(
        (dict(features), labels))
    dataset = dataset.shuffle(1000).repeat().batch(
        batch_size)
    return dataset.make_one_shot_iterator().get_next()

def main(argv):
    args = parser.parse_args(argv[1:])

    # Fetch the data
    (train_x, train_y), (test_x, test_y) = iris_data.load_data()