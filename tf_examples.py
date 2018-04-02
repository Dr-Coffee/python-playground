from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

def train_input_fn(features, labels, batch_size):
    # Convert the inputs to a Dataset
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
    # Shuffle, repeat, and batch the examples
    return dataset.shuffle(1000).repeat().batch(batch_size)




