import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import RMSprop
np.random.seed(133)

# download the mnist dataset to the path '~/.keras/datasets/' if it is the first time to be called
(list_x_train, list_y_train), (list_x_test, list_y_test) = mnist.load_data()

# data pre-processing
list_x_train = list_x_train.reshape(list_x_train.shape[0], -1) / 255
list_x_test = list_x_test.reshape(list_x_test.shape[0], -1) / 255
list_y_train = np_utils.to_categorical(list_y_train, num_classes = 10)
list_y_test = np_utils.to_categorical(list_y_test, num_classes=10)

# another way to build your nn
model = Sequential([
    Dense(32, input_dim=784),
    Activation('relu'),
    Dense(10),
    Activation('softmax')
])

model.summary()

