import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
np.random.seed(1337)

list_x = np.linspace(-1,1,200)
np.random.shuffle(list_x)
list_y = 0.5 * list_x + 2 + np.random.normal(0, 0.05, (200,))

plt.scatter(list_x, list_y)
plt.show()

list_x_train, list_y_train = list_x[:160], list_y[:160]
list_x_test, list_y_test = list_x[160:], list_y[160:]

model = Sequential()
model.add(
    Dense(output_dim=1, input_dim=1)
)

model.compile(
    loss='mse',
    optimizer='sgd'
)

print("Training----------------------------------------------")
for step in range(301):
    cost = model.train_on_batch(list_x_train, list_y_train)
    if step % 100 == 0:
        print("training cost:", cost)

print("Testing-----------------------------------------------")
w, b = model.layers[0].get_weights()
print("weights=", w, " biases=", b)
cost = model.evaluate(list_x_test, list_y_test, batch_size=40)
print("test cost:", cost)

list_y_predicted = model.predict(list_x_test)
plt.scatter(list_x_test, list_y_test)
plt.plot(list_x_test, list_y_predicted)
plt.show()

















