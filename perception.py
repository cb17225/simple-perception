import pandas as pd
import numpy as np
import tensorflow as tf

# train_data = np.genfromtxt(<filename>, delimiter=',')
xTrain, yTrain = train_data[0], train_data[1]

model = tf.keras.models.Sequential([tf.keras.layers.Dense(64, activation = 'relu', input_shape = (1,)), tf.keras.layers.Dense(64, activation = 'relu'), tf.keras.layers.Dense(1)])
model.summary()
opt = tf.keras.optimizers.Adam(learning_rate = 0.02)

model.compile(optimizer=opt, loss='MeanSquaredError')
history = model.fit(x = xTrain, y = yTrain, epochs=1000, batch_size = 500)

import matplotlib.pyplot as plt
print(history.history.keys())

plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()

predicted_values = model.predict([5,6,7])
print(predicted_values)