import datetime

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow_core.python.keras.layers.core import Dense, Dropout
from tensorflow_core.python.keras.layers.recurrent import LSTM
from tensorflow_core.python.keras.utils.vis_utils import plot_model

from utils.batch_generator import batch_generator
from utils.data_frame_utils import rename_frame

data_frame = pd.read_csv('data/stocks.csv')

ibov_frame = data_frame.iloc[1:, 0:2]
rename_frame(ibov_frame)

petr4_frame = data_frame.iloc[1:, 3:5]
rename_frame(petr4_frame)

dolar_frame = data_frame.iloc[1:, 9:11]
rename_frame(dolar_frame)

TRAIN_SPLIT = 3000

# plt.plot(range(0, ibov_frame.shape[0]),
#          [y for y in ibov_frame['Close']],
#          label='Ibov', alpha=0.5)
#
# plt.plot(range(0, dolar_frame.shape[0]),
#          [y for y in dolar_frame['Close']],
#          label='Dólar', alpha=0.5)
#
# plt.plot(range(0, petr4_frame.shape[0]),
#          [y for y in petr4_frame['Close']],
#          label='Petr4')
#
# plt.legend()
# plt.xlabel('Days after 05/09/2006')
# plt.ylabel('Normalized Variation')
# plt.title('Petr4 x Dólar x Ibov')
# plt.show()

# configure seed's
tf.random.set_seed(22)
np.random.seed(22)

features = pd.DataFrame()
# features['Ibov'] = ibov_frame['Close'].to_numpy()
# features['Dolar'] = dolar_frame['Close'].to_numpy()
features['Petr4'] = petr4_frame['Close'].to_numpy()

print(features.head())
print(features.shape)
# features.plot(subplots=True)
# plt.show()

tf.random.set_seed(13)

past_history = 5 * 10
future_target = 1

dataset = features.values
X_train, y_train, X_test, y_test = batch_generator(dataset,
                                                   past_history,
                                                   future_target)

model = tf.keras.Sequential()
hidden_layer_size = 100
model.add(
    LSTM(hidden_layer_size, activation=tf.nn.relu,
         return_sequences=True,
         input_shape=(X_train.shape[1], X_train.shape[2])),
)
model.add(
    Dropout(0.2)
)
model.add(
    LSTM(hidden_layer_size, activation=tf.nn.relu,
         return_sequences=True,
         input_shape=(X_train.shape[1], X_train.shape[2])),
)
model.add(
    Dropout(0.2)
)
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')

EPOCHS = 300

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

model.fit(X_train, y_train, epochs=EPOCHS, batch_size=32,
          callbacks=[tensorboard_callback])

predicted = model.predict(X_test)

predicted2 = np.ravel(predicted)
y_test2 = np.ravel(y_test)

plt.plot(range(0, past_history), predicted2, '*', label='predict')
plt.plot(range(0, past_history), y_test2, 'x', label='real')
plt.legend()
plt.show()

loss, acc = model.evaluate(X_test,  y_test, verbose=2)
print("Untrained model, accuracy: {:5.2f}%".format(100*acc))


model.save_weights('models/deep_weights.rnn')

# EVALUATION_INTERVAL = 7 * 3
# single_step_history = model.fit(X_train,
#                                 epochs=EPOCHS,
#                                 steps_per_epoch=EVALUATION_INTERVAL,
#                                 validation_data=X_test,
#                                 validation_steps=2,
#                                 callbacks=[tensorboard_callback])

# def plot_train_history(history, title):
#     loss = history.history['loss']
#     val_loss = history.history['val_loss']
#
#     epochs = range(len(loss))
#
#     plt.figure()
#
#     plt.plot(epochs, loss, 'b', label='Training loss')
#     plt.plot(epochs, val_loss, 'r', label='Validation loss')
#     plt.title(title)
#     plt.legend()
#
#     plt.show()
#
#
# plot_train_history(single_step_history,
#                    'Single Step Training and validation loss')
#
# plot_model(single_step_model)
