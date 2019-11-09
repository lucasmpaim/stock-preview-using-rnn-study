import datetime

import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow_core.python.keras.layers.core import Dense, Dropout
from tensorflow_core.python.keras.layers.recurrent import LSTM

from utils.split_dataset import split_dataset
from utils.data_frame_utils import rename_frame, normalize

data_frame = pd.read_csv('data/stocks.csv')

ibov_frame = data_frame.iloc[1:, 0:2]
rename_frame(ibov_frame)
ibov_frame['Close'] = normalize(ibov_frame['Close'])

petr4_frame = data_frame.iloc[1:, 3:5]
rename_frame(petr4_frame)
petr4_frame['Close'] = normalize(petr4_frame['Close'])

dolar_frame = data_frame.iloc[1:, 9:11]
rename_frame(dolar_frame)
dolar_frame['Close'] = normalize(dolar_frame['Close'])

# configure seed's
tf.random.set_seed(22)
np.random.seed(22)

features = pd.DataFrame()
features['Petr4'] = petr4_frame['Close'].to_numpy()

print(features.head())
print(features.shape)

tf.random.set_seed(13)

past_history = 25
future_target = 1

dataset = features.values
X_train, y_train, X_val, y_val = split_dataset(dataset,
                                               past_history,
                                               future_target)

batch_size = 5

tensor_dataset_slice_train = tf.data.Dataset.from_tensor_slices((X_train, y_train))
tensor_dataset_slice_train = tensor_dataset_slice_train.cache() \
    .batch(batch_size).repeat()

tensor_dataset_slice_val = tf.data.Dataset.from_tensor_slices((X_val, y_val))
tensor_dataset_slice_val = tensor_dataset_slice_val.cache() \
    .batch(batch_size).repeat()

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

EPOCHS = 100

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

model.fit(X_train, y_train, epochs=EPOCHS,
          steps_per_epoch=125,
          validation_steps=25,
          validation_data=tensor_dataset_slice_val,
          callbacks=[tensorboard_callback])

model.save_weights('models/deep_weights.rnn')
