import datetime

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate
from tensorflow_core.python.keras.layers.core import Dense, Dropout
from tensorflow_core.python.keras.layers.recurrent import LSTM
from tensorflow_core.python.keras.utils.vis_utils import plot_model

from utils.split_dataset import split_dataset
from utils.data_frame_utils import rename_frame, undo_normalize, normalize

data_frame = pd.read_csv('data/stocks.csv')

petr4_frame = data_frame.iloc[1:, 3:5]
rename_frame(petr4_frame)
normalized = petr4_frame.copy()
normalized['Close'] = normalize(normalized['Close'])

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
features['Petr4'] = normalized['Close'].to_numpy()

print(features.head())
print(features.shape)
# features.plot(subplots=True)
# plt.show()

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
    .shuffle(batch_size).batch(batch_size).repeat()

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

model.load_weights('models/deep_weights.rnn')

predicted = model.predict(X_val)
predicted2 = [undo_normalize(y, petr4_frame['Close']) for y in np.ravel(predicted)]
y_val2 = [undo_normalize(y, petr4_frame['Close']) for y in np.ravel(y_val)]

plt.plot(range(0, past_history), [undo_normalize(x, petr4_frame['Close']) for x in X_val[0]], label='history')
plt.plot(range(len(X_val[0]), len(X_val[0]) + past_history), predicted2, '*', label='predict')
plt.plot(range(len(X_val[0]), len(X_val[0]) + past_history), y_val2, 'x', label='real')
plt.legend()
plt.show()

print(model.evaluate(X_val, y_val, verbose=2))


table = []
for predicted, true in zip(predicted2, y_val2):
    table.append([str(predicted), str(true), str(abs(predicted - true))])
print(tabulate(table))

global_mse = sum(
    [abs(predicted - true)
     for predicted, true in zip(predicted2, y_val2)]
) / len(predicted2)

print(f'Global MSE: {global_mse}')

predict_next_day_X = np.array(features[-25:]).reshape(-1, 25, 1)
result = model.predict(predict_next_day_X)[0]
print(f'Próximo valor: {undo_normalize(result, petr4_frame["Close"])}')
