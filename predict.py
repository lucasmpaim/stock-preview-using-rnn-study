import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tabulate import tabulate
from tensorflow_core.python.keras.layers.core import Dense, Dropout
from tensorflow_core.python.keras.layers.recurrent import LSTM

from utils.data_frame_utils import rename_frame, undo_normalize, normalize
from utils.split_dataset import split_dataset

data_frame = pd.read_csv('data/stocks.csv')

petr4_frame = data_frame.iloc[1:, 3:5]
rename_frame(petr4_frame)
normalized = petr4_frame.copy()
normalized['Close'] = normalize(normalized['Close'])

TRAIN_SPLIT = 3000
# configure seed's
tf.random.set_seed(22)
np.random.seed(22)

features = pd.DataFrame()
features['Petr4'] = normalized['Close'].to_numpy()

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
plt.plot(range(len(X_val[0]), len(X_val[0]) + past_history), y_val2, 'o', label='real', alpha=0.5)
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
