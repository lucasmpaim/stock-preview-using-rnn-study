import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils.batch_generator import multivariate_data
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
features['Ibov'] = ibov_frame['Close'].to_numpy()
features['Dolar'] = dolar_frame['Close'].to_numpy()
features['Petr4'] = petr4_frame['Close'].to_numpy()

print(features.head())
print(features.shape)
features.plot(subplots=True)
plt.show()

TRAIN_SPLIT = 3000
tf.random.set_seed(13)

past_history = 720
future_target = 72
STEP = 6  # 6 observations == one hour

x_train_single, y_train_single = multivariate_data(features, features[:, 1], 0,
                                                   TRAIN_SPLIT, past_history,
                                                   future_target, STEP,
                                                   single_step=True)
x_val_single, y_val_single = multivariate_data(features, features[:, 1],
                                               TRAIN_SPLIT, None, past_history,
                                               future_target, STEP,
                                               single_step=True)

BATCH_SIZE = 256
BUFFER_SIZE = 10000

train_data_single = tf.data.Dataset.from_tensor_slices((x_train_single, y_train_single))
train_data_single = train_data_single.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

val_data_single = tf.data.Dataset.from_tensor_slices((x_val_single, y_val_single))
val_data_single = val_data_single.batch(BATCH_SIZE).repeat()

single_step_model = tf.keras.models.Sequential()
single_step_model.add(tf.keras.layers.LSTM(32,
                                           input_shape=x_train_single.shape[-2:]))
single_step_model.add(tf.keras.layers.Dense(1))

single_step_model.compile(optimizer=tf.keras.optimizers.RMSprop(), loss='mae')

EVALUATION_INTERVAL = 200
EPOCHS = 100

single_step_history = single_step_model.fit(train_data_single, epochs=EPOCHS,
                                            steps_per_epoch=EVALUATION_INTERVAL,
                                            validation_data=val_data_single,
                                            validation_steps=50)


def plot_train_history(history, title):
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(loss))

    plt.figure()

    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title(title)
    plt.legend()

    plt.show()


plot_train_history(single_step_history,
                   'Single Step Training and validation loss')
