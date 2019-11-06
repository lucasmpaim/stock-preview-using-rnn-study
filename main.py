import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils.data_frame_utils import rename_frame

data_frame = pd.read_csv('data/stocks.csv')


ibov_frame = data_frame.iloc[1:, 0:2]
rename_frame(ibov_frame)

petr4_frame = data_frame.iloc[1:, 3:5]
rename_frame(petr4_frame)

dolar_frame = data_frame.iloc[1:, 9:11]
rename_frame(dolar_frame)

plt.plot(range(0, ibov_frame.shape[0]),
         [y for y in ibov_frame['Close']],
         label='Ibov', alpha=0.5)

plt.plot(range(0, dolar_frame.shape[0]),
         [y for y in dolar_frame['Close']],
         label='Dólar', alpha=0.5)

plt.plot(range(0, petr4_frame.shape[0]),
         [y for y in petr4_frame['Close']],
         label='Petr4')

plt.legend()
plt.xlabel('Days after 05/09/2006')
plt.ylabel('Normalized Variation')
plt.title('Petr4 x Dólar x Ibov')
plt.show()
