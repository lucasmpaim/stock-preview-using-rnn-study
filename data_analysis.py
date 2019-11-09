import pandas as pd

from utils.data_frame_utils import rename_frame
import matplotlib.pyplot as plt

data_frame = pd.read_csv('data/stocks.csv')

petr4_frame = data_frame.iloc[1:, 3:5]
rename_frame(petr4_frame)

petr4_frame.plot(legend=False)
plt.show()
