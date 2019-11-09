import numpy as np


def rename_frame(df):
    df.rename(columns={df.columns[0]: 'Date',
                       df.columns[1]: 'Close'},
              inplace=True)
    df['Close'] = df['Close'].astype(np.float)


def normalize(serie):
    return serie / np.abs(serie.max())


def undo_normalize(value, serie):
    return value * np.abs(serie.max())
