import numpy as np


def rename_frame(df):
    df.rename(columns={df.columns[0]: 'Date',
                       df.columns[1]: 'Close'},
              inplace=True)
    df['Close'] = df['Close'].astype(np.float)
    df['Close'] = normalize(df['Close'])


def normalize(serie):
    return (serie - serie.mean()) / np.abs(serie.max())

