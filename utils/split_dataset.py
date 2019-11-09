import numpy as np


def split_dataset(dataframe, chunk_size, target=1):
    X = dataframe[0: len(dataframe) - (len(dataframe) % chunk_size)]
    X_batches = X.reshape(-1, chunk_size, 1)

    Y = dataframe[1: len(dataframe) - (len(dataframe) % chunk_size) + target]
    Y_batches = Y.reshape(-1, chunk_size, 1)

    X_val = dataframe[-(chunk_size + target):]
    X_val = X_val[:chunk_size]
    X_val = X_val.reshape(-1, chunk_size, 1)

    y_val = dataframe[-chunk_size:]
    y_val = y_val.reshape(-1, chunk_size, 1)

    return X_batches, Y_batches, X_val, y_val
