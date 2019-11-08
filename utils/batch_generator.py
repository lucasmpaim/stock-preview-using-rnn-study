import numpy as np


def batch_generator(dataframe, chunk_size, target=1):
    X = dataframe[0: len(dataframe) - (len(dataframe) % chunk_size)]
    X_batches = X.reshape(-1, chunk_size, 1)

    Y = dataframe[1: len(dataframe) - (len(dataframe) % chunk_size) + target]
    Y_batches = Y.reshape(-1, chunk_size, 1)

    X_test = dataframe[-(chunk_size + target):]
    X_test = X_test[:chunk_size]
    X_test = X_test.reshape(-1, chunk_size, 1)

    y_test = dataframe[-chunk_size:]
    y_test = y_test.reshape(-1, chunk_size, 1)

    return X_batches, Y_batches, X_test, y_test
