import numpy as np


#
#
# def batch_generator(dataframe, chunk_size):
#     l = dataframe.shape[0]
#     X, y = [], []
#
#     for ndx in range(0, l, chunk_size):
#         X.append(dataframe.iloc[ndx:ndx + chunk_size, :-1])
#         y.append(dataframe.iloc[ndx:ndx + chunk_size, -1:])
#
#     return X, y


def multivariate_data(dataset, target, start_index, end_index, history_size,
                      target_size, step, single_step=False):
    data = []
    labels = []

    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset) - target_size

    for i in range(start_index, end_index):
        indices = range(i - history_size, i, step)
        data.append(dataset[indices])

        if single_step:
            labels.append(target[i + target_size])
        else:
            labels.append(target[i:i + target_size])

    return np.array(data), np.array(labels)
