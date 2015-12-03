'''
Utility Functions.
TODO(nautilik): These functions have currently only been tested with the Boston
data set!

Authors:
    Alex Wang (alexwang@college.harvard.edu)
    Luis Perez (luis.perez.live@gmail.com)
Copyright 2015, Harvard University
'''

import numpy as np
import sys

# The mapping of data matrix columns to indexes.
columns = {'t': 0, 'x': 1, 'y': 2, 'count': 3}


def split(X, tr_size):
    '''
    Splits input matrix X. If tr_size = 0, splits by final year.
    Note that the ratio for the final year is currently hard-coded!
    '''
    n_col = np.shape(X)[1]
    if tr_size != 0:
        Y = np.copy(X)
        np.random.shuffle(Y)
        break_pt = tr_size * np.shape(X)[0]
        train, test = Y[:break_pt, :], Y[break_pt:, :]
    else:
        break_pt = (3500. / 4400.) * np.shape(X)[0]
        train, test = X[:break_pt, :], X[break_pt:, :]

    tr_t, te_t = train[:, n_col - 1], test[:, n_col - 1]
    tr, te = train[:, range(n_col - 1)], test[:, range(n_col - 1)]
    return tr, tr_t, te, te_t


def normalize_features(X_train):
    '''
    Implementation notes: set NaN to mean.
    Generally normalizes X_train across all columns.
    '''
    mean_X_train = np.nanmean(X_train, 0)
    for i in xrange(np.shape(X_train)[1]):
        col = X_train[:, i]
        col[np.isnan(col)] = mean_X_train[i]
    std_X_train = np.std(X_train, 0)
    std_X_train[std_X_train == 0] = 1
    X_train_normalized = (X_train - mean_X_train) / std_X_train
    return X_train_normalized


def bucket(X, cols, num_buckets):
    '''
    Note: bucket edits in place
    '''
    Y = np.copy(X)
    for col in cols:
        buckets = np.linspace(np.min(X[:, col]), np.max(
            X[:, col]), num=num_buckets + 1)
        for i in xrange(num_buckets):
            X_col = Y[:, col]
            X_col[(buckets[i] <= X_col) & (X_col <= buckets[i + 1])] = i
            Y[:, col] = X_col
    return Y


def rmse(predict, true):
    '''
    Root mean square error between the predictions and the true value.
    '''
    return np.sqrt(1.0 / np.shape(predict)[0] * np.sum(np.square(predict - true)))


def createBuckets(good_data, n_buckets=15, logSpace=True):
    '''
    Count data for each cell. If logSpace is true, returns log values.
    '''

    data_b = bucket(good_data, [1, 2], n_buckets)

    n_time = int(data_b[np.argmax(data_b[:, 0])][0])

    # buckets = np.zeros((n_time, n_buckets, n_buckets))
    buckets2 = np.zeros((n_buckets * n_buckets * n_time, 4))

    # divide the data up by year and month
    for i in xrange(n_time):
        for j in xrange(n_buckets):
            for k in xrange(n_buckets):
                count = data_b[(data_b[:, 0] == i + 1) &
                               (data_b[:, 1] == j) &
                               (data_b[:, 2] == k)]
                # buckets[i][j][k] = np.size(count,0)
                buckets2[i * (n_buckets * n_buckets) +
                         j * (n_buckets) + k][0] = i
                buckets2[i * (n_buckets * n_buckets) +
                         j * (n_buckets) + k][1] = j
                buckets2[i * (n_buckets * n_buckets) +
                         j * (n_buckets) + k][2] = k
                buckets2[i * (n_buckets * n_buckets) + j *
                         (n_buckets) + k][3] = np.size(count, 0)
    print np.shape(buckets2)

    if logSpace:
        buckets2[:, 3] = np.log(np.add(sys.float_info.epsilon, buckets2[:, 3]))

    return buckets2
