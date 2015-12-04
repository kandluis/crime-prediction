import numpy as np
import pandas as pd
import csv
import sys

'''
Utility Functions
'''
# DATA: months since 2012, X coord, Y coord
# if split size = 0, do non
def split(X, tr_size):
    n_col = np.shape(X)[1]
    if tr_size != 0:
        Y = np.copy(X)
        np.random.shuffle(Y)
        break_pt = tr_size * np.shape(X)[0]
        train, test = Y[:break_pt,:], Y[break_pt:,:]
    else:
        break_pt = (3500./4400.) * np.shape(X)[0]
        train, test = X[:break_pt,:], X[break_pt:,:]

    tr_t, te_t = train[:,n_col-1], test[:,n_col-1]
    tr, te = train[:,range(n_col-1)], test[:,range(n_col-1)]
    return tr, tr_t, te, te_t

# implementation notes: set NaN to mean
def normalize_features(X_train):
    mean_X_train = np.nanmean(X_train, 0)
    for i in xrange(np.shape(X_train)[1]):
        col = X_train[:,i]
        col[ np.isnan(col) ] = mean_X_train[i]
    std_X_train = np.std(X_train, 0)
    std_X_train[ std_X_train == 0 ] = 1
    X_train_normalized = (X_train - mean_X_train) / std_X_train
    return X_train_normalized

# Note: bucket edits in place
def bucket(X, cols, num_buckets):
    Y = np.copy(X)
    for col in cols:
        buckets = np.linspace(np.min(X[:,col]), np.max(X[:,col]), num=num_buckets+1)
        for i in xrange(num_buckets):
            X_col = Y[:,col]
            X_col[ (buckets[i] <= X_col) & (X_col <= buckets[i+1])] = i
            Y[:,col] = X_col
    return Y

def rmse(predict, true):
    return np.sqrt(1.0/np.shape(predict)[0] * np.sum(np.square(predict - true)))

'''
Read in data
'''

import re
import warnings

def readData():

    bos_file = '/home/luis/boston.csv'
    target_type = str  # The desired output type

    with warnings.catch_warnings(record=True) as ws:
      warnings.simplefilter("always")

      bos_data = pd.read_csv(bos_file, sep=",", header=0)
      print("Warnings raised:", ws)
      # We have an error on specific columns, try and load them as string
      for w in ws:
          s = str(w.message)
          print("Warning message:", s)
          match = re.search(r"Columns \(([0-9,]+)\) have mixed types\.", s)
          if match:
              columns = match.group(1).split(',') # Get columns as a list
              columns = [int(c) for c in columns]
              print("Applying %s dtype to columns:" % target_type, columns)
              bos_data.iloc[:,columns] = bos_data.iloc[:,columns].astype(target_type)

    '''
    Featurize data
    '''
    # temporal features
    # day of week
    day = np.array(bos_data.DAY_WEEK)
    day[ day == "Sunday"] = 0
    day[ day == "Monday"] = 1
    day[ day == "Tuesday"] = 2
    day[ day == "Wednesday"] = 3
    day[ day == "Thursday"] = 4
    day[ day == "Friday"] = 5
    day[ day == "Saturday"] = 6

    # Split mm/dd/yyyy xx:yy:zz AM/PM into components
    date_time = np.array([x.split() for x in bos_data.FROMDATE])
    date = date_time[:,0]
    time = date_time[:,1]
    tod = date_time[:,2]

    # month, day, year
    date = np.array([x.split('/') for x in date])
    month = [int(x) for x in date[:,0]]
    dom = [int(x) for x in date[:,1]]
    year = [int(x) for x in date[:,2]]
    # months since Jan 2012
    time_feat = np.subtract(year, 2012)*12 + month

    # time of day
    time_c = [x.split(':') for x in time]
    time = [int(x[1]) if (y == 'AM' and int(x[0]) == 12) else 60*int(x[0])+int(x[1])
          if (y =='AM' and int(x[0]) != 12) or (int(x[0]) == 12 and y == 'PM') else 12*60+60*int(x[0])+int(x[1])
          for x,y in zip(time_c, tod)]

    # grab the features we want
    data_unnorm = np.transpose(np.vstack((time_feat, bos_data.X, bos_data.Y))).astype(float)
    # remove NaNs
    good_data = data_unnorm[~(np.isnan(data_unnorm[:,1]))]

    return good_data

'''
Count data for each cell. If logSpace is true, returns log values.
'''
good_data = readData()
def createBuckets(n_buckets = 15, logSpace = True):
    data_b = bucket(good_data, [1, 2], n_buckets)

    years = [2012, 2013, 2014, 2015]
    n_time = int(data_b[np.argmax(data_b[:,0])][0])

    # buckets = np.zeros((n_time, n_buckets, n_buckets))
    buckets2 = np.zeros((n_buckets * n_buckets * n_time, 4))

    # divide the data up by year and month
    for i in xrange(n_time):
        for j in xrange(n_buckets):
            for k in xrange(n_buckets):
                count = data_b[ (data_b[:,0] == i+1) &
                                (data_b[:,1] == j) &
                                (data_b[:,2] == k)]
                #print count
                #print count.shape
                # buckets[i][j][k] = np.size(count,0)
                buckets2[i*(n_buckets * n_buckets)+j*(n_buckets)+k, 0] = i
                buckets2[i*(n_buckets * n_buckets)+j*(n_buckets)+k, 1] = j
                buckets2[i*(n_buckets * n_buckets)+j*(n_buckets)+k, 2] = k
                buckets2[i*(n_buckets * n_buckets)+j*(n_buckets)+k, 3] = np.size(count,0)
    print np.shape(buckets2)

    if logSpace:
        buckets2[:,3] = np.log(np.add(sys.float_info.epsilon, buckets2[:,3]))

    return buckets2

'''
Our GP
    other implementations:
    - scikit-learn
    - GPy
'''

# compute the kernel matrix
# use square exponential by default
def ker_se(x, y, l, horz = 1.0):

    n = np.shape(x)[0]
    m = np.shape(y)[0]

    t = np.reshape(x, (np.shape(x)[0], 1, np.shape(x)[1]))
    s = np.reshape(y, (1, np.shape(y)[0], np.shape(y)[1]))

    # tile across columns
    cols = np.tile(t, (1, m, 1))
    # tile across rows
    rows = np.tile(s, (n, 1, 1))
    # get the differences and vectorize
    diff_vec = np.reshape(cols - rows, (n*m, np.shape(t)[2]))

    M = np.diag(l)
    # print l
    # print M

    # use multiply and sum to calculate matrix product
    print M.shape
    print diff_vec.shape
    s = np.multiply(-.5, np.sum(np.multiply(diff_vec, np.transpose(np.dot(M, np.transpose(diff_vec)))), axis=1))
    se = np.reshape(np.multiply(horz, np.exp(s)), (n, m))

    return se

'''
Calculate kernels
'''
def GaussianProcess(train, train_t, test, test_t, l,
                    horz, sig_eps, predict=True, rmse=True, ker='se'):
    # Try to be memory efficient by deleting data after use!
    if ker == 'se':
        ker_fun = ker_se
    else:
        raise Exception("Kernal {} Not Supported!".format(ker))

    ker1 = ker_fun(train, train, l, horz)
    L = np.linalg.cholesky(ker1 + np.multiply(sig_eps, np.identity(np.shape(ker1)[0])))

    alpha = np.linalg.solve(L.T, np.linalg.solve(L, train_t))

    # Only do this if we request the predictions or rmse
    ret = []
    if predict or rmse:
        ker2 = ker_fun(train,test, l, horz)
        preds = np.dot(np.transpose(ker2), alpha)
        del ker2
        ret.append(preds)


    # Only if we request the rmse
    if rmse:
        npreds = preds / float(preds.sum())
        ntest_t = test_t / float(test_t.sum())
        rmse_val = np.sqrt(np.sum(np.square(npreds - ntest_t))/np.shape(preds)[0])
        print rmse
        ret.append(rmse_val)

    # Calculate the marginal likelihood
    likelihood = -.5 * np.dot(np.transpose(train_t), alpha) - np.sum(np.log(np.diagonal(L))) - np.shape(ker1)[0]/2 * np.log(2*np.pi)
    ret.append(likelihood)

    del alpha
    del L
    del ker1

    return tuple(ret)

def optimizeGaussianProcess(n, l1, l2, l3, horz, sig_eps):
    # Bucketize the data as specified! By default, does Boston data.
    data = createBuckets(n_buckets=n)
    print "Created Data!"

    # Split for latest year.
    train, train_t, test, test_t = split(data, 0)
    print "Split Data!"

    # Calculate the likelihood
    l = [l1, l2, l3]
    likelihood = GaussianProcess(train, train_t, test, test_t,
                                 l, horz, sig_eps,
                                 predict = False, rmse = False)[0]
    # The objective actually minimizes!
    print "likelihood of {}".format(likelihood)
    return -1 * likelihood

# Write a function like this called 'main'
def main(job_id, params):
    print 'Job #%d' % job_id
    print params
    return optimizeGaussianProcess(
      params['n'][0], params['l1'][0], params['l2'][0],
      params['l3'][0], params['horz'][0], params['sig_eps'][0])
