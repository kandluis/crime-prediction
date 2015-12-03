
# coding: utf-8

# In[ ]:

import numpy as np
import pandas as pd
import csv
import sklearn.linear_model as lm
import matplotlib.pyplot as plt
import seaborn as sns
import sys


# In[ ]:

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


# In[ ]:

'''
Read in data
'''

import re
import warnings

bos_file = '../data/boston.csv'
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


# In[ ]:

'''
Count data for each cell. If logSpace is true, returns log values.
'''
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
                # buckets[i][j][k] = np.size(count,0)
                buckets2[i*(n_buckets * n_buckets)+j*(n_buckets)+k][0] = i
                buckets2[i*(n_buckets * n_buckets)+j*(n_buckets)+k][1] = j
                buckets2[i*(n_buckets * n_buckets)+j*(n_buckets)+k][2] = k
                buckets2[i*(n_buckets * n_buckets)+j*(n_buckets)+k][3] = np.size(count,0)
    print np.shape(buckets2)
    
    if logSpace:
        buckets2[:,3] = np.log(np.add(sys.float_info.epsilon, buckets2[:,3]))
    
    return buckets2


# In[ ]:

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
    
    # use multiply and sum to calculate matrix product
    s = np.multiply(-.5, np.sum(np.multiply(diff_vec, np.transpose(np.dot(M, np.transpose(diff_vec)))), axis=1))
    se = np.reshape(np.multiply(horz, np.exp(s)), (n, m))
    
    return se


# In[ ]:

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


# In[ ]:

columns = { 't' : 0, 'x' : 1, 'y' : 2, 'count' : 3}
def createHeatMap(X):
    '''
    Given a data set, creates a heatmap of it based on x,y coordinates.
    Ignore the temporal feature. You should subset the data before passing
    it into this function if you'd like a heatmap for a specific time period.
    '''
    n = X[:, columns['x']].astype(int).max()
    m = X[:, columns['y']].astype(int).max()
    heatmap = np.zeros((n,m))
    for i in xrange(n):
        for j in xrange(m):
            total = X[(X[:, columns['x']] == i) & 
                      (X[:, columns['y']] == j), columns['count']].sum()
            if total > 0:
                heatmap[i,j] = total
    heatmap = heatmap / float(heatmap.sum())
    return heatmap


# In[ ]:

# Make some plots for n = 15 for GP process
def plotDistribution(predict, true, city, n, process='GP'):
    minValue = min(len(predict), 100)
    yPred = predict[-minValue:]
    yTrue = true[-minValue:]
    yPred = yPred / float(np.sum(yPred))
    yTrue = yTrue / float(np.sum(yTrue))
    plt.clf()
    plt.plot(yPred, label="Predictions")
    plt.plot(yTrue, label="Actual Data")
    plt.title('Predictive Distribution for {}'.format(process))
    plt.xlabel('Compressed Features')
    plt.ylabel('Probability')
    plt.legend()
    plt.savefig('../figures/{}_results/{}_n={}_periods={}.png'.format(
        city, process, n,12))
    plt.close()


# In[ ]:

def plotHeatMaps(X_test, predict, city, n, process='GP'):
    # Attach the predictions to the data
    trueValues = np.copy(X_test)
    predictedValues = np.copy(X_test)
    predictedValues[:, columns['count']] = predict

    # Now we want to plot the heatmaps for the predictions/actual data
    # by time period
    months = np.unique(X_test[:, columns['t']])
    for month in months:
        # Create the heatmaps 
        selected = (X_test[:, columns['t']] == month)
        if selected.sum() > 0:
            plt.clf()
            m = createHeatMap(trueValues[selected, :])
            if m.sum() > 0:
                sns.heatmap(m)
                plt.title('True Density Distribution in Month {}'.format(month))
                plt.savefig('../figures/{}_results/{}_heatmap_true_n={}_t={}.png'.format(
                    city, process, n, month))
                plt.close()

            plt.clf()
            m = createHeatMap(predictedValues[selected, :])
            if m.sum() > 0:
                sns.heatmap(m)
                plt.title('Predicted Density Distribution in Month {}'.format(month))
                plt.savefig('../figures/{}_results/{}_heatmap_pred_n={}_t={}.png'.format(
                    city, process, n, month))
                plt.close()


# In[ ]:

log = False
get_ipython().magic(u'time data = createBuckets(n_buckets=15, logSpace=log)  # default is 15, logSpace=True')
get_ipython().magic(u'time train, train_t, test, test_t = split(data, 0)')
sig_eps = train_t.std()


# In[ ]:

# These are the optimal parameters for n = 10 (we can try with other values too)
l = [9620.11949755, 9620.11949755, 9620.11949755]
horz = 0.82754075018
sig_eps = train_t.std()
get_ipython().magic(u'time predictions, rmse, likelihood = GaussianProcess(train, train_t, test, test_t, l, horz, sig_eps)')


# In[ ]:

# Only do the below if logspace !
if log:
    test_t = np.exp(test_t)
    predictions = np.exp(predictions)


# In[ ]:

plotDistribution(predictions, test_t, 'boston', 15, process='GPSEOptimzied')


# In[ ]:

np.diag([1,2,3])


# In[ ]:

X_test = np.zeros((test.shape[0], test.shape[1] + 1)).astype(int)
X_test[:,:-1] = test
X_test[:,-1] = test_t


# In[ ]:

plotHeatMaps(X_test, predictions, 'boston', 15, process='GPSELog')


# In[ ]:

'''
Easier method for calling our GP model! Kernal defaults to SE.
'''
def optimizeGaussianProcess(n, l1, l2, l3, horz, sig_eps,
                            log=False):
    # Bucketize the data as specified! By default, does Boston data.
    data = createBuckets(n, logSpace=log)
    
    # Split for latest year.
    train, train_t, test, test_t = split(data, 0)
    
    # Calculate the likelihood
    l = [l1,l2,l3]
    likelihood = GaussianProcess(train, train_t, test, test_t,
                                 l, horz, sig_eps,
                                 predict = False, rmse = False)
    return likelihood


# In[ ]:

# Collect likelihoods for different n values
testN = range(2,10) + range(10,20,5)
likelihoods = []
for n in testN:
    likelihood = optimizeGaussianProcess(n, 1.0, 1.0, 1.0, 1.0, 1.0,
                                        log=False)
    likelihoods.append(likelihood)


# In[ ]:

x = testN
y = likelihoods
line1 = plt.plot(x, y, label="Log Likelihood")
plt.title('GP Predictions for Boston')
plt.xlabel('Dimension of Grid')
plt.ylabel('Log Likelihood')
plt.legend()
plt.show()


# In[ ]:

train_t = train_t.reshape((train_t.shape[0], 1))


# In[ ]:

'''
Smart GP
'''

import GPy as gp


# In[ ]:

kern = gp.kern.RBF(input_dim=3, variance=1., lengthscale=1.)


# In[ ]:

from IPython.display import display


# In[ ]:

train_t = train_t.reshape((train_t.shape[0], 1))
get_ipython().magic(u'time m = gp.models.GPRegression(train, train_t, kern)')


# In[ ]:

display(m)


# In[ ]:

# We fix the Gaussian_noise.variance to the std of the training data!
m.Gaussian_noise.variance.constrain_fixed(train_t.std())


# In[ ]:

display(m)


# In[ ]:

# We're going to constrain some 
m.optimize(messages=True, max_iters=100)
# Total of 110 iterations are being run :)


# In[ ]:

display(m)


# In[ ]:

predictions_optimal = m.predict(test)


# In[ ]:

preds2 = predictions_optimal[0]


# In[ ]:

plotDistribution(preds2, test_t, 'boston', 10, process='GPSEOptimized')


# In[ ]:

len(preds2), len(test_t)


# In[ ]:

process = 'GPSEOptimzed'
city = 'boston'
n = 10
minValue = min(len(preds2), 100)
yPred = preds2[-minValue:]
yTrue = test_t[-minValue:]
yPred = yPred / float(np.sum(yPred))
yTrue = yTrue / float(np.sum(yTrue))


# In[ ]:

plt.clf()
plt.plot(yPred, label="Predictions")
plt.plot(yTrue, label="Actual Data")
plt.title('Predictive Distribution for {}'.format('process'))
plt.xlabel('Compressed Features')
plt.ylabel('Probability')
plt.legend()
plt.savefig('../figures/{}_results/{}_n={}_periods={}.png'.format(
    city, process, n,12))
plt.close()


# In[ ]:

plt.plot(yPred)


# In[ ]:

plt.show()


# In[ ]:



