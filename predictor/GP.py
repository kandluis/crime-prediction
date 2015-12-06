'''
Our GP implementations.
    other implementations:
    - scikit-learn
    - GPy

Authors:
    Alex Wang (alexwang@college.harvard.edu)
    Luis Perez (luis.perez.live@gmail.com)
Copyright 2015, Harvard University
'''

import numpy as np
import GPy
import time
from . import util
from . import plot


def ker_se(x, y, l, horz=1.0):
    '''
    Compute the kernel matrix
    Use square exponential by default
    '''

    n = np.shape(x)[0]
    m = np.shape(y)[0]

    t = np.reshape(x, (np.shape(x)[0], 1, np.shape(x)[1]))
    s = np.reshape(y, (1, np.shape(y)[0], np.shape(y)[1]))

    # tile across columns
    cols = np.tile(t, (1, m, 1))
    # tile across rows
    rows = np.tile(s, (n, 1, 1))
    # get the differences and vectorize
    diff_vec = np.reshape(cols - rows, (n * m, np.shape(t)[2]))

    M = np.diag(l)

    # use multiply and sum to calculate matrix product
    s = np.multiply(-.5, np.sum(np.multiply(diff_vec,
                                            np.transpose(np.dot(M, np.transpose(diff_vec)))), axis=1))
    se = np.reshape(np.multiply(horz, np.exp(s)), (n, m))

    return se


def GaussianProcess(train, train_t, test, test_t, l,
                    horz, sig_eps, predict=True, rmse=True, ker='se'):
    '''
    Given the split data and parameters, train the GP with the specified kernel
    and return the specified results.

    At minimum, returns the log likelihood. If predict, returns predictions. If
    rmse, returns the RMSE between the prediction distribution and the true test
    data.

    Tries to be memory efficient.
    '''
    # Try to be memory efficient by deleting data after use!
    if ker == 'se':
        ker_fun = ker_se
    else:
        raise Exception("Kernal {} Not Supported!".format(ker))

    ker1 = ker_fun(train, train, l, horz)
    L = np.linalg.cholesky(
        ker1 + np.multiply(sig_eps, np.identity(np.shape(ker1)[0])))

    alpha = np.linalg.solve(L.T, np.linalg.solve(L, train_t))

    # Only do this if we request the predictions or rmse
    ret = []
    if predict or rmse:
        ker2 = ker_fun(train, test, l, horz)
        preds = np.dot(np.transpose(ker2), alpha)
        del ker2
        ret.append(preds)

    # Only if we request the rmse
    if rmse:
        npreds = preds / float(preds.sum())
        ntest_t = test_t / float(test_t.sum())
        rmse_val = util.rmse(npreds, ntest_t)
        print rmse
        ret.append(rmse_val)

    # Calculate the marginal likelihood
    likelihood = -.5 * np.dot(np.transpose(train_t), alpha) - np.sum(
        np.log(np.diagonal(L))) - np.shape(ker1)[0] / 2 * np.log(2 * np.pi)
    ret.append(likelihood)

    del alpha
    del L
    del ker1

    return tuple(ret)


def optimizeGaussianProcess(data, n, l1, l2, l3, horz, sig_eps,
                            log=False):
    '''
    Easier method for calling our GP model! Kernal defaults to SE.
    '''
    # Bucketize the data as specified! By default, does Boston data.
    data = util.createBuckets(data, n, logSpace=log)

    # Split for latest year.
    train, train_t, test, test_t = util.split(data, 0)

    # Calculate the likelihood
    l = [l1, l2, l3]
    likelihood = GaussianProcess(train, train_t, test, test_t,
                                 l, horz, sig_eps,
                                 predict=False, rmse=False)
    return likelihood


def run_gp(good_data, buckets, l, horz, sig_eps_f, logTransform,
           file_prefix, city, GPy=False):
    '''
    Runs our typical GP training process.
    '''
    # Split as specified by the user
    # default is 15, logSpace=True')
    start = time.clock()
    data = util.createBuckets(
        good_data, n_buckets=buckets, logSpace=logTransform)
    train, train_t, test, test_t = util.split(data, 0)
    end = time.clock()

    print "Finished splitting into specified regions in {}...".format(end - start)

    # Calculate sig_eps
    start = time.clock()
    sig_eps = sig_eps_f(train_t)

    # Run the gaussian process
    predictions, rmse, likelihood = GaussianProcess(
        train, train_t, test, test_t, l, horz, sig_eps)
    end = time.clock()

    print "Finishes training the GP Process, and generating the predictions in {}...".format(end - start)

    # Only do the below if logspace !
    if logTransform:
        test_t = np.exp(test_t)
        predictions = np.exp(predictions)

    # Save the results to boston
    plot.plotDistribution(predictions, test_t, city,
                          buckets, process=file_prefix)
    print "Finished plotting the distributions. Results are saved..."

    # Contatenate new test matrix -- this is the expected input.
    X_test = np.zeros((test.shape[0], test.shape[1] + 1)).astype(int)
    X_test[:, :-1] = test
    X_test[:, -1] = test_t

    plot.plotHeatMaps(X_test, predictions, city,
                      buckets, process=file_prefix)
    print "Finished plotting the heatmaps. Results are saved..."

    # Repeat the process with Gpy
    if testGPy:
        start = time.clock()
        kern = GPy.kern.RBF(input_dim=3, variance=horz, lengthscale=l[0])
        train_t = train_t.reshape((train_t.shape[0], 1))
        m = GPy.models.GPRegression(train, train_t, kern)
        m.Gaussian_noise.variance.constrain_fixed(train_t.std())
        end = time.clock()

        print "Finished training GPy in {}...".format(end - start)

        start = time.clock()
        predictions_optimal = m.predict(test)[0].reshape(
            (test_t.shape[0]))
        end = time.clock()

        print "Finished GPy predictions in {}...".format(end - start)

        plot.plotDistribution(predictions_optimal, test_t, city, buckets,
                              process='GPy' + file_prefix)
        print "Finished GPy Distribution Plots..."
        plot.plotHeatMaps(X_test, predictions_optimal, city, buckets,
                          process='GPy' + file_prefix)
        print "Finished GPy Heatmaps"
