'''
Plotting utilities.

Authors:
    Alex Wang (alexwang@college.harvard.edu)
    Luis Perez (luis.perez.live@gmail.com)
Copyright 2015, Harvard University
'''

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# So we know how the data is supposed to be mapped
from .util import columns


def createHeatMap(X):
    '''
    Given a data set, creates a heatmap of it based on x,y coordinates.
    Ignore the temporal feature. You should subset the data before passing
    it into this function if you'd like a heatmap for a specific time period.
    '''
    n = X[:, columns['x']].astype(int).max()
    m = X[:, columns['y']].astype(int).max()
    heatmap = np.zeros((n, m))
    for i in xrange(n):
        for j in xrange(m):
            total = X[(X[:, columns['x']] == i) &
                      (X[:, columns['y']] == j), columns['count']].sum()
            if total > 0:
                heatmap[i, j] = total
    heatmap = heatmap / float(heatmap.sum())
    return heatmap


def plotDistribution(predict, true, city, n, process='GP'):
    '''
    Make some plots for n = 15 for GP process
    '''
    minValue = min(len(predict), 100)
    yPred = predict[-minValue:]
    yTrue = true[-minValue:]

    # Plot Crime for Final Time Period
    plt.clf()
    plt.plot(yPred, label="Predictions")
    plt.plot(yTrue, label="Actual Data")
    plt.title('Crimes using {} in Last Time Period'.format(process))
    plt.xlabel('Final {}} Regions'.format(minValue))
    plt.ylabel('Crime Count')
    plt.legend()
    savefile = os.path.abspath('figures/{}_results/{}_crime_n={}_periods=last.png'.format(
        city, process, n))
    plt.savefig(savefile)
    plt.close()

    print "Crimes for final period saved to {}".format(savefile)

    # Plot crime distribution for final period
    yPred = yPred / float(np.sum(yPred))
    yTrue = yTrue / float(np.sum(yTrue))
    plt.clf()
    plt.plot(yPred, label="Predictions")
    plt.plot(yTrue, label="Actual Data")
    plt.title('Predictive Distribution using {} in Last Time Period'.format(process))
    plt.xlabel('Final {}} Regions'.format(minValue))
    plt.ylabel('Probability')
    plt.legend()
    savefile = os.path.abspath('figures/{}_results/{}_dist_n={}_periods=last.png'.format(
        city, process, n))
    plt.savefig(savefile)
    plt.close()

    print "Distribution saved to {}".format(savefile)

    yPred = predict[:minValue]
    yTrue = true[:minValue]

    # Plot Crime for First Time Period
    plt.clf()
    plt.plot(yPred, label="Predictions")
    plt.plot(yTrue, label="Actual Data")
    plt.title('Crimes using {} in First Time Period'.format(process))
    plt.xlabel('Final {}} Regions'.format(minValue))
    plt.ylabel('Crime Count')
    plt.legend()
    savefile = os.path.abspath('figures/{}_results/{}_crime_n={}_periods=first.png'.format(
        city, process, n))
    plt.savefig(savefile)
    plt.close()

    print "Crimes for first period saved to {}".format(savefile)

    yPred = yPred / float(np.sum(yPred))
    yTrue = yTrue / float(np.sum(yTrue))
    plt.clf()
    plt.plot(yPred, label="Predictions")
    plt.plot(yTrue, label="Actual Data")
    plt.title(
        'Predictive Distribution using {} in First Time Period'.format(process))
    plt.xlabel('Final 100 Regions')
    plt.ylabel('Probability')
    plt.legend()
    savefile = os.path.abspath('figures/{}_results/{}_dist_n={}_period=first.png'.format(
        city, process, n))
    plt.savefig(savefile)
    plt.close()

    print "Distribution saved to {}".format(savefile)


def plotHeatMaps(X_test, predict, city, n, process='GP'):
    '''
    Plots the heatmap based on the X_test data matrix and the predictions.
    Note that X_test must have a final column with the true values of the input
    vectors.
    '''
    # Attach the predictions to the data
    trueValues = np.copy(X_test)
    predictedValues = np.copy(X_test)
    predictedValues[:, columns['count']] = predict.reshape((predict.shape[0]))

    # Now we want to plot the heatmaps for the predictions/actual data
    # by time period
    months = np.unique(X_test[:, columns['t']])
    for month in months:
        # Create the heatmaps
        selected = (X_test[:, columns['t']] == month)
        if selected.sum() > 0:
            plt.clf()
            m1 = createHeatMap(trueValues[selected, :])
            m2 = createHeatMap(predictedValues[selected, :])

            # Make a plot only if both have data available :)
            if m1.sum() > 0 and m2.sum > 0:
                sns.heatmap(m1)
                plt.title('True Density Distribution in Month {}'.format(month))
                savefile = os.path.abspath('figures/{}_results/{}_heatmap_true_n={}_t={}.png'.format(
                    city, process, n, month))
                plt.savefig(savefile)
                plt.close()
                print "True heatmap saved to {}".format(savefile)

                sns.heatmap(m2)
                plt.title(
                    'Predicted Density Distribution in Month {}'.format(month))
                savefile = os.path.abspath('figures/{}_results/{}_heatmap_pred_n={}_t={}.png'.format(
                    city, process, n, month))
                plt.savefig(savefile)
                plt.close()
                print "Predictions heatmap saved to {}".format(savefile)
