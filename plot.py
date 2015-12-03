'''
Plotting utilities.

Alex Wang (alexwang@college.harvard.edu)
Luis Perez (luis.perez.live@gmail.com)
'''

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


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
        city, process, n, 12))
    plt.close()


def plotHeatMaps(X_test, predict, city, n, process='GP'):
    '''
    Plots the heatmap based on the X_test data matrix and the predictions.
    Note that X_test must have a final column with the true values of the input
    vectors.
    '''
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
                plt.title(
                    'Predicted Density Distribution in Month {}'.format(month))
                plt.savefig('../figures/{}_results/{}_heatmap_pred_n={}_t={}.png'.format(
                    city, process, n, month))
                plt.close()
