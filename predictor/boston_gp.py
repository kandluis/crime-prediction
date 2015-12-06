'''
Script to process the boston data. Currently, parameters must be changed
directly at the script level!

To run the script, use the following command from the parent directory (ie.
    make sure you're in the crime-predictions directory)
    python -m predictor.boston_gp

Authors:
    Alex Wang (alexwang@college.harvard.edu)
    Luis Perez (luis.perez.live@gmail.com)
Copyright 2015, Harvard University
'''

import numpy as np
import pandas as pd
import re
import warnings
import os

from . import GP

''' Global Variables '''

bos_file = os.path.abspath('data/boston.csv')  # Location of data
buckets = 15  # Number of buckets.

# Square Exponential Kernel Parameters
# These are the optimal parameters for n = 10 (we can try with other
# values too)
# BAYESIAN
# l = [9.164520,  0.296120, 10.153288]
# horz = 33.522111
# GPy
l = [0.82754075018, 0.82754075018, 0.82754075018]
horz = 9620.11949755

# This is a function that takes as input the training data results.
# BAYESIAN
# sig_eps_f = lambda train_t: 105.693084
# GPy
sig_eps_f = lambda train_t: train_t.std()

logTransform = False  # Should we do GP under the logspace?

# Prefix to use for plots created in bos directory.
file_prefix = 'GPSEOptimizedGPyTrain'


def read_data(bos_file):
    ''' Read data '''
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
                columns = match.group(1).split(',')  # Get columns as a list
                columns = [int(c) for c in columns]
                print("Applying %s dtype to columns:" % target_type, columns)
                bos_data.iloc[:, columns] = bos_data.iloc[
                    :, columns].astype(target_type)

    ''' Featurize data '''
    # temporal features
    # day of week
    day = np.array(bos_data.DAY_WEEK)
    day[day == "Sunday"] = 0
    day[day == "Monday"] = 1
    day[day == "Tuesday"] = 2
    day[day == "Wednesday"] = 3
    day[day == "Thursday"] = 4
    day[day == "Friday"] = 5
    day[day == "Saturday"] = 6

    # Split mm/dd/yyyy xx:yy:zz AM/PM into components
    date_time = np.array([x.split() for x in bos_data.FROMDATE])
    date = date_time[:, 0]
    time = date_time[:, 1]
    tod = date_time[:, 2]

    # month, day, year
    date = np.array([x.split('/') for x in date])
    month = [int(x) for x in date[:, 0]]
    dom = [int(x) for x in date[:, 1]]
    year = [int(x) for x in date[:, 2]]
    # months since Jan 2012
    time_feat = np.subtract(year, 2012) * 12 + month

    # time of day
    time_c = [x.split(':') for x in time]
    time = [int(x[1]) if (y == 'AM' and int(x[0]) == 12) else 60 * int(x[0]) + int(x[1])
            if (y == 'AM' and int(x[0]) != 12) or (int(x[0]) == 12 and y == 'PM') else 12 * 60 + 60 * int(x[0]) + int(x[1])
            for x, y in zip(time_c, tod)]

    # grab the features we want
    data_unnorm = np.transpose(
        np.vstack((time_feat, bos_data.X, bos_data.Y))).astype(float)
    # remove NaNs
    good_data = data_unnorm[~(np.isnan(data_unnorm[:, 1]))]

    return good_data

print "Finished processing Boston data..."

GP.run_gp(read_data(bos_file), buckets, l, horz, sig_eps_f, logTransform,
          file_prefix, 'boston')
