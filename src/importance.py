from sklearn.ensemble.partial_dependence import partial_dependence
from sklearn.ensemble.partial_dependence import plot_partial_dependence
import os
import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt
from scipy import stats
from joblib import Parallel, delayed
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, ShuffleSplit
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from xgboost import XGBRegressor
import time
import code

dataset = 'concrete'

# define constants
DATA_PATH = 'data/data_{}.csv'.format(dataset)
TIME_PATH = 'data/time_elapsed_{}.csv'.format(dataset)
SEED = 15
RESPONSE_VAR = 'y'

def main():
    # loop different models
    # import the data
    data = import_data()
    # fit model
    forest_reg, x_train = py_rf_rParams(data)
    # get the variable importance
    indices,vars = plot_importances(forest_reg,data)
    # get the partial dependence
    plot_dependence(data)

def import_data():
    # import the data and separate the test and train
    # import data
    data = pd.read_csv(DATA_PATH)
    data = data.dropna()
    # drop columns
    # data = data.drop(['x6','x7','x9','x10'], axis=1)
    if DATA_PATH == 'data/data_zeroinflate.csv':
        # convert categorical variables
        var_cat = ['x48', 'x49','x50','x51','x52','x53','x54']
        for v in var_cat:
            data[v] = pd.factorize(data[v])[0]
    else:
        data = data.apply(pd.to_numeric)
        data = data.dropna()
    return(data)


def py_rf_rParams(data):
    # Random forest
    # data split
    x_train = data.copy()
    y_train = data[RESPONSE_VAR].copy()
    x_train = x_train.drop(RESPONSE_VAR, axis=1)
    # train
    forest_reg = RandomForestRegressor(random_state=SEED, n_estimators=500, max_features=1/3)
    forest_reg.fit(x_train, y_train)
    return(forest_reg,x_train)


def plot_importances(forest_reg,data):
    importances = forest_reg.feature_importances_
    # num_to_show = np.min(10,len(data.columns))
    indices = np.argsort(importances)[::-1]
    # import code
    # code.interact(local=locals())
    vars = data.columns[indices]
    plt.bar(range(len(indices)), importances[indices])
    plt.title('Python Feature Importance: {}'.format(dataset))
    plt.xticks(range(len(indices)), vars, rotation=90)
    plt.xlabel('')
    plt.xlim([-1,len(indices)])
    plt.tight_layout()
    plt.savefig('fig/varimp_py_{}.png'.format(dataset), format='png', dpi=200, transparent=False)
    # plt.show()
    plt.clf()
    return(indices,vars)

def plot_dependence(data):
    ''' Plot the partial dependence '''
    # train a gbm
    x_train = data.copy()
    y_train = data[RESPONSE_VAR].copy()
    x_train = x_train.drop(RESPONSE_VAR, axis=1)
    # train
    reg = GradientBoostingRegressor(random_state=SEED, n_estimators=500, max_features=1/3)
    reg.fit(x_train, y_train)

    # determine importances
    importances = reg.feature_importances_
    indices = np.argsort(importances)[::-1]
    var_names = x_train.columns[indices]

    # partial dependence
    features = list(indices[0:4])
    names = list(var_names[0:4])
    # import code
    # code.interact(local=locals())
    fig, axs = plot_partial_dependence(reg, x_train, features,
                                       feature_names=x_train.columns,
                                       n_jobs=3, grid_resolution=50,
                                       n_cols=2)
    plt.tight_layout()  # tight_layout causes overlap with suptitle
    plt.savefig('fig/pdp_py_{}.png'.format(dataset), format='png', dpi=200, transparent=False)
    plt.show()

if __name__ == '__main__':
    main()
