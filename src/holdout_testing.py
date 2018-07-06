'''
Fit different models to the dataset
'''
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


# define constants
DATA_PATH = 'data/data_zeroinflate.csv'
TIME_PATH = 'data/time_elapsed_lst.csv'
HOLDOUT_NUM = 10
SEED = 15
CORES_NUM = 10 #min(25,int(os.cpu_count()))
PAR = True
RESPONSE_VAR = 'y'

def main():
    # loop different models

    # import the data
    data = import_data()

    # create the holdout datasets
    create_holdouts(data)

    # models to test
    models = [py_rf_default, py_rf_rParams, rf_randomsearch, gbm_rf_default, gbm_rf_rParams, xgboost_rf ] #, py_rf_Rparams, random_forest_rsearch, gradient_boost_rf]
    for model in models:
        # cross validation
        elapsed = cross_validation(data, model)
        print("time to run {} holdouts in python: {} min".format(HOLDOUT_NUM, elapsed/60))

        # write time elapsed
        file_exists = os.path.isfile(TIME_PATH)
        header = ['Language','Model','Time']
        with open(TIME_PATH, 'a') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(header)
            writer.writerow(['Python', models[0].__name__, elapsed])



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


def create_holdouts(data):
    '''
        creates csvs for the test and train datasets
    '''
    # divide the data
    skfolds = ShuffleSplit(n_splits=HOLDOUT_NUM, random_state=SEED, test_size=0.2)
    # list of indices
    fold_indices = list(skfolds.split(data, data[RESPONSE_VAR]))

    # make a boolean list where it is one if it's in the training set, column number is the holdout Number
    row_nums = pd.DataFrame(dict.fromkeys(range(HOLDOUT_NUM),[False] * len(data)))
    for i in range(HOLDOUT_NUM):
        row_nums.loc[fold_indices[i][0], i] = True

    # save the pd dataframe as a csv so that it can be imported and have models trained on it
    row_nums.to_csv('data/holdout_indices.csv')


def cross_validation(train, model):
    '''
        divide the data into the training and validating sets
        train and test the models specified in the models list
        input: data frame, list of model functions, random_state variable
        return predictive accuracy
    '''
    # import the data divisions
    train_indices = pd.read_csv('data/holdout_indices.csv')
    # create this list of indices with a dummy variable to act as the iterator for saving results
    fold_indices_iterated = []
    for i in range(HOLDOUT_NUM):
        fold_indices_iterated += [(np.where(train_indices[str(i)])[0], np.where(train_indices[str(i)]==False)[0], i)]
    # parallelize the cross-validation
    t = time.time()
    if PAR:
        Parallel(n_jobs=CORES_NUM)(delayed(model_cross_validate)(train, model, train_index, test_index, cv_num) for train_index, test_index, cv_num in fold_indices_iterated)
    else:
        for train_index, test_index, cv_num in fold_indices_iterated:
            model_cross_validate(train, model, train_index, test_index, cv_num)

    return(time.time() - t)


def model_cross_validate(train, model, train_index, test_index, cv_num):
    # divide the data, then train on multiple models
    # split data
    x_train, y_train, x_valid, y_valid = k_fold_data(train, train_index, test_index)

    # init evaluation vectors
    predictions = y_valid.as_matrix()
    predictions = np.expand_dims(predictions, axis=1)
    predictions_names = ['y_valid']

    # fit models
    # model name
    model_name = model.__name__
    # train model and return predictions
    results = model(x_train, y_train, x_valid)
    if len(results.shape) == 1:
        results = np.reshape(results, (len(results),1))
    # append predictive probabilities to np.array
    predictions = np.append(predictions, results, axis=1)
    # append column headings
    predictions_names += [model_name]

    # save result to csv
    results_table = pd.DataFrame(data=predictions, columns=predictions_names)
    results_table.to_csv('data/predictions/{}_py_{}.csv'.format(model_name,cv_num))


def k_fold_data(train, train_index, test_index):
    # return train and validation sets
    x_train = train.iloc[train_index]
    y_train = train[RESPONSE_VAR].iloc[train_index]
    x_valid = train.iloc[test_index]
    y_valid = train[RESPONSE_VAR].iloc[test_index]
    #
    # drop y from x
    x_train = x_train.drop(RESPONSE_VAR, axis=1)
    x_valid = x_valid.drop(RESPONSE_VAR, axis=1)
    #
    return(x_train, y_train, x_valid, y_valid)


'''
Model functions
'''

def py_rf_default(x_train, y_train, x_valid):
    # Random forest
    # train
    forest_reg = RandomForestRegressor(random_state=SEED)
    forest_reg.fit(x_train, y_train)
    # validate
    y_pred = forest_reg.predict(x_valid)
    # return predictive probabilities
    return(y_pred)


def py_rf_rParams(x_train, y_train, x_valid):
    # Random forest
    # train
    forest_reg = RandomForestRegressor(random_state=SEED, n_estimators=500, max_features=1/3)
    forest_reg.fit(x_train, y_train)
    # validate
    y_pred = forest_reg.predict(x_valid)
    # return predictive probabilities
    return(y_pred)


def rf_randomsearch(x_train, y_train, x_valid):
    # Random forest
    # train
    forest_reg = RandomForestRegressor(random_state=SEED)
    param_grid = {'n_estimators': [10, 100, 250, 500, 750, 1000], 'max_features': [0.2, 0.4, 0.6, 0.8, 1]}
    reg = RandomizedSearchCV(forest_reg, param_grid, cv=5, scoring='neg_mean_absolute_error')
    reg.fit(x_train, y_train)
    # validate
    y_pred = reg.predict(x_valid)
    # return predictive probabilities
    return(y_pred)


def rf_pso(x_train, y_train, x_valid):
    # Random forest using particle swarm opt on the hyperparameters
    # pso
    x_train = x_train.as_matrix()
    y_train = y_train.as_matrix()
    @optunity.cross_validated(x=x_train, y=y_train, num_iter=2, num_folds=5)
    def performance(x_train, y_train, x_test, y_test,n_estimators, max_features):
        model = RandomForestRegressor(n_estimators=int(n_estimators),
                                       max_features=max_features)
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        return(optunity.metrics.mse(y_test, predictions))
    # hyperparameter search
    optimal_pars, _, _ = optunity.minimize(performance, num_evals = 100, n_estimators=[10,800], max_features=[0.2,10])
    # train
    forest_reg = RandomForestRegressor(random_state=SEED, **optimal_pars)
    reg.fit(x_train, y_train)
    # validate
    y_pred = reg.predict(x_valid)
    # return predictive probabilities
    return(y_pred)


def rf_gridsearch(x_train, y_train, x_valid):
    # Random forest using grid search
    # train
    forest_reg = RandomForestRegressor(random_state=SEED)
    param_grid = {'n_estimators': [3, 250, 500], 'max_features': [0.2, 0.4, 0.6, 0.8, 1]}
    forest_grid = GridSearchCV(forest_reg, param_grid, cv=5, scoring='mean_absolute_error')
    forest_grid.fit(x_train, y_train)
    # validate
    y_pred = forest_grid.predict(x_valid)
    # return predictive probabilities
    return(y_pred)


def gbm_rf_rParams(x_train, y_train, x_valid):
    # Gradient Boosting Random Forest
    # train
    gbm_reg = GradientBoostingRegressor(max_depth=5, random_state=SEED, learning_rate=0.1, n_estimators=500, loss='ls')
    gbm_reg.fit(x_train, y_train)
    # validate
    y_pred = gbm_reg.predict(x_valid)
    # return predictive probabilities
    return(y_pred)

def gbm_rf_default(x_train, y_train, x_valid):
    # Gradient Boosting Random Forest
    # train
    gbm_reg = GradientBoostingRegressor(random_state=SEED)
    gbm_reg.fit(x_train, y_train)
    # validate
    y_pred = gbm_reg.predict(x_valid)
    # return predictive probabilities
    return(y_pred)


def xgboost_rf(x_train, y_train, x_valid):
    # Random forest

    # train
    gbm_reg = XGBRegressor(random_state=SEED, n_estimators=500)
    gbm_reg.fit(x_train, y_train)
    # validate
    y_pred = gbm_reg.predict(x_valid)
    # return predictive probabilities
    return(y_pred)


def multilayer_perceptron(x_train, y_train, x_valid):
    # multi-layer perceptron classifier
    # train
    mpl_clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=SEED)
    mpl_clf.fit(x_train, y_train)
    # validate
    y_pred = mpl_clf.predict_proba(x_valid)
    # return predictive probabilities
    return(y_pred)


def multilayer_perceptron_rsearch(x_train, y_train, x_valid):
    # random hyperparameter search multilayer perceptron
    param_grid = {
        'hidden_layer_sizes': [(7, 7), (128,), (128, 7)],
        'tol': [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
    }
    # train
    mpl_rsearch = RandomizedSearchCV(MLPClassifier(learning_rate='adaptive', learning_rate_init=1., early_stopping=True, shuffle=True),
        param_distributions=param_grid)
    mpl_rsearch.fit(x_train, y_train)
    # validate
    y_pred = mpl_rsearch.predict_proba(x_valid)
    # return predictive probabilities
    return(y_pred)


'''
Support functions
'''


def calculate_vif(X, thresh=5.0):
    # from https://stats.stackexchange.com/questions/155028/how-to-systematically-remove-collinear-variables-in-python
    # drops variables with vif > thresh
    cols = X.columns
    variables = np.arange(X.shape[1])
    dropped=True
    while dropped:
        dropped=False
        c = X[cols[variables]].values
        vif = [variance_inflation_factor(c, ix) for ix in np.arange(c.shape[1])]
        maxloc = vif.index(max(vif))
        if max(vif) > thresh:
            # print('dropping \'' + X[cols[variables]].columns[maxloc] + '\' at index: ' + str(maxloc))
            variables = np.delete(variables, maxloc)
            dropped=True
    print('Remaining variables:')
    print(X.columns[variables])
    return X[cols[variables]]


if __name__ == '__main__':
    main()
