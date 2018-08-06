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
import time
import code
# from statsmodels.stats.outliers_influence import variance_inflation_factor


# define constants
DATA_PATH = 'data/data_lst.csv'
MODEL_NAME_PATH = 'data/predictions/model_names.csv'
HOLDOUT_NUM = 10
SEED = 15
CORES_NUM = 10 #min(25,int(os.cpu_count()))
PAR = True
RESPONSE_VAR = 'y'

def main():
    # loop different models

    # import the data
    data = import_data()

    # models to test
    models = [random_forest_default, random_forest_Rparams, random_forest_rsearch, gradient_boost_rf]

    # cross validation
    t = time.time()
    cross_validation(data, models)
    elapsed = time.time() - t
    print("time to run {} holdouts in python: {} min".format(HOLDOUT_NUM, elapsed/60))

    # compare results
    evaluation = evaluate_models()

    # plot comparison of models
    evaluation = pd.pivot_table(evaluation, index=['model', 'holdout'], columns='measure', values='value')
    evaluation.boxplot(by=['model'], rot = 90)
    plt.show()

    # save results to csv
    evaluation_mean = evaluation.groupby(['model']).mean()
    evaluation_mean.to_csv('data/model_performance-thresh0-000001.csv')


def import_data():
    # import the data and separate the test and train
    # import data
    data = pd.read_csv(DATA_PATH)

    # convert categorical variables
    data['x6'] = pd.factorize(data['x6'])[0]
    data['x7'] = pd.factorize(data['x7'])[0]
    data['x9'] = pd.factorize(data['x9'])[0]
    data['x10'] = pd.factorize(data['x10'])[0]

    return(data)


def cross_validation(train, models):
    '''
        divide the data into the training and validating sets
        train and test the models specified in the models list
        input: data frame, list of model functions, random_state variable
        return predictive accuracy
    '''
    # divide the data
    skfolds = ShuffleSplit(n_splits=HOLDOUT_NUM, random_state=SEED, test_size=0.2)
    # create this list of indices with a dummy variable to act as the iterator for saving results
    fold_indices = list(skfolds.split(train, train[RESPONSE_VAR]))
    fold_indices_iterated = [(fold_indices[i][0], fold_indices[i][1], i) for i in range(HOLDOUT_NUM)]
    # parallelize the cross-validation
    if PAR:
        Parallel(n_jobs=CORES_NUM)(delayed(model_cross_validate)(train, models, train_index, test_index, cv_num) for train_index, test_index, cv_num in fold_indices_iterated)
    else:
        for train_index, test_index, cv_num in fold_indices_iterated:
            model_cross_validate(train, models, train_index, test_index, cv_num)


def model_cross_validate(train, models, train_index, test_index, cv_num):
    # divide the data, then train on multiple models
    # split data
    x_train, y_train, x_valid, y_valid = k_fold_data(train, train_index, test_index)

    # init evaluation vectors
    predictions = y_valid.as_matrix()
    predictions = np.expand_dims(predictions, axis=1)
    predictions_names = ['y_valid']
    model_names = []

    # fit models
    for model in models:
        # model name
        model_name = model.__name__
        model_names += [model_name]
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
    results_table.to_csv('data/predictions/py_{}.csv'.format(cv_num))
    # save model names
    if cv_num == 1:
        with open(MODEL_NAME_PATH, 'w') as myfile:
            wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
            wr.writerow(model_names)


def k_fold_data(train, train_index, test_index):
    # return train and validation sets
    x_train = train.iloc[train_index]
    y_train = train[RESPONSE_VAR].iloc[train_index]
    x_valid = train.iloc[test_index]
    y_valid = train[RESPONSE_VAR].iloc[test_index]

    # drop y from x
    x_train = x_train.drop(RESPONSE_VAR, axis=1)
    x_valid = x_valid.drop(RESPONSE_VAR, axis=1)

    return(x_train, y_train, x_valid, y_valid)


def evaluate_models():
    '''
        assigns predictions to classes
        calculates performance metrics
        plots results
    '''
    # model names
    with open(MODEL_NAME_PATH, 'r') as myfile:
        reader = csv.reader(myfile)
        model_names = list(reader)[0]
    # import predictions
    predictions = get_predictions(model_names)
    # evaluate predictions with performance measures
    evaluation = evaluate_predictions(predictions)
    return(evaluation)


def get_predictions(model_names):
    # from the saved results, compile into single dictionary
    # init prediction probability dictionary
    predictions = {model_name: dict() for model_name in model_names}
    predictions['y_valid'] = dict()
    # convert probabilistic predictions to class predictions
    for i in range(HOLDOUT_NUM):
        # import csv
        results = pd.read_csv('data/predictions/py_{}.csv'.format(i))
        # get the actual results
        predictions['y_valid'][i] = results['y_valid'].as_matrix()
        # get the prediction probabilities for each model
        for model_name in model_names:
            predictions[model_name][i] = results[model_name].as_matrix()
    return(predictions)


def evaluate_predictions(predictions):
    # evaluate the predictions
    # init the evaluation dataframe
    eval_cols = ['value', 'model', 'measure', 'holdout']
    evaluation = pd.DataFrame(columns = eval_cols)
    # loop through models in the predictions (note it needs to be in the predictions, so i can add thresholds and update names later)
    model_names = list(predictions.keys()); model_names.remove('y_valid')
    for model_name in model_names:
        # loop through the recorded holdouts
        for i in predictions[model_name].keys():
            # pull the prediction and actual
            y_valid = predictions['y_valid'][i]
            y_pred = predictions[model_name][i]
            # calculate some of the performance measures and add to evaluation dataframe
            measures = list()
            # mean square error
            square_error = (y_valid - y_pred)**2
            measures.append([square_error.mean(), model_name, 'MSE', i])

            # mean absolute error
            absolute_error = np.abs(y_valid - y_pred)
            measures.append([absolute_error.mean(), model_name, 'MAE', i])

            # append to dataframe
            evaluation = evaluation.append(pd.DataFrame(measures, columns = eval_cols), ignore_index=True)
            code.interact(local=locals())
    return(evaluation)


'''
Model functions
'''

def random_forest_default(x_train, y_train, x_valid):
    # Random forest
    # train
    forest_reg = RandomForestRegressor(random_state=SEED)#, n_estimators=500, max_features=1/3)
    forest_reg.fit(x_train, y_train)
    # validate
    y_pred = forest_reg.predict(x_valid)
    # return predictive probabilities
    return(y_pred)


def random_forest_Rparams(x_train, y_train, x_valid):
    # Random forest
    # train
    forest_reg = RandomForestRegressor(random_state=SEED, n_estimators=500, max_features=1/3)
    forest_reg.fit(x_train, y_train)
    # validate
    y_pred = forest_reg.predict(x_valid)
    # return predictive probabilities
    return(y_pred)


def random_forest_rsearch(x_train, y_train, x_valid):
    # Random forest
    # train
    forest_clf = RandomForestClassifier(random_state=SEED)
    param_grid = {'n_estimators': [3, 100, 300, 500], 'max_features': [1/3, 2, 5, 8]}
    forest_grid = RandomizedSearchCV(forest_clf, param_grid, cv=5, scoring='accuracy')
    forest_grid.fit(x_train, y_train)
    # validate
    y_pred = forest_grid.predict_proba(x_valid)
    # return predictive probabilities
    return(y_pred)


def random_forest_grid(x_train, y_train, x_valid):
    # Random forest
    # train
    forest_clf = RandomForestClassifier(random_state=SEED)
    param_grid = {'n_estimators': [3, 250, 500], 'max_features': [1/3, 2, 5, 8]}
    forest_grid = GridSearchCV(forest_clf, param_grid, cv=5, scoring='accuracy')
    forest_grid.fit(x_train, y_train)
    # validate
    y_pred = forest_grid.predict_proba(x_valid)
    # return predictive probabilities
    return(y_pred)


def gradient_boost_rf(x_train, y_train, x_valid):
    # Random forest
    # train
    gbm_reg = GradientBoostingRegressor(max_depth=2, random_state=RANDOM_SEED, learning_rate=0.1, n_estimators=500, loss='ls')
    gbm_reg.fit(x_train, y_train)
    # validate
    y_pred = gbm_reg.predict(x_valid)
    # return predictive probabilities
    return(y_pred)


def xgboost_rf(x_train, y_train, x_valid):
    # Random forest
    from xgboost import XGBRegressor
    # train
    gbm_reg = XGBRegressor(random_state=RANDOM_SEED)
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
