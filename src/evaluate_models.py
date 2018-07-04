'''
Evaluate the predictive models
'''
import os
import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt
from scipy import stats



def main():

    # compare results
    evaluation = evaluate_models()

    # plot comparison of models
    evaluation = pd.pivot_table(evaluation, index=['model', 'holdout'], columns='measure', values='value')
    evaluation.boxplot(by=['model'], rot = 90)
    plt.show()

    # save results to csv
    evaluation_mean = evaluation.groupby(['model']).mean()
    evaluation_mean.to_csv('data/model_performance-thresh0-000001.csv')



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
