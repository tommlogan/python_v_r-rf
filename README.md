# Comparing R and Python Methods for Predictive Modeling
Tom Logan  
tomlogan.co.nz

I heard recently that statisticians still favor R over Python because they're suspicious of Python's accuracy.
I doubt it.
But there's an easy way to check.
I'm going to answer three questions:
1. For comparable predictive algorithms, which of the two languages runs faster.
2. Are the predictions similar between the languages?
3. Does variable importance and partial dependence change?

## Data
x6, x7, x9, x10 are categorical variables


## Models
1. R with default
2. Python with default
3. Python with R's defaults
6. Python's gradient descent boosted RF
7. XGBoost in Python

What are the differences in computational time between these models?  
In terms of the predictive performance measures, how do these models compare?


## Holdout Cross-validation
Conduct cross-validation for 10 holdouts, using 80% training and 20% validation.
The holdout indices have been saved into a csv `data\holdout_indices.csv` so the holdout is consistent between R and Python.
The results for the predictions will be saved to csv.
The time will be recorded.
The mean absolute error and mean square error will be calculated and saved.

## Random forest
The following hyperparameters will be used:  

| Hyperparameter | Value |
| --- | --- |  
| Number of trees | 500 |
