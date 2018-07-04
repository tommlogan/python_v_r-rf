import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# import the csvs and add to df
# df columns: actual, predicted, model_name, holdout
df = pd.DataFrame()

# add R
for i in range(1,11):
    df_new = pd.read_csv('data/predictions/r_rf_default_r_{}.csv'.format(i))
    df_new['holdout'] = i
    df_new['model'] = 'rf_r_default'
    df_new['predict'] = df_new['rf']
    df = df.append(df_new, ignore_index=True)

# add python
for i in range(0,10):
    df_new = pd.read_csv('data/predictions/py_rf_Rparams_py_{}.csv'.format(i))
    df_new['holdout'] = i + 1
    df_new['model'] = 'rf_py_rparam'
    df_new['actual'] = df_new['y_valid']
    df_new['predict'] = df_new['py_rf_Rparams']
    df = df.append(df_new, ignore_index=True)

for i in range(0,10):
    df_new = pd.read_csv('data/predictions/py_rf_default_py_{}.csv'.format(i))
    df_new['holdout'] = i + 1
    df_new['model'] = 'rf_py_default'
    df_new['actual'] = df_new['y_valid']
    df_new['predict'] = df_new['py_rf_default']
    df = df.append(df_new, ignore_index=True)

for i in range(0,10):
    df_new = pd.read_csv('data/predictions/gradient_boost_rf_py_{}.csv'.format(i))
    df_new['holdout'] = i + 1
    df_new['model'] = 'gradient_boost_rf'
    df_new['actual'] = df_new['y_valid']
    df_new['predict'] = df_new['gradient_boost_rf']
    df = df.append(df_new, ignore_index=True)

for i in range(0,10):
    df_new = pd.read_csv('data/predictions/gbm_rf_default_py_{}.csv'.format(i))
    df_new['holdout'] = i + 1
    df_new['model'] = 'gbm_rf_default'
    df_new['actual'] = df_new['y_valid']
    df_new['predict'] = df_new['gbm_rf_default']
    df = df.append(df_new, ignore_index=True)


# for i in range(0,10):
#     df_new = pd.read_csv('data/predictions/xgboost_rf_py_{}.csv'.format(i))
#     df_new['holdout'] = i + 1
#     df_new['model'] = 'xgboost_rf'
#     df_new['actual'] = df_new['y_valid']
#     df_new['predict'] = df_new['xgboost_rf']
#     df = df.append(df_new, ignore_index=True)

df['absolute_error'] = abs(df['actual'] - df['predict'])
sns.boxplot(x='model', y='absolute_error', data=df)
plt.show()

df_mean = df[['holdout','model','absolute_error']].groupby(['holdout','model']).mean()
df_mean = df_mean.reset_index(level=['holdout', 'model'])
sns.boxplot(x='model', y='absolute_error', data=df_mean)
plt.show()
