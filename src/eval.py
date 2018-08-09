import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# import the csvs and add to df
# df columns: actual, predicted, model_name, holdout
df = pd.DataFrame()

dataset = 'lst'

dir_results = 'results/{}/predictions/'.format(dataset)

# add R
for i in range(1,11):
    df_new = pd.read_csv(dir_results+'r_rf_default_r_{}.csv'.format(i))
    df_new['holdout'] = i
    df_new['model'] = 'rf_r_default'
    df_new['predict'] = df_new['rf']
    df = df.append(df_new, ignore_index=True)

# add python
for i in range(0,10):
    df_new = pd.read_csv(dir_results+'py_rf_rParams_py_{}.csv'.format(i))
    df_new['holdout'] = i + 1
    df_new['model'] = 'rf_py_rparam'
    df_new['actual'] = df_new['y_valid']
    df_new['predict'] = df_new['py_rf_rParams']
    df = df.append(df_new, ignore_index=True)

for i in range(0,10):
    df_new = pd.read_csv(dir_results+'r_rf_pyParams_r_{}.csv'.format(i+1))
    df_new['holdout'] = i + 1
    df_new['model'] = 'rf_r_pyparam'
    # df_new['actual'] = df_new['y_valid']
    df_new['predict'] = df_new['rf']
    df = df.append(df_new, ignore_index=True)

for i in range(0,10):
    df_new = pd.read_csv(dir_results+'py_rf_default_py_{}.csv'.format(i))
    df_new['holdout'] = i + 1
    df_new['model'] = 'rf_py_default'
    df_new['actual'] = df_new['y_valid']
    df_new['predict'] = df_new['py_rf_default']
    df = df.append(df_new, ignore_index=True)

# for i in range(0,10):
#     df_new = pd.read_csv(dir_results+'gradient_boost_rf_py_{}.csv'.format(i))
#     df_new['holdout'] = i + 1
#     df_new['model'] = 'gradient_boost_rf'
#     df_new['actual'] = df_new['y_valid']
#     df_new['predict'] = df_new['gradient_boost_rf']
#     df = df.append(df_new, ignore_index=True)

for i in range(0,10):
    df_new = pd.read_csv(dir_results+'gbm_rf_default_py_{}.csv'.format(i))
    df_new['holdout'] = i + 1
    df_new['model'] = 'gbm_rf_default'
    df_new['actual'] = df_new['y_valid']
    df_new['predict'] = df_new['gbm_rf_default']
    df = df.append(df_new, ignore_index=True)


for i in range(0,10):
    df_new = pd.read_csv(dir_results+'xgboost_rf_py_{}.csv'.format(i))
    df_new['holdout'] = i + 1
    df_new['model'] = 'xgboost_rf'
    df_new['actual'] = df_new['y_valid']
    df_new['predict'] = df_new['xgboost_rf']
    df = df.append(df_new, ignore_index=True)

df['absolute_error'] = abs(df['actual'] - df['predict'])
# sns.boxplot(x='model', y='absolute_error', data=df)
# plt.show()
# plt.savefig('fig/ae_{}.pdf'.format(dataset), format='pdf', dpi=1000, transparent=True)
# plt.clf()

# accuracy
df_mean = df[['holdout','model','absolute_error']].groupby(['holdout','model']).mean()
df_mean = df_mean.reset_index(level=['holdout', 'model'])

# plot
sns.boxplot(x='model', y='absolute_error', data=df_mean)
plt.ylabel('Mean Absolute Error')
plt.xlabel('')
locs = plt.xticks()
plt.xticks(locs[0],('GBM','Py RF\n(Py_param)','Py RF\n(R_param)', 'R RF\n(R_param)', 'R RF\n(Py_param)', 'XGBoost'),rotation=0)
plt.title('Predictive accuracy between models: {}'.format(dataset))
if dataset=='lst':
    plt.ylim([0,0.5])
# plt.show()
plt.savefig('fig/mae_{}.png'.format(dataset), format='png', dpi=200, transparent=True)
plt.show()
plt.clf()
