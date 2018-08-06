import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# import the csvs and add to df
# df columns: actual, predicted, model_name, holdout
df = pd.DataFrame()
df_2 = pd.DataFrame()
dataset = 'lst'

dir_results = 'results/{}/predictions/'.format(dataset)

# add R
for i in range(1,11):
    df_new = pd.read_csv(dir_results+'r_rf_default_r_{}.csv'.format(i))
    df_new['holdout'] = i
    df_new['R'] = df_new['rf']
    df_new['actual_r'] = df_new['actual']
    df_new = df_new.drop(['rf','actual'], axis=1)
    df = df.append(df_new, ignore_index=True)

# add python
for i in range(0,10):
    df_py = pd.read_csv(dir_results+'py_rf_rParams_py_{}.csv'.format(i))
    df_py['holdout'] = i + 1
    # df_new['model'] = 'rf_py_rparam'
    df_py['actual'] = df_py['y_valid']
    df_py['Python'] = df_py['py_rf_rParams']
    df_py = df_py.drop(['y_valid','py_rf_rParams'], axis=1)
    df_2 = df_2.append(df_py, ignore_index=True)

# concat
df = pd.concat([df, df_2],axis=1)

plt.scatter(df['R'],df['Python'])
plt.xlabel('R')
plt.ylabel('Python')
plt.title('Comparison of predictions: {}'.format(dataset))
plt.savefig('fig/compare_RF_{}.png'.format(dataset), format='png', dpi=200, transparent=True)
# plt.show()
plt.clf()
