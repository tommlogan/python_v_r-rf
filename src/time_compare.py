import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

dataset = 'zeroinflate'
dir_results = 'results/{}/'.format(dataset)

# import time data
df = pd.read_csv(dir_results+'time_elapsed_lst.csv'.format(dataset))
df['dataset'] = dataset

# df = df.drop(2)

if dataset=='lst':
    df2 = df.iloc[[7,1,0,6,3,5],]
else:
    df2 = df.iloc[[6,1,0,7,3,5],]

# plot the results
if dataset=='lst':
    sns.barplot(df2['Model'],df2['Time'])
    plt.ylabel('Time (sec)')
else:
    sns.barplot(df2['Model'],np.log10(df2['Time']))
    plt.ylabel('Logged Time (log10(sec))')

plt.xlabel('')
locs = plt.xticks()
plt.xticks(locs[0],('R RF\n(R_param)','Py RF\n(R_param)','Py RF\n(Py_param)','R RF\n(Py_param)', 'GBM', 'XGBoost'),rotation=0)
plt.title('Run time for the models: {}'.format(dataset))
# if dataset=='lst':
#     plt.ylim([0,0.5])
# plt.show()
plt.savefig('fig/time_{}.png'.format(dataset), format='png', dpi=200, transparent=True)
plt.show()
plt.clf()
