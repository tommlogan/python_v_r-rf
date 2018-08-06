import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

dataset = 'lst'
dir_results = 'results/{}/'.format(dataset)

# import time data
df = pd.read_csv(dir_results+'time_elapsed_{}.csv'.format(i))

# plot the results
