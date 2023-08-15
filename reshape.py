import pandas as pd
import numpy as np

# load data
df = pd.read_csv('gait_features4.csv')

# drop the label and save it in a separate variable
Y = df['label']
df.drop('label', axis=1, inplace=True)

# calculate the number of padded samples needed
num_samples = df.shape[0]
padded_samples = (num_samples // (100*100)) * (100*100) + (100*100) - num_samples

# pad the data with [0]
df = np.pad(df.values, [(0, padded_samples), (0, 0)], mode='constant', constant_values=[0])

# reshape the data to (-1, 100, 100, 1)
X = df.reshape(-1, 100, 100, 1)

# add the label back to the data
Y = pd.concat([Y, pd.Series(np.zeros(padded_samples))])
Y = np.resize(Y, (22528, 1, 1, 1))
# save the reshaped and padded data to a new csv file
data = np.concatenate((X.reshape(len(X), -1), Y.reshape(-1,1)), axis=1)
columns = ['feature'+str(i) for i in range(X.shape[1]*X.shape[2]*X.shape[3])]
columns.append('label')
df = pd.DataFrame(data=data, columns=columns)
df.to_csv('gait_features4_reshaped.csv', index=False)
