import pandas as pd

# Load the CSV file
df = pd.read_csv('gait_features4_reshaped.csv')

# Replace '[0]' with 0 and '[1]' with 1
df = df.replace({'[0]': 0, '[1]': 1})

# Save the modified DataFrame to a new CSV file
df.to_csv('gait_features4_modified.csv', index=False)
