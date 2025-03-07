# import necessary packages 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

df = pd.read_csv('<data_name.csv>')

# visualize missing data
sns.heatmap(df.isnull(), yticklabels=False, cbar = False, cmap=('viridis'))

# Plotting binary value of a column of interest
sns.set_style('whitegrid')
sns.countplot(x='<column of interest>', data=df, palette='RdBu_r')

# Plotting the distribution of a column/feature with continuous values
sns.displot(df['<column of interest>'].dropna(), color='darkred', bins=30)

# Add a line of best fit to the distribution plot 
sns.displot(df['<column of interest>'].dropna(), kde = True, color='darkred', bins=30)

# Histogram displaying values in a column of interest 
sns.histplot(data=df, x='<column_name>', bins=40)
