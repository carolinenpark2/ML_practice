import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

df = pd.read_csv('titanic_mod.csv') # read csv into a pandas dataframe


# visualize any missing data
sns.heatmap(df.isnull(), yticklabels=False, cbar = False, cmap=('viridis'))

# countplot of the dependent value we will look at and try to predict
sns.set_style('whitegrid')
sns.countplot(x='Survived', data=df, palette='RdBu_r')

# explore sex differences in survival 
sns.set_style('whitegrid')
sns.countplot(x='Survived', hue='Sex', data=df, palette='RdBu_r')

# visualize economic differences in survival 
sns.set_style('whitegrid')
sns.countplot(x='Survived', hue = 'Pclass', data=df, palette='rainbow')

# plot the distribution of age 
sns.displot(df['Age'].dropna(), kde = True, color='darkred', bins=30)

# histogram of ticket prices
sns.histplot(data=df, x='Fare', bins=40)

# fill in missing ages with the mean age for the dataset, visualize missing values
df['Age'].fillna(df['Age'].mean(),inplace=True)
sns.heatmap(df.isnull(), yticklabels=False,cbar=False,cmap='viridis')

# drop the column labeled cabin, too many missing values
df.drop('Cabin',axis = 1, inplace=True)
