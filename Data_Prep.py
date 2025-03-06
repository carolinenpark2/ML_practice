import pandas as pd
import seaborn as sns
%matplotlib inline   # import necessary packages 

df = pd.read_csv('<file_name>.csv')    # read csv into a data frame 

df.info()   # shows column names, data types, and nulls 

del df['<column name>']   # to delete a column 

df.drop('profit',axis=1, inplace = True)   # drop a column or row, axis = 1: column, axis = 0: row

df.isnull().sum() # to check for missing values 

df.dropna()   # automatically removes columns or rows with missing values

df['<column_name>'].fillna((df['<column_name>'].mean()), inplace=True)   # replaces values in a column with the mean of the values in that column

df.describe()  # shows basic statistics for dataframe 
