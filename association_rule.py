# import necessary packages and libraries
import pandas as pd
import numpy as np
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth

# read data (csv file) into a pandas dataframe
dataset = pd.read_csv('BigSports Purchase Data.csv', header = None)

# replace missing values with the string "none"
dataset.fillna('none', inplace = True)

# take array from dataframe, transform data to one-hot encoded format 
ds = dataset.values
te = TransactionEncoder()
te_array = te.fit_transform(ds)
