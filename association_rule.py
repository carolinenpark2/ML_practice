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

# array back to data frame 
df = pd.DataFrame(te_array, columns = te.columns_)

# drop missing values
df = df.drop('none', axis = 1)

# show frequent itemsets that appear together in at least 20% of transactions using apriori
freq_itemsets = apriori(df, min_support = 0.2, use_colnames = True)

# generate association rules from the frequent itemsets
res = association_rules(freq_itemsets, metric = 'confidence', min_threshold = 0.4)

# subset dataframe
res1 = res[['antecedents', 'consequents', 'support', 'confidence', 'lift']]

# sort values based in lift
sorted_res1 = res1.sort_values(by = 'lift')

# show frequent itemsets that appear together in at least 20% of transactions using fpg
freq_itemsets = fpgrowth(df, min_support = 0.2, use_colnames = True)

resfpg = association_rules(freq_itemsets, metric = 'confidence', min_threshold = 0.4)


