# Import libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import missingno as msno
from _datetime import date
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)


# Import the dataset for the large-scale applications
def load_application_train():
    data = pd.read_csv('01_miuul_machine_learning_summercamp/00_datasets/application_train.csv')
    data.columns = [col.lower() for col in data.columns]
    return data


dff = load_application_train()
dff.head()


# Import the dataset for the small-scale applications
def load():
    data = pd.read_csv('01_miuul_machine_learning_summercamp/00_datasets/titanic.csv')
    data.columns = [col.lower() for col in data.columns]
    return data


df = load()
df.head()


##############################
# Defining a function to get outliers
##############################
def outlier_threshholds(dataframe, col_name, q1=0.05, q3=0.95):
    quantile1 = dataframe[col_name].quantile(q1)
    quantile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quantile3 - quantile1
    up_limit = quantile3 + 1.5 * interquantile_range
    low_limit = quantile1 - 1.5 * interquantile_range

    return low_limit, up_limit


num_cols = [col for col in df.columns if df[col].dtypes in ['int64', 'float64'] and df[col].nunique() > 10]
num_cols = [col for col in num_cols if col not in 'passengerid']
for col in num_cols:
    low_limit, up_limit = outlier_threshholds(df, col)
    print(low_limit, up_limit)


# Checking outliers
def check_outliers(dataframe, col_name):
    low_limit, up_limit = outlier_threshholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] < low_limit) | (dataframe[col_name] > up_limit)].any(axis=None):
        return True
    else:
        return False


for col in num_cols:
    print(col, check_outliers(df, col))


# Grabing outliers
def grab_outliers(dataframe, col_name, index=False):
    low_limit, up_limit = outlier_threshholds(dataframe, col_name)
    if dataframe[((dataframe[col_name] < low_limit) | (dataframe[col] > up_limit))].shape[0] > 10:
        print(dataframe[((dataframe[col_name] < low_limit) | (dataframe[col_name] > up_limit))].head())
    else:
        print(dataframe[((dataframe[col_name] < low_limit) | (dataframe[col_name] > up_limit))])

    if index:
        outlier_index = dataframe[((dataframe[col_name] < low_limit) | (dataframe[col_name] > up_limit))].index
        return outlier_index


for col in num_cols:
    grab_outliers(df, col)


# Removing outliers
def remove_outliers(dataframe, col_name):
    low_limit, up_limit = outlier_threshholds(dataframe, col_name)
    df_without_outliers = dataframe[~((dataframe[col_name] < low_limit) | (dataframe[col_name] > up_limit))]

    return df_without_outliers


for col in num_cols:
    new_df = remove_outliers(df, col)

df.shape[0] - new_df.shape[0]


# Reassignment the outliers with thresholds
def replace_with_thresholds(dataframe, col_name):
    low_limit, up_limit = outlier_threshholds(dataframe, col_name)
    dataframe.loc[(dataframe[col_name] < low_limit), col_name] = low_limit
    dataframe.loc[(dataframe[col_name] > up_limit), col_name] = up_limit


for col in num_cols:
    print(col, check_outliers(df, col))

for col in num_cols:
    replace_with_thresholds(df, col)


# Multivariate outlier analysis (Local Outlier Factor (LOF Method))
df_diamond = sns.load_dataset('diamonds')
df_diamond = df_diamond.select_dtypes(include=['float64', 'int64']) # to select only numerical variables
df_diamond = df_diamond.dropna()
df_diamond.head()

for col in df_diamond.columns:
    print(col, check_outliers(df_diamond, col))

# Selecting only one variable from outliers
low_limit, up_limit = outlier_threshholds(df_diamond, 'carat')

# how many variables are there in the carat variable
df_diamond[((df_diamond['carat'] < low_limit) | (df_diamond['carat'] > up_limit))].shape

# lof method
clf = LocalOutlierFactor(n_neighbors=20)
clf.fit_predict(df_diamond)
df_diamond_scores = clf.negative_outlier_factor_
df_diamond_scores[:5]
np.sort(df_diamond_scores)

# using elbow method
scores = pd.DataFrame(np.sort(df_diamond_scores))
scores.plot(stacked=True, xlim=[0, 50], style='.-')
plt.show()

# deterining threshold
th = np.sort(df_diamond_scores)[3]
df_diamond[df_diamond_scores < th]
df_diamond.describe([0.01, 0.05, 0.75, 0.90, 0.99]).T
df_diamond[df_diamond_scores < th].index
df_diamond[df_diamond_scores < th].drop(axis=0, labels=df_diamond[df_diamond_scores < th].index)