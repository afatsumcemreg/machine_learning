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


# Import the dataset for the small-scale applications
def load():
    data = pd.read_csv('01_miuul_machine_learning_summercamp/00_datasets/titanic.csv')
    data.columns = [col.lower() for col in data.columns]
    return data


df = load()
df.head()

##############################
# Standard scaler
##############################
ss = StandardScaler()
df['age_standard_scaler'] = ss.fit_transform(df[['age']])
df.head()


##############################
# Robust scaler
##############################
rs = RobustScaler()
df['age_robust_scaler'] = rs.fit_transform(df[['age']])
df.head()


##############################
# MinMax scaler
##############################
mms = MinMaxScaler()
df['age_min_max_scaler'] = mms.fit_transform(df[['age']])
df.head()
df.describe().T


##############################
# Getting num_summary function
##############################
def num_summary(dataframe, col_name, plot=False):
    """
    for col in num_cols:
        print(f'\n***************-{col.upper()}-***************')
        num_summary(df, col, plot=False)
    """
    quantiles = [0.05, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    print(dataframe[col_name].describe(quantiles).T)

    if plot:
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        sns.boxplot(y=dataframe[col_name], data=dataframe)
        plt.subplot(1, 2, 2)
        sns.histplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)


age_cols = [col for col in df.columns if 'age' in col]
for col in age_cols:
    print(f'\n***************-{col.upper()}-***************')
    num_summary(df, col, plot=True)


##############################
# Converting numerical variables to categorical variables
##############################
df['age_qcut'] = pd.qcut(df.age, 5)
df.head()

df['age_cut'] = pd.cut(df.age, bins=[0, 18, 25, 45, 60, 100], labels=['0_18', '19_25', '26_45', '46_60', '61_100'])
df.head()