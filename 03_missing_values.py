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
# Grabing missing values
##############################
def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 1)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end='\n\n')

    if na_name:
        return na_columns


missing_values_table(df)

##############################
# Removing missing values
##############################
df.dropna()
df.dropna().shape
df.shape

##############################
# Filling missing values with reassignment
##############################
df.head()
df.shape
df.columns
df.age.fillna(df.age.mean()).isnull().sum()
df.age.fillna(df.age.median()).isnull().sum()
df.age.fillna(0)

df.apply(lambda x: x.fillna(x.mean()) if x.dtype != 'O' else x, axis=0).head()

df['embarked'].fillna(df.embarked.mode()[0])
df.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == 'O' and len(x.unique()) <= 10) else x, axis=0).isnull().sum()

##############################
# value assignment in categorical variable breakdown
##############################
df.groupby('sex')['age'].mean()

df.age.fillna(df.groupby('sex')['age'].transform('mean')).isnull().sum()

df.loc[((df['age'].isnull()) & (df['sex'] == 'female')), 'age'] = df.groupby('sex')['age'].mean()['female']

df.loc[((df['age'].isnull()) & (df['sex'] == 'male')), 'age'] = df.groupby('sex')['age'].mean()['male']

df.isnull().sum().sort_values(ascending=False)

##############################
# Prediction-based reassignment proces
##############################
df.head()


def grab_col_names(dataframe, cat_th=10, car_th=20):
    # categorical columns
    cat_cols = [col for col in dataframe.columns if str(dataframe[col].dtypes) in ['category', 'object', 'bool']]
    num_but_cat = [col for col in dataframe.columns if
                   dataframe[col].nunique() < cat_th and dataframe[col].dtypes in ['int64', 'float64']]
    cat_but_car = [col for col in dataframe.columns if
                   dataframe[col].nunique() > car_th and str(dataframe[col].dtypes) in ['category', 'object']]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # numerical columns
    num_cols = [col for col in dataframe.columns if
                dataframe[col].dtypes in ['int64', 'float64']]
    num_cols = [col for col in num_cols if col not in cat_cols]

    # Reporting section
    print(f'Observation: {dataframe.shape[0]}')
    print(f'Variables: {dataframe.shape[1]}')
    print(f'Number of categorical variables: {len(cat_cols)}')
    print(f'Number of numerical variables: {len(num_cols)}')
    print(f'Number of categorical but cardinal variables: {len(cat_but_car)}')
    print(f'Number of numeric but categorical variables: {len(num_but_cat)}')

    return cat_cols, num_cols, cat_but_car


cat_cols, num_cols, cat_but_car = grab_col_names(df)
num_cols = [col for col in num_cols if col not in 'passengerid']

dff = pd.get_dummies(df[cat_cols + num_cols], drop_first=True)
dff.head()

scaler = MinMaxScaler()
dff = pd.DataFrame(scaler.fit_transform(dff), columns=dff.columns)
dff.head()

from sklearn.impute import KNNImputer

imputer = KNNImputer(n_neighbors=5)

dff = pd.DataFrame(imputer.fit_transform(dff), columns=dff.columns)
dff.head()

dff = pd.DataFrame(scaler.inverse_transform(dff), columns=dff.columns)
dff.head()

df['age_imputed_knn'] = dff['age']
df.loc[df['age'].isnull(), ['age', 'age_imputed_knn']]
df.loc[df['age'].isnull()]

##############################
# Advanced analysis
##############################
df = load()
msno.bar(df)
plt.show(block=True)

msno.matrix(df)
plt.show(block=True)

msno.heatmap(df)
plt.show(block=True)


##############################
# Analysis missing values with dependent variables
##############################
def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 1)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end='\n\n')

    if na_name:
        return na_columns


na_cols = missing_values_table(df, True)


def missing_vs_target(dataframe, target, na_columns):
    temp_df = dataframe.copy()
    for col in na_columns:
        temp_df[col + '_na_flag'] = np.where(temp_df[col].isnull(), 1, 0)
    na_flags = temp_df.loc[:, temp_df.columns.str.contains('_na_')].columns

    for col in na_flags:
        print(pd.DataFrame({
            'Target_Mean': temp_df.groupby(col)[target].mean(),
            'Count': temp_df.groupby(col)[target].count()
        }), end='\n\n')


missing_vs_target(df, 'survived', na_cols)
