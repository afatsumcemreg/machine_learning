# Import libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import datetime
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
# Binary features
##############################
df['new_cabin_bool'] = df['cabin'].notnull().astype(int)
df.head()
df.groupby('new_cabin_bool').agg({'survived': 'mean'})

# comparison test between two groups (new_cabin_bool and survived)
from statsmodels.stats.proportion import proportions_ztest

test_stat, pvalue = proportions_ztest(
    count=[df.loc[df['new_cabin_bool'] == 1, 'survived'].sum(), df.loc[df['new_cabin_bool'] == 0, 'survived'].sum()],
    nobs=[df.loc[df['new_cabin_bool'] == 1, 'survived'].shape[0],
          df.loc[df['new_cabin_bool'] == 0, 'survived'].shape[0]])

print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

# new variables derivation
df.loc[((df['sibsp'] + df['parch']) > 0), 'new_is_alone'] = 'no'
df.loc[((df['sibsp'] + df['parch']) == 0), 'new_is_alone'] = 'yes'
df.head()

df.groupby('new_is_alone').agg({'survived': 'mean'})

# comparison test between two groups (new_is_alone and survived)
from statsmodels.stats.proportion import proportions_ztest

test_stat, pvalue = proportions_ztest(
    count=[df.loc[df['new_is_alone'] == 'no', 'survived'].sum(), df.loc[df['new_is_alone'] == 'yes', 'survived'].sum()],
    nobs=[df.loc[df['new_is_alone'] == 'no', 'survived'].shape[0],
          df.loc[df['new_is_alone'] == 'yes', 'survived'].shape[0]])

print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

##############################
# Text features
##############################

# feature derivation from the variable 'name'
## counting the letters
df['new_name_count'] = df['name'].str.len()
df.head()

## counting the words
df['new_name_word_count'] = df['name'].apply(lambda x: len(str(x).split(" ")))
df.head()

## grabing special structures
df['new_name_dr'] = df['name'].apply(lambda x: len([x for x in x.split() if x.startswith('Dr.')]))
df.head()

df.groupby('new_name_dr').agg({'survived': ['mean', 'count']})

##############################
# Regex features
##############################

# variable derivation proces with regular expressions
df['new_title'] = df['name'].str.extract(' ([A-Za-z]+)\.', expand=False)
df.head()

# apply the groupby method
df[['new_title', 'survived', 'age']].groupby(['new_title']).agg({
    'survived': 'mean',
    'age': ['count', 'mean']})


##############################
# Date features
##############################

dff = pd.read_csv('01_miuul_machine_learning_summercamp/00_datasets/course_reviews.csv')
dff.columns = [col.lower() for col in dff.columns]
dff.head()

# Derivationg new variables from the variable 'timestamp'
dff['timestamp'] = pd.to_datetime(dff['timestamp'], format='%Y-%m-%d')
dff.head()
dff.info()

# derivationg a year variable
dff['year'] = dff['timestamp'].dt.year
dff.head()

# derivationg a mont variable
dff['month'] = dff['timestamp'].dt.month
dff.head()

# derivationg a day_name variable
dff['day_name'] = dff['timestamp'].dt.day_name()
dff.head()

# getting the year difference
dff['year_difference'] = date.today().year - dff['timestamp'].dt.year
dff.tail()

# expression of the difference between two dates in months
dff['month_difference'] = (date.today().year - dff['timestamp'].dt.year) * 12 + (date.today().month - dff['timestamp'].dt.month)
dff.tail()
dff.head()


##############################
# Feature interactions
##############################

# Multiplication of the variables age and pclass
df['new_age_pclass'] = df['age'] * df['pclass']
df.head()

# Aggregation of sibsp and parch
df['new_sibsp_parch'] = df['sibsp'] + df['parch']
df.head()

# Interaction of age and sex
df.loc[(df['sex'] == 'male') & (df['age'] <= 21), 'new_sex_cat'] = 'young_male'
df.loc[(df['sex'] == 'female') & (df['age'] <= 21), 'new_sex_cat'] = 'young_female'
df.loc[(df['sex'] == 'male') & ((df['age'] > 21) & (df['age'] <= 50)), 'new_sex_cat'] = 'matur_male'
df.loc[(df['sex'] == 'female') & ((df['age'] > 21) & (df['age'] <= 50)), 'new_sex_cat'] = 'matur_female'
df.loc[(df['sex'] == 'male') & (df['age'] > 50), 'new_sex_cat'] = 'senior_male'
df.loc[(df['sex'] == 'female') & (df['age'] > 50), 'new_sex_cat'] = 'senior_female'
df.head()

df.groupby('new_sex_cat').agg({'survived': 'mean'})
