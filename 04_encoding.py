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
# Label encoding
##############################
le = LabelEncoder().fit_transform(df.sex)[: 5]


# Functionalization
def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe


binary_cols = [col for col in df.columns if df[col].dtypes not in ['int64', 'float64'] and df[col].nunique() == 2]
for col in binary_cols:
    label_encoder(df, col)
df.sex.head()


##############################
# OneHot and Label encoding
##############################
def one_hot_encoder(dataframe, categorical_cols, dropfirst=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=dropfirst)
    return dataframe


df = load()
ohe_cols = [col for col in df.columns if 10 >= df[col].nunique() > 2]
one_hot_encoder(df, ohe_cols).head()


##############################
# Rare encoding
##############################
# Import the dataset for the large-scale applications
def load_application():
    data = pd.read_csv('01_miuul_machine_learning_summercamp/00_datasets/application_train.csv')
    data.columns = [col.lower() for col in data.columns]
    return data


dff = load_application()
dff.head()

dff.name_education_type.value_counts()


##############################
# Grabing the variables and generalizing the processe
##############################
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


cat_cols, num_cols, cat_but_car = grab_col_names(dff)


##############################
# cat_summmary function for the categorical variables
##############################
def cat_summary(dataframe, col_name, plot=False):
    print(f'\n########-{col_name.upper()}-##########')
    if dataframe[col_name].dtypes == 'bool':
        dataframe[col_name] = dataframe[col_name].astype(int)
    print(pd.DataFrame({
        col_name: dataframe[col_name].value_counts(),
        'Ratio': 100 * dataframe[col_name].value_counts() / len(dataframe)
    }))

    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)


for col in cat_cols:
    print(col, cat_summary(dff, col, plot=False))


##############################
# Rare analyser
##############################
def rare_analyzer(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col, ':', len(dataframe[col].value_counts()))
        print(pd.DataFrame({
            'Count': dataframe[col].value_counts(),
            'Ratio': dataframe[col].value_counts()/len(dataframe),
            'Target_Mean': dataframe.groupby(col)[target].mean()
        }), end='\n\n')


rare_analyzer(dff, 'target', cat_cols)


##############################
# Rare encoder
##############################
def rare_encoder(dataframe, rare_percent):
    temp_df = dataframe.copy()
    rare_columns = [col for col in temp_df.columns if temp_df[col].dtypes == 'O' and
                    (temp_df[col].value_counts()/len(temp_df) < rare_percent).any(axis=None)]

    for var in rare_columns:
        tmp = temp_df[var].value_counts()/len(temp_df)
        rare_labels = tmp[tmp < rare_percent].index
        temp_df[var] = np.where(temp_df[var].isin(rare_labels), 'Rare', temp_df[var])

    return temp_df


new_df = rare_encoder(dff, 0.01)
rare_analyzer(new_df, 'target', cat_cols)