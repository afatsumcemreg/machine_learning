# Import libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

# Import the dataset
df = sns.load_dataset('titanic')
df.columns = [col.lower() for col in df.columns]
df.head()


##############################
# Check the dataframe
##############################
def check_df(dataframe, head=5):
    print('**************-SHAPE-**************')
    print(dataframe.shape)
    print('\n**************-DTYPES-**************')
    print(dataframe.dtypes)
    print('\n**************-HEAD-**************')
    print(dataframe.head(head))
    print('\n**************-TAIL-**************')
    print(dataframe.tail(head))
    print('\n**************-INFO-**************')
    print(dataframe.info())
    print('\n**************-COLUMNS-**************')
    print(dataframe.columns)
    print('\n**************-INDEX-**************')
    print(dataframe.index)
    print('\n**************-IS THERE ANY NULL VALUE?-**************')
    print(dataframe.isnull().values.any())
    print('\n**************-NAN NUMBERS-**************')
    print(dataframe.isnull().sum())
    print('\n**************-DESCRIPTIVE STATISTICS-**************')
    print(dataframe.describe([0.05, 0.25, 0.50, 0.75, 0.95, 0.99]).T)


check_df(df)


##############################
# Grabing the variables and generalizing the processes
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


cat_cols, num_cols, cat_but_car = grab_col_names(df)


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
    cat_summary(df, col, plot=False)


##############################
# num_summmary function for the numerical variables
##############################
def num_summary(dataframe, col_name, plot=False):
    quantiles = [0.05, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    print(dataframe[col_name].describe(quantiles).T)

    if plot:
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        sns.boxplot(y=dataframe[col_name], data=dataframe)
        plt.subplot(1, 2, 2)
        sns.histplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)


for col in num_cols:
    print(f'\n***************-{col.upper()}-***************')
    num_summary(df, col, plot=False)


##############################
# Dependent variable analysis with categorical variables
##############################
def target_summary_with_cat(dataframe, target, categorical_cols):
    print(f'***************-{categorical_cols.upper()}-***************')
    print(pd.DataFrame({
        'Target_Mean': dataframe.groupby(categorical_cols)[target].mean()
    }), end='\n\n')


for col in cat_cols:
    target_summary_with_cat(df, 'survived', col)


##############################
# Dependent variable analysis with numerical variables
##############################
def target_summary_with_num(dataframe, target, numerical_cols):
    print(f'\n****************-{numerical_cols.upper()}-****************')
    print(pd.DataFrame({
        'Target_Mean': dataframe.groupby(target)[numerical_cols].mean()
    }))


for col in num_cols:
    target_summary_with_num(df, 'survived', col)

##############################
# Correlation analaysis
##############################
dff = sns.load_dataset('diamonds')
dff.columns = [col.lower() for col in dff.columns]


# Functionalization of the above processes
def high_correlated_cols(dataframe, plot=False, corr_th=0.90):
    corr = dataframe.corr()
    cor_matrix = dff.corr().abs()
    upper_triangle_matrix = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(np.bool))
    drop_list = [col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col] > corr_th)]

    if plot:
        sns.set(rc={'figure.figsize': (12, 8)})
        sns.heatmap(corr, annot=True, cmap='RdBu')
        plt.show()

    return drop_list


drop_list = high_correlated_cols(dff, plot=True)
high_correlated_cols(dff.drop(drop_list, axis=1), plot=True)
