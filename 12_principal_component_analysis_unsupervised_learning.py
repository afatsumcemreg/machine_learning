# Importing libraries
import numpy as np
import pandas as pd
import seaborn as sns
import random
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

######################################
# PRINCIPLE COMPONENT ANALYSIS (PCA)
######################################

# Importing dataset
df = pd.read_csv("01_miuul_machine_learning_summercamp/00_datasets/hitters.csv")
df.head()

# Checking the dataset
df.isnull().sum()
df.info()
df.describe().T

# Removing categorical variables and dependent variable from the dataset
num_cols = [col for col in df.columns if df[col].dtypes != 'O' and 'Salary' not in col]

# Creating a new dataset from nuerical columns
df[num_cols].head()
df = df[num_cols]

# Removing the missing values from the dataset
df.dropna(inplace=True)
df.shape

# Standardization
df = StandardScaler().fit_transform(df)

# Getting PCA model object
pca = PCA()
pca_fit = pca.fit_transform(df)

# Variance ratio for successful
pca.explained_variance_ratio_

# Calculation cumulative variance
np.cumsum(pca.explained_variance_ratio_)

# Determining optimum component number
pca = PCA().fit(df)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Component number')
plt.ylabel('Cumulative varioance ratio')
plt.show(block=True)

# Creation final PCA model
pca = PCA(n_components=3)
pca_fit = pca.fit_transform(df)
pca.explained_variance_ratio_
np.cumsum(pca.explained_variance_ratio_)

######################################
# PRINCIPLE COMPONENT REGRESSION MODEL (PCR)
######################################

# Reading dataset
df = pd.read_csv('01_miuul_machine_learning_summercamp/00_datasets/hitters.csv')
df.shape

# Selecting numerical columns
num_cols = [col for col in df.columns if df[col].dtypes != 'O' and 'Salary' not in col]

# Other columns except numerical columns
others = [col for col in df.columns if col not in num_cols]

# Converting pca_fit object to dataframe and naming of components
pd.DataFrame(pca_fit, columns=['PC1', 'PC2', 'PC3']).head()
df[others].head()

# Combining pca_fit and df[others] variables
final_df = pd.concat([pd.DataFrame(pca_fit, columns=['PC1', 'PC2', 'PC3']), df[others]], axis=1)
final_df.head()


# Modeling with linear regression and decision tree classifier
# Encoding label encoder
def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe


binary_cols = [col for col in final_df.columns if final_df[col].dtypes == 'O' and final_df[col].nunique() == 2]
for col in binary_cols:
    label_encoder(final_df, col)

final_df.head()

# Removing NaN values in 'Salary' variable
final_df.dropna(inplace=True)
final_df.head()
final_df.shape

# Selecting independent and dependent variables
y = final_df['Salary']
X = final_df.drop(['Salary'], axis=1)

# Linear regression model
lm = LinearRegression()
rmse = np.mean(np.sqrt(-cross_val_score(lm, X, y, cv=5, scoring='neg_mean_squared_error')))
rmse
y.mean() > rmse  # True

# Decision Tree Classifier
cart = DecisionTreeRegressor()
rmse = np.mean(np.sqrt(-cross_val_score(cart, X, y, cv=5, scoring='neg_mean_squared_error')))
rmse
y.mean() > rmse  # True

# CART hyperparameter optimization
cart_params = {'max_depth': range(1, 11),
               'min_samples_split': range(2, 20)}

cart_best_grid = GridSearchCV(cart, cart_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)
cart_best_grid.best_params_

# CART final model
cart_final = cart.set_params(**cart_best_grid.best_params_, random_state=17).fit(X, y)
rmse = np.mean(np.sqrt(-cross_val_score(cart_final, X, y, cv=5, scoring='neg_mean_squared_error')))
rmse
y.mean() > rmse  # True

######################################
# PCA VISUALIZATION
######################################

# Getting breast_cancer dataset
df = pd.read_csv('01_miuul_machine_learning_summercamp/00_datasets/breast_cancer.csv')
df.head()

# Independent and dependent variables
y = df['diagnosis']
X = df.drop(['id', 'diagnosis'], axis=1)

# Defining a function to obtain 2D dataset
def create_pca_df(X, y):
    X = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2)
    pca_fit = pca.fit_transform(X)
    pca_df = pd.DataFrame(data=pca_fit, columns=['PC1', 'PC2'])
    final_df = pd.concat([pca_df, pd.DataFrame(y)], axis=1)
    return final_df


pca_df = create_pca_df(X, y)

# Defining a function to visualize after reducing the dataset to 2D
def plot_pca(dataframe, target):
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('PC1', fontsize=15)
    ax.set_ylabel('PC2', fontsize=15)
    ax.set_title(f'{target.capitalize()}', fontsize=20)

    targets = list(dataframe[target].unique())
    colors = []

    while len(colors) != len(targets):
        colors.append(list(np.random.random_sample(3)))

    for t, color in zip(targets, colors):
        indices = dataframe[target] == t
        ax.scatter(dataframe.loc[indices, 'PC1'], dataframe.loc[indices, 'PC2'], color=color, s=50)
    ax.legend(targets)
    ax.grid()
    plt.show(block=True)


plot_pca(pca_df, 'diagnosis')

# Getting 'iris' dataset from seaborn library
df = sns.load_dataset('iris')
df.head()

# independent and dependent variables
y = df['species']
X = df.drop('species', axis=1)

# Getting the above defined pca functions
pca_df = create_pca_df(X, y)
plot_pca(pca_df, 'species')