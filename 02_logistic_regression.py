# importing libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, plot_roc_curve, roc_auc_score
from helpers.data_prep import *
from helpers.eda import *

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 170)
pd.set_option('display.max_rows', 20)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

# import dataset
df = load_csv('01_miuul_machine_learning_summercamp/00_datasets/diabetes.csv')
df.head()

# grabing variables
cat_cols, num_cols, cat_but_car = grab_col_names(df)

# filling the outliers with thresholds
for col in num_cols:
    outlier_thresholds(df, col, q1=0.05, q3=0.95)
for col in num_cols:
    print(col, check_outlier(df, col, q1=0.05, q3=0.95))
for col in num_cols:
    replace_with_thresholds(df, col, q1=0.05, q3=0.95)
for col in num_cols:
    print(col, check_outlier(df, col, q1=0.05, q3=0.95))

#################################
# Exploratory Data Analysis
#################################

# dependentent variable analysis
df['outcome'].value_counts()
sns.countplot(x=df['outcome'])
plt.show(block=True)

# independent variable analysis
check_df(df)
sns.histplot(x=df['bloodpressure'])
plt.show(block=True)


# defining a function to show numerical variables
num_cols = [col for col in num_cols if 'outcome' not in col]
for col in num_cols:
    plot_numerical_col(df, col)

# Together evaluation of independent and dependent variables
def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")


for col in num_cols:
    target_summary_with_num(df, 'outcome', col)


#################################
# Data Preprocessing
#################################
df.isnull().sum()
df.describe().T

for col in num_cols:
    print(col, check_outlier(df, col, q1=0.05, q3=0.95))

for col in num_cols:
    replace_with_thresholds(df, 'insulin', q1=0.05, q3=0.95)


#################################
# Scaling the Variables
#################################
for col in num_cols:
    df[col] = RobustScaler().fit_transform(df[[col]])
df.head()


#################################
# Modeling
#################################
y = df['outcome']
X = df.drop('outcome', axis=1)
log_model = LogisticRegression().fit(X, y)
log_model.intercept_
log_model.coef_
y_pred = log_model.predict(X)
y_pred[:10]
y[:10]


#################################
# Evaluation of the model
#################################
# confusion matrix
def plot_confusion_matrix(y, y_pred):
    acc = round(accuracy_score(y, y_pred), 3)
    cm = confusion_matrix(y, y_pred)
    sns.heatmap(cm, annot=True, fmt='.0f')
    plt.xlabel('y_pred')
    plt.ylabel('y')
    plt.title('Accuracy score: {0}'.format(acc), size=10)
    plt.show(block=True)


plot_confusion_matrix(y, y_pred)

# classification report
print(classification_report(y, y_pred))

# roc_acuc value
y_prob = log_model.predict_proba(X)[:, 1]
roc_auc_score(y, y_prob)

#################################
# Model validation
#################################
# Hold-Out Approach
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=17)
log_model = LogisticRegression().fit(X_train, y_train)
y_pred = log_model.predict(X_test)
y_prob = log_model.predict_proba(X_test)[:, 1]
print(classification_report(y_test, y_pred))
roc_auc_score(y_test, y_prob)

## roc curve
plot_roc_curve(log_model, X_test, y_test)
plt.title('ROC Curve')
plt.plot([0, 1], [0, 1], 'r--')
plt.show(block=True)

# k-Fold Cross Validation Approach
X = df.drop('outcome', axis=1)
y = df['outcome']
log_model = LogisticRegression().fit(X, y)
cv_results = cross_validate(log_model, X, y, cv=10, scoring=['accuracy', 'precision', 'recall', 'f1', 'roc_auc'])
print(cv_results['test_accuracy'].mean())
print(cv_results['test_precision'].mean())
print(cv_results['test_recall'].mean())
print(cv_results['test_f1'].mean())
print(cv_results['test_roc_auc'].mean())

#################################
# Prediction for a New Observation
#################################
print(X.columns)
random_user = X.sample(1, random_state=45)
print(log_model.predict(random_user))


#################################
# Modeling with Classification Models
#################################
def all_models(X, y, test_size=0.20, random_state=12345, classification=True):
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report, plot_roc_curve, mean_squared_error

    # Tum Base Modeller (Classification)
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.ensemble import GradientBoostingClassifier
    from catboost import CatBoostClassifier
    from lightgbm import LGBMClassifier
    from sklearn.svm import SVC

    # Tum Base Modeller (Regression)
    from catboost import CatBoostRegressor
    from lightgbm import LGBMRegressor
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.svm import SVR
    from sklearn.tree import DecisionTreeRegressor
    from xgboost import XGBRegressor

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    all_models = []

    if classification:
        models = [('LR', LogisticRegression(random_state=random_state)),
                  ('KNN', KNeighborsClassifier()),
                  ('CART', DecisionTreeClassifier(random_state=random_state)),
                  ('RF', RandomForestClassifier(random_state=random_state)),
                  ('SVM', SVC(gamma='auto', random_state=random_state)),
                  ('XGB', GradientBoostingClassifier(random_state=random_state)),
                  ("LightGBM", LGBMClassifier(random_state=random_state)),
                  ("CatBoost", CatBoostClassifier(verbose=False, random_state=random_state))]

        for name, model in models:
            model.fit(X_train, y_train)
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            acc_train = accuracy_score(y_train, y_pred_train)
            acc_test = accuracy_score(y_test, y_pred_test)
            precision_train = precision_score(y_train, y_pred_train)
            precision_test = precision_score(y_test, y_pred_test)
            recall_train = recall_score(y_train, y_pred_train)
            recall_test = recall_score(y_test, y_pred_test)
            f1_train = f1_score(y_train, y_pred_train)
            f1_test = f1_score(y_test, y_pred_test)
            roc_auc_train = roc_auc_score(y_train, y_pred_train)
            roc_auc_test = roc_auc_score(y_test, y_pred_test)
            values = dict(name=name, acc_train=acc_train, acc_test=acc_test, precision_train=precision_train, precision_test=precision_test, recall_train=recall_train, recall_test=recall_test, f1_train=f1_train, f1_test=f1_test, roc_auc_train=roc_auc_train, roc_auc_test=roc_auc_test)
            all_models.append(values)

        sort_method = False
    else:
        models = [('LR', LinearRegression()),
                  ("Ridge", Ridge()),
                  ("Lasso", Lasso()),
                  ("ElasticNet", ElasticNet()),
                  ('KNN', KNeighborsRegressor()),
                  ('CART', DecisionTreeRegressor()),
                  ('RF', RandomForestRegressor()),
                  ('SVR', SVR()),
                  ('GBM', GradientBoostingRegressor()),
                  ("XGBoost", XGBRegressor()),
                  ("LightGBM", LGBMRegressor()),
                  ("CatBoost", CatBoostRegressor(verbose=False))]

        for name, model in models:
            model.fit(X_train, y_train)
            y_pred_test = model.predict(X_test)
            y_pred_train = model.predict(X_train)
            rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
            rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
            values = dict(name=name, RMSE_TRAIN=rmse_train, RMSE_TEST=rmse_test)
            all_models.append(values)

        sort_method = True
    all_models_df = pd.DataFrame(all_models)
    all_models_df = all_models_df.sort_values(all_models_df.columns[2], ascending=sort_method)
    print(all_models_df)
    return all_models_df


all_models = all_models(X, y, test_size=0.2, random_state=46, classification=True)