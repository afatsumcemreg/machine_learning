# import libraries
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter(action='ignore', category=Warning)
import joblib
import pydotplus
import graphviz
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, export_graphviz, export_text
from sklearn.metrics import classification_report, roc_auc_score, plot_roc_curve
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate, validation_curve
from sklearn.preprocessing import StandardScaler
from skompiler import skompile
from helpers.eda import *
from helpers.data_prep import *

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 170)
pd.set_option('display.max_rows', 20)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

# import dataset
df = load_csv('01_miuul_machine_learning_summercamp/00_datasets/diabetes.csv')
df.head()

# independent and dependendt variables
y = df['outcome']
X = df.drop('outcome', axis=1)

##################################
# Exploaratory data analysis
##################################

# General picture
check_df(df)

# Grabing numerical and categorical variables
cat_cols, num_cols, cat_but_car = grab_col_names(df)

# Analysis of categorical variables
cat_summary(df, 'outcome')

# Analysis of numerical variables
for col in num_cols:
    num_summary(df, col, plot=True)

# Analysis of numerical variables to dependent variable
for col in num_cols:
    target_summary_with_num(df, 'outcome', col)

# correlation analysis
drop_list = high_correlated_cols(df, plot=True)
high_correlated_cols(df.drop(drop_list, axis=1), plot=True)

##################################
# Installing base model
##################################

# model set
df = load_csv('01_miuul_machine_learning_summercamp/00_datasets/diabetes.csv')
y = df['outcome']
X = df.drop('outcome', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=17)
cart_model = DecisionTreeClassifier(random_state=1).fit(X_train, y_train)
y_pred = cart_model.predict(X_test)
y_prob = cart_model.predict_proba(X_test)[:, 1]
print(classification_report(y_test, y_pred))
roc_auc_score(y_test, y_prob)


# feature importance for base model
def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(12, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show(block=True)
    if save:
        plt.savefig('importances.png')


plot_importance(cart_model, X_train)

##################################
# Feature engineering
##################################

# Missing values
zero_columns = [col for col in df.columns if (df[col].min() == 0 and col not in ["pregnancies", "outcome"])]

for col in zero_columns:
    df[col] = np.where(df[col] == 0, np.nan, df[col])

na_columns = missing_values_table(df, na_name=True)

missing_vs_target(df, 'outcome', na_columns)

# Filling the missing values
for col in zero_columns:
    df.loc[df[col].isnull(), col] = df[col].median()

# Outlier analysis
for col in df.columns:
    outlier_thresholds(df, col, q1=0.05, q3=0.95)
for col in df.columns:
    print(col, check_outlier(df, col, q1=0.05, q3=0.95))
    if check_outlier(df, col, q1=0.05, q3=0.95):
        replace_with_thresholds(df, col, q1=0.05, q3=0.95)
for col in df.columns:
    print(col, check_outlier(df, col, q1=0.05, q3=0.95))

# Feature extraction
# Cretaion new age variable
df.loc[(df["age"] >= 21) & (df["age"] < 50), "new_age_cat"] = "mature"
df.loc[(df["age"] >= 50), "new_age_cat"] = "senior"

# Creating new bmi variable
df['new_bmi'] = pd.cut(x=df['bmi'], bins=[0, 18.5, 24.9, 29.9, 100],
                       labels=["Underweight", "Healthy", "Overweight", "Obese"])

# Converting glucose to categorical variables
df["new_glucose"] = pd.cut(x=df["glucose"], bins=[0, 140, 200, 300], labels=["Normal", "Prediabetes", "Diabetes"])

# Creating a categorical variable together with the variables age and bmi
df.loc[(df["bmi"] < 18.5) & ((df["age"] >= 21) & (df["age"] < 50)), "new_age_bmi_nom"] = "underweightmature"
df.loc[(df["bmi"] < 18.5) & (df["age"] >= 50), "new_age_bmi_nom"] = "underweightsenior"
df.loc[((df["bmi"] >= 18.5) & (df["bmi"] < 25)) & (
        (df["age"] >= 21) & (df["age"] < 50)), "new_age_bmi_nom"] = "new_age_bmi_nom"
df.loc[((df["bmi"] >= 18.5) & (df["bmi"] < 25)) & (df["age"] >= 50), "new_age_bmi_nom"] = "healthysenior"
df.loc[((df["bmi"] >= 25) & (df["bmi"] < 30)) & (
        (df["age"] >= 21) & (df["age"] < 50)), "new_age_bmi_nom"] = "overweightmature"
df.loc[((df["bmi"] >= 25) & (df["bmi"] < 30)) & (df["age"] >= 50), "new_age_bmi_nom"] = "overweightsenior"
df.loc[(df["bmi"] > 18.5) & ((df["age"] >= 21) & (df["age"] < 50)), "new_age_bmi_nom"] = "obesemature"
df.loc[(df["bmi"] > 18.5) & (df["age"] >= 50), "new_age_bmi_nom"] = "obesesenior"

# Creating a categorical variable together with the variables age and glucose
df.loc[(df["glucose"] < 70) & ((df["age"] >= 21) & (df["age"] < 50)), "new_age_glucose_nom"] = "lowmature"
df.loc[(df["glucose"] < 70) & (df["age"] >= 50), "new_age_glucose_nom"] = "lowsenior"
df.loc[((df["glucose"] >= 70) & (df["glucose"] < 100)) & (
        (df["age"] >= 21) & (df["age"] < 50)), "new_age_glucose_nom"] = "normalmature"
df.loc[((df["glucose"] >= 70) & (df["glucose"] < 100)) & (df["age"] >= 50), "new_age_glucose_nom"] = "normalsenior"
df.loc[((df["glucose"] >= 100) & (df["glucose"] <= 125)) & (
        (df["age"] >= 21) & (df["age"] < 50)), "new_age_glucose_nom"] = "hiddenmature"
df.loc[((df["glucose"] >= 100) & (df["glucose"] <= 125)) & (df["age"] >= 50), "new_age_glucose_nom"] = "hiddensenior"
df.loc[(df["glucose"] > 125) & ((df["age"] >= 21) & (df["age"] < 50)), "new_age_glucose_nom"] = "highmature"
df.loc[(df["glucose"] > 125) & (df["age"] >= 50), "new_age_glucose_nom"] = "highsenior"


# Creating a categoric variable with insuline values
def set_insulin(dataframe, col_name="insulin"):
    if 16 <= dataframe[col_name] <= 166:
        return "Normal"
    else:
        return "Abnormal"


df["new_insulin_score"] = df.apply(set_insulin, axis=1)

df["new_glucose*insulin"] = df["glucose"] * df["insulin"]
df["new_glucose*pregnancies"] = df["glucose"] * df["pregnancies"]
df.shape
df.head()

##################################
# ENCODING
##################################
cat_cols, num_cols, cat_but_car = grab_col_names(df)

# Label encoding
binary_cols = [col for col in df.columns if df[col].dtypes == 'O' and df[col].nunique() == 2]
for col in binary_cols:
    label_encoder(df, col)
df.head()

# One hot encoding
cat_cols = [col for col in cat_cols if col not in binary_cols and col not in ['outcome']]
df = one_hot_encoder(df, cat_cols, drop_first=True)
df.shape
df.head()

##################################
# Standardization
##################################
num_cols
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])
df.head()

##################################
# CART Modeling
##################################
y = df['outcome']
X = df.drop('outcome', axis=1)

cart_model = DecisionTreeClassifier(random_state=17)

# k-Fold Cross Validation
cv_results = cross_validate(cart_model, X, y, cv=10, scoring=['accuracy', 'f1', 'roc_auc'])
cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()

# Hyperparameter optimization
cart_model.get_params()
cart_params = {'max_depth': range(1, 11),
               'min_samples_split': range(2, 20)}

cart_best_grid = GridSearchCV(cart_model, cart_params, cv=10, n_jobs=-1, verbose=True).fit(X, y)
cart_best_grid.best_params_
cart_best_grid.best_score_

# CART final model
cart_final = cart_model.set_params(**cart_best_grid.best_params_).fit(X, y)
cv_results = cross_validate(cart_final, X, y, cv=10, scoring=['accuracy', 'f1', 'roc_auc'])
cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()

def roc_auc_curve(model, X, y):
    plot_roc_curve(model, X, y)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.title(f'ROC Curve for {type(model).__name__}')
    plt.show(block=True)


roc_auc_curve(cart_final, X, y)


# Feature importance
def plot_importance(model, features, num=len(X), save=False):
    """
    plot_importance(rf_model, X_train)
    """
    feature_imp = pd.DataFrame({
        'Value': model.feature_importances_,
        'Feature': features.columns
    })

    plt.figure(figsize=(12, 8))
    sns.set(font_scale=1)
    sns.barplot(x='Value', y='Feature', data=feature_imp.sort_values(by='Value', ascending=False)[0: num])
    plt.title('Features')
    plt.tight_layout()
    plt.show(block=True)

    if save:
        plt.savefig('importance.png')


plot_importance(cart_final, X)


# Analyzing the model complexity with the leraning curves
def val_curve_params(model, X, y, param_name, param_range, scoring='roc_auc', cv=10):
    train_score, test_score = validation_curve(model, X=X, y=y, param_name=param_name, param_range=param_range,
                                               scoring=scoring, cv=cv)
    mean_train_score = np.mean(train_score, axis=1)
    mean_test_score = np.mean(test_score, axis=1)
    plt.plot(param_range, mean_train_score, label='Training Score', color='b')
    plt.plot(param_range, mean_test_score, label='Validation Score', color='r')
    plt.title(f'Validation Curve for {type(model).__name__}')
    plt.xlabel(f'Number of {param_name}')
    plt.ylabel(f'{scoring}')
    plt.tight_layout()
    plt.legend(loc='best')
    plt.show(block=True)


cart_val_params = [['max_depth', range(1, 11)],
                   ['min_samples_split', range(2, 20)]]

for i in range(len(cart_val_params)):
    val_curve_params(cart_final, X, y, cart_val_params[i][0], cart_val_params[i][1])

# Visualization
def tree_graph(model, col_names, file_name):
    tree_str = export_graphviz(model, feature_names=col_names, filled=True, out_file=None)
    graph = pydotplus.graph_from_dot_data(tree_str)
    graph.write_png(file_name)


tree_graph(model=cart_final, col_names=X.columns, file_name='cart_final.png')

# Decision rules
def decision_rules(model, feature_names=list(X.columns)):
    tree_rules = export_text(model, feature_names=feature_names)
    print(tree_rules)


decision_rules(cart_final)

# Exporting Python/SQL/Excel Codes of Decision Rules
## Exporting Python codes
print(skompile(cart_final.predict).to('python/code'))

## Exporting SQL codes
print(skompile(cart_final.predict).to('sqlalchemy/sqlite'))

## Exporting Excel codes
print(skompile(cart_final.predict).to('excel'))

# Prediction with the exported python codes
## Defining a function
def predict_with_rules(x):
    return ((((0 if x[6] <= 0.8082553148269653 else 0) if x[10] <= -0.2811785340309143
            else 0 if x[7] <= -0.4033912867307663 else 0) if x[11] <=
            0.446418434381485 else 0 if x[5] <= -0.9104179739952087 else 0 if x[6] <=
            0.4277203679084778 else 1) if x[1] <= 0.19211194664239883 else ((0 if x
            [11] <= -0.46984511613845825 else 0) if x[1] <= 0.7838578224182129 else
            0 if x[2] <= -0.9419257938861847 else 1) if x[23] <= 0.5 else (0 if x[7
            ] <= -0.2332157865166664 else 1) if x[1] <= 1.178355097770691 else 0 if
            x[4] <= -0.7207903861999512 else 1)


x1 = [24, 56, 34, 23, 100, 32, 21, 34, 56, 78, 90, 98, 65, 76, 54, 32, 31, 21, 12, 11, 15, 19, 25, 29, 32, 78]
predict_with_rules(x1)
x2 = [3, 1, 34, 0, 4, 8, 0, 1, 1, 78, 0, 1, 0, 76, 54, 0, 31, 21, 0, 11, 0, 0, 0, 0, 32, 78]
predict_with_rules(x2)

# Saving and calling the model

## Use 'joblib' library
joblib.dump(cart_final, 'cart_final.pkl')

## Reading the model from the disc
cart_final_from_disc = joblib.load('cart_final.pkl')
x = [3, 1, 34, 0, 4, 8, 0, 1, 1, 78, 0, 1, 0, 76, 54, 0, 31, 21, 0, 11, 0, 0, 0, 0, 32, 78]
cart_final_from_disc.predict(pd.DataFrame(x).T)