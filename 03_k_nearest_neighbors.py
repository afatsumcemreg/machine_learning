# importing libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, cross_validate, train_test_split
from sklearn.metrics import classification_report, roc_auc_score, plot_roc_curve
from helpers.data_prep import *
from helpers.eda import *

# import dataset
df = load_csv('01_miuul_machine_learning_summercamp/00_datasets/diabetes.csv')
df.head()

# grabing variables
cat_cols, num_cols, cat_but_car = grab_col_names(df)

# check dataframe
check_df(df)

df.outcome.value_counts()

#################################
# Data Preprocessing
#################################
y = df.outcome
X = df.drop('outcome', axis=1)

# Standardization
X_scaled = StandardScaler().fit_transform(X)
X = pd.DataFrame(X_scaled, columns=X.columns)

#################################
# KNN Modeling and Prediction
#################################
knn_model = KNeighborsClassifier().fit(X, y)
random_user = X.sample(1, random_state=45)
knn_model.predict(random_user)

#################################
# Evaluation of KNN Model
#################################
y_pred = knn_model.predict(X)
y_prob = knn_model.predict_proba(X)[:, 1]

print(classification_report(y, y_pred))
roc_auc_score(y, y_prob)

#################################
# Model validation
#################################
# Hold-Out Approach
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=45)
knn_model = KNeighborsClassifier().fit(X_train, y_train)
y_pred = knn_model.predict(X_test)
y_prob = knn_model.predict_proba(X_test)[:, 1]
print(classification_report(y_test, y_pred))
roc_auc_score(y_test, y_prob)

plot_roc_curve(knn_model, X_test, y_test)
plt.title('ROC Curve')
plt.plot([0, 1], [0, 1], 'r--')
plt.show(block=True)

# k-Fold Cross Validation Approach
cv_results = cross_validate(knn_model, X, y, cv=10, scoring=['accuracy', 'f1', 'roc_auc'])
cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()

#################################
# KNN Model Hyperparameter Optimization
#################################
knn_model.get_params()
knn_params = {'n_neighbors': range(2, 50)}

# Usign GridSerachCv method
knn_best_grid = GridSearchCV(knn_model, knn_params, cv=10, n_jobs=-1, verbose=True).fit(X, y)

knn_final_model = knn_model.set_params(**knn_best_grid.best_params_).fit(X, y)

# k-Fold Cross Validation Approach for KNN final model
cv_results = cross_validate(knn_final_model, X, y, cv=10, scoring=['accuracy', 'f1', 'roc_auc'])
cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()

#################################
# Prediction using KNN Model
#################################
X.columns
random_user = X.sample(1)
knn_final_model.predict(random_user)


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