# Import libraries
import warnings
warnings.filterwarnings('ignore')
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, cross_validate, RandomizedSearchCV, validation_curve
from sklearn.metrics import roc_auc_score, plot_roc_curve
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from helpers.eda import *
from helpers.data_prep import *

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

# Import dataset
df = load_csv('01_miuul_machine_learning_summercamp/00_datasets/diabetes.csv')
df.head()

# Selecting dependent and independent variables
y = df['outcome']
X = df.drop('outcome', axis=1)

##################################
# Gradient Boosting Machines
##################################

# GBM model
gbm_model = GradientBoostingClassifier(random_state=17)

# Getting hyperparameters
gbm_model.get_params()

# Error values of the gbm model before hyperparameter optimization
cv_results = cross_validate(gbm_model, X, y, cv=5, scoring=['accuracy', 'f1', 'roc_auc'])
cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()

# Determining the best hypreparameters
gbm_params = {'learning_rate': [0.01, 0.1],
              'max_depth': [3, 8, 10],
              'n_estimators': [100, 500, 1000],
              'subsample': [0.5, 0.7, 1]}

gbm_best_grid = GridSearchCV(gbm_model, gbm_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)
gbm_best_grid.best_params_

# GBM final model
gbm_final = gbm_model.set_params(**gbm_best_grid.best_params_, random_state=17).fit(X, y)

# Errors of the gbm final model
cv_results = cross_validate(gbm_final, X, y, cv=5, scoring=['accuracy', 'f1', 'roc_auc'])
cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()

# Feature importance of the variables
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


plot_importance(gbm_final, X)

# ROC_AUC plot of the final model
def roc_auc_curve(model, X, y):
    plot_roc_curve(model, X, y)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.title(f'ROC Curve for {type(model).__name__}')
    plt.show(block=True)


roc_auc_curve(gbm_model, X, y)
roc_auc_curve(gbm_final, X, y)

# Validation curve plot
def val_curve_params(model, X, y, param_name, param_range, scoring='roc_auc', cv=10):
    """
    cart_val_params = [['max_depth', range(1, 11)],
                   ['min_samples_split', range(2, 20)]]

    for i in range(len(cart_val_params)):
        val_curve_params(cart_final, X, y, cart_val_params[i][0], cart_val_params[i][1])
    """
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


val_curve_params(gbm_final, X, y, param_name='max_depth', param_range=range(1, 11), scoring='roc_auc')
val_curve_params(gbm_final, X, y, param_name='max_depth', param_range=range(1, 11), scoring='f1')
val_curve_params(gbm_final, X, y, param_name='max_depth', param_range=range(1, 11), scoring='accuracy')