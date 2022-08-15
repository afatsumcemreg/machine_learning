# Import libraries
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import GridSearchCV, cross_validate, validation_curve
from sklearn.metrics import plot_roc_curve
from lightgbm import LGBMClassifier
from helpers.eda import *
from helpers.data_prep import *

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

# Import dataset
df = load_csv('01_miuul_machine_learning_summercamp/00_datasets/diabetes.csv')
df.head()

##################################
# Light Gradient Boosting Machines (LightGBM)
##################################

# Selection of dependent and independent variables
X = df.drop('outcome', axis=1)
y = df['outcome']

# Model defining
lgbm_model = LGBMClassifier(random_state=17)

# Getting hyperparameters
lgbm_model.get_params()

# Errors of the base model before hyperparameter optimization
cv_results = cross_validate(lgbm_model, X, y, cv=5, scoring=['accuracy', 'f1', 'roc_auc'])
cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()

# Determining the best hyperparameters
lgbm_params = {'learning_rate': [0.01, 1.0],
               'n_estimators': [100, 300, 500, 1000],
               'colsample_bytree': [0.5, 0.7, 1.0]}

lgbm_best_grid = GridSearchCV(lgbm_model, lgbm_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)
lgbm_best_grid.best_params_

# Light GBM final model
lgbm_final = lgbm_model.set_params(**lgbm_best_grid.best_params_, random_state=17).fit(X, y)

# Errors of the base model after hyperparameter optimization
cv_results = cross_validate(lgbm_final, X, y, cv=5, scoring=['accuracy', 'f1', 'roc_auc'])
cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()

# Feature importances
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


plot_importance(lgbm_final, X)

# Plot roc_auc curve
def roc_auc_curve(model, X, y):
    plot_roc_curve(model, X, y)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.title(f'ROC Curve for {type(model).__name__}')
    plt.show(block=True)


roc_auc_curve(lgbm_final, X, y)

# Validation curve visualization
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


val_curve_params(lgbm_final, X, y, 'n_estimators', range(100, 1000, 100), scoring='roc_auc')
val_curve_params(lgbm_final, X, y, 'max_depth', range(100, 1000, 100), scoring='f1')
val_curve_params(lgbm_final, X, y, 'max_depth', range(100, 1000, 100), scoring='accuracy')