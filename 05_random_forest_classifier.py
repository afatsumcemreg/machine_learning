# Import libraries
import warnings
warnings.filterwarnings('ignore')
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_validate, validation_curve
from sklearn.metrics import plot_roc_curve
import random
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
# Random Forest Classifier
##################################

# RF model
rf_model = RandomForestClassifier(random_state=17)

# Getting hyperparameters
rf_model.get_params()

# Determining the errors at the beginning of the study using cross validation approach
cv_results = cross_validate(rf_model, X, y, cv=10, scoring=['accuracy', 'f1', 'roc_auc'])
cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()

# Hyperparameter selection
rf_params = {'max_depth': [5, 8, None],
             'max_features': [3, 5, 7, 'auto'],
             'min_samples_split': [2, 5, 8, 15, 20],
             'n_estimators': [100, 200, 500]}

# Determining the optimal hyperparameters

rf_best_grid = GridSearchCV(rf_model, rf_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)
rf_best_grid.best_params_

# Final model
rf_final = rf_model.set_params(**rf_best_grid.best_params_, random_state=17).fit(X, y)

# Error values of the final model
cv_results = cross_validate(rf_final, X, y, cv=10, scoring=['accuracy', 'f1', 'roc_auc'])
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


plot_importance(rf_final, X)

# ROC_AUC plot of the final model
def roc_auc_curve(model, X, y):
    plot_roc_curve(model, X, y)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.title(f'ROC Curve for {type(model).__name__}')
    plt.show(block=True)


roc_auc_curve(rf_model, X, y)
roc_auc_curve(rf_final, X, y)

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


val_curve_params(rf_final, X, y, param_name='max_depth', param_range=range(1, 11), scoring='roc_auc')

# Determining the hyperparameter values using RandomSearchCV
rf_random_params = {'max_depth': np.random.randint(5, 50, 10),
                    'max_features': [3, 5, 7, 'auto', 'sqrt'],
                    'min_samples_split': np.random.randint(2, 50, 20),
                    'n_estimators': [int(x) for x in np.linspace(start=200, stop=1500, num=10)]}

rf_random = RandomizedSearchCV(estimator=rf_model, param_distributions=rf_random_params, n_iter=100, cv=5, verbose=True, random_state=42, n_jobs=-1).fit(X, y)
rf_random.best_params_

# rf_random_final model
rf_random_final = rf_model.set_params(**rf_random.best_params_, random_state=17).fit(X, y)

# Error values of the rf_random_final model
cv_results = cross_validate(rf_random_final, X, y, cv=5, scoring=['accuracy', 'f1', 'roc_auc'])
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


plot_importance(rf_random_final, X)

# ROC_AUC plot of the final model
def roc_auc_curve(model, X, y):
    plot_roc_curve(model, X, y)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.title(f'ROC Curve for {type(model).__name__}')
    plt.show(block=True)


roc_auc_curve(rf_random_final, X, y)

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


val_curve_params(rf_random_final, X, y, param_name='max_depth', param_range=range(5, 50, 5), scoring='roc_auc')
val_curve_params(rf_random_final, X, y, param_name='max_depth', param_range=range(5, 50, 5), scoring='f1')
val_curve_params(rf_random_final, X, y, param_name='max_depth', param_range=range(5, 50, 5), scoring='accuracy')