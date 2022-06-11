import requests
import json
import lxml
# from google.colab import files
from bs4 import BeautifulSoup
import pandas as pd
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from joblib import dump, load
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split


def evaluate_model(model, x_test, y_test):
    from sklearn import metrics

    # Predict Test Data
    y_pred = model.predict(x_test)

    # Calculate accuracy, precision, recall, f1-score, and kappa score
    acc = metrics.accuracy_score(y_test, y_pred)
    prec = metrics.precision_score(y_test, y_pred)
    rec = metrics.recall_score(y_test, y_pred)
    f1 = metrics.f1_score(y_test, y_pred)
    kappa = metrics.cohen_kappa_score(y_test, y_pred)

    # Calculate area under curve (AUC)
    y_pred_proba = model.predict_proba(x_test)[::,1]
    fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba)
    auc = metrics.roc_auc_score(y_test, y_pred_proba)

    # Display confussion matrix
    cm = metrics.confusion_matrix(y_test, y_pred)

    return {'acc': acc, 'prec': prec, 'rec': rec, 'f1': f1, 'kappa': kappa,
            'fpr': fpr, 'tpr': tpr, 'auc': auc, 'cm': cm}




df = pd.read_csv('baseball history.csv')
df.dropna(inplace=True)
print(len(df))
df['ou'] = np.where(df['total'] > 6, 1, 0)
# df['ou'] = np.where((df['score2'] > df['score1']) & (df['score1'] > 2), 1, 0)  # away win
# df['ou'] = np.where((df['score2'] < df['score1']) & (int(df['score2']) + int(df['score1']) > 6), 1, 0)  # home win
df.drop(['score1','score2', 'total',  'elo_prob1', 'elo_prob2', ], axis=1, inplace=True)
print(df.describe())
# standard scaler
df_ready = df.copy()
# scaler = StandardScaler()
# num_cols = ['elo1_pre', 'elo2_pre', 'rating1_pre', 'rating2_pre', 'pitcher1_rgs', 'pitcher2_rgs', 'pitcher1_adj', 'pitcher2_adj', 'rating_prob1', 'rating_prob2']
# # num_cols = ['elo1_pre', 'elo2_pre', 'elo_prob1', 'elo_prob2', 'rating1_pre', 'rating2_pre', 'pitcher1_rgs', 'pitcher2_rgs', 'pitcher1_adj', 'pitcher2_adj']
# df_ready[num_cols] = scaler.fit_transform(df[num_cols])
# print(df_ready)
print(df_ready.groupby("ou").size())  # print number of dead or alive
df_ready['ou'].value_counts()
print(df_ready.isnull().sum())
# upsampling data
# using Synthetic Minority Oversampling Technique to upsample
X_train_smote = df_ready.drop(["ou"], axis=1)
Y_train_smote = df_ready["ou"]
# print(X_train_smote.shape, Y_train_smote.shape)
sm = SMOTE(random_state=2)
X_train_res, Y_train_res = sm.fit_resample(X_train_smote, Y_train_smote.ravel())
print(X_train_res.shape, Y_train_res.shape)
print(len(Y_train_res[Y_train_res == 0]), len(Y_train_res[Y_train_res == 1]))
print(X_train_res)  # dataset
print(Y_train_res)  # over or under
X_train, X_test, y_train, y_test = train_test_split(X_train_res, Y_train_res,
                                                    shuffle=True,
                                                    test_size=0.2,
                                                    random_state=42)


from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier

# Building Random Forest model
# rf = RandomForestClassifier(random_state=0, criterion="entropy")
# rf.fit(X_train, y_train)

# y_pred = rf.predict(X_test)
# print(metrics.classification_report(y_test, y_pred))
# # Evaluate Model
# rf_eval = evaluate_model(rf, X_test, y_test)

# # Print result
# print('Random forest')
# print('Accuracy:', rf_eval['acc'])
# print('Precision:', rf_eval['prec'])
# print('Recall:', rf_eval['rec'])
# print('F1 Score:', rf_eval['f1'])
# print('Cohens Kappa Score:', rf_eval['kappa'])
# print('Area Under Curve:', rf_eval['auc'])
# print('Confusion Matrix:\n', rf_eval['cm'])

# # testing
# preds = rf.predict(X_test)
# print(pd.crosstab(y_test, preds, rownames=['Actual Result'], colnames=['Predicted Result']))
# print(list(zip(X_train, rf.feature_importances_)))

# dump(rf, 'random_forest_baseball_history.joblib')
# files.download('random_forest_baseball_history.joblib')

# model optimization using cross validation
from sklearn.model_selection import GridSearchCV

# Create the parameter grid based on the results of random search
param_grid = {
    'max_depth': [50, 80, 100],
    'max_features': [2, 3, 4],
    'min_samples_leaf': [3, 4, 5],
    'min_samples_split': [8, 10, 12],
    'n_estimators': [100, 300, 500, 750, 1000]
}

# Create a base model
rf_grids = RandomForestClassifier(random_state=0)

# Initiate the grid search model
grid_search = GridSearchCV(estimator=rf_grids, param_grid=param_grid, scoring='accuracy',
                           cv=5, n_jobs=-1, verbose=3)

# Fit the grid search to the data
grid_search.fit(X_train, y_train)

print(grid_search.best_params_)


# Select best model with best fit
best_grid = grid_search.best_estimator_

# Evaluate Model
best_grid_eval = evaluate_model(best_grid, X_test, y_test)

print('Grid search')
# Print result
print('Accuracy:', best_grid_eval['acc'])
print('Precision:', best_grid_eval['prec'])
print('Recall:', best_grid_eval['rec'])
print('F1 Score:', best_grid_eval['f1'])
print('Cohens Kappa Score:', best_grid_eval['kappa'])
print('Area Under Curve:', best_grid_eval['auc'])
print('Confusion Matrix:\n', best_grid_eval['cm'])

dump(best_grid, 'cross_valid_baseball_history 90.joblib')


# files.download('cross_valid_baseball_history.joblib')


# ANOVA feature selection for numeric input and categorical output
from sklearn.datasets import make_classification
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
# define feature selection
# fs = SelectKBest(score_func=f_classif, k=12)
# X_new = fs.fit(X_train, y_train)
# X_train.columns.values[fs.get_support()]