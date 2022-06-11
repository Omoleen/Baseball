import timeit
import requests
import json
import lxml
# from google.colab import files
from bs4 import BeautifulSoup
from sklearn import metrics
import pandas as pd
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from joblib import dump, load
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split


# df = pd.read_csv('baseball history.csv')
# df.dropna(inplace=True)
# print(len(df))
# df['ou'] = np.where(df['total'] > 6, 1, 0)
df = pd.read_csv('https://projects.fivethirtyeight.com/mlb-api/mlb_elo_latest.csv')
df.drop([ 'elo_prob1', 'elo_prob2', 'season', 'neutral', 'playoff', 'elo1_post', 'elo2_post', 'pitcher1', 'pitcher2', 'rating1_post', 'rating2_post', ], axis=1, inplace=True)

# print(len(df))

df = df[(df['date'] >= '2022-01-18') & (df['date'] < '2022-05-22')].copy()

df.dropna(inplace=True)
df.reset_index(inplace=True)
df.drop(['index'], axis=1, inplace=True)
print(df.tail(150))


# df['ou'] = np.where((df['score2'] > df['score1']) & (df['score1'] > 2), 1, 0)  # away win
df['ou'] = np.where((df['score2'] + df['score1'] > 6), 1, 0)  # home win
df.drop(['score1', 'score2', 'date', 'team1', 'team2'], axis=1, inplace=True)
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
sm = SMOTE(random_state=42)
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
from sklearn.model_selection import cross_val_score

from hyperopt import hp,fmin,tpe,STATUS_OK,Trials

space = {'criterion': hp.choice('criterion', ['entropy', 'gini']),
         'max_depth': hp.quniform('max_depth', 10, 1200, 10),
         'max_features': hp.choice('max_features', ['auto', 'sqrt','log2', None]),
         'min_samples_leaf': hp.uniform('min_samples_leaf', 0, 0.5),
         'min_samples_split' : hp.uniform ('min_samples_split', 0, 1),
         'n_estimators' : hp.choice('n_estimators', [10, 50, 300, 750, 1200,1300,1500])
         }

def objective(space):
    model = RandomForestClassifier(criterion = space['criterion'], max_depth = space['max_depth'],
                                   max_features = space['max_features'],
                                   min_samples_leaf = space['min_samples_leaf'],
                                   min_samples_split = space['min_samples_split'],
                                   n_estimators = space['n_estimators'],
                                   )
    #5 times cross validation fives 5 accuracies=>mean of these accuracies will be considered
    accuracy = cross_val_score(model, X_train, y_train, cv = 5).mean()
    # We aim to maximize accuracy, therefore we return it as a negative value
    return {'loss': -accuracy, 'status': STATUS_OK }


trials = Trials()
best = fmin(fn= objective,
            space= space,
            algo= tpe.suggest,
            max_evals = 80,
            trials= trials)

crit = {0: 'entropy', 1: 'gini'}
feat = {0: 'auto', 1: 'sqrt', 2: 'log2', 3: None}
est = {0: 10, 1: 50, 2: 300, 3: 750, 4: 1200,5:1300,6:1500}
print(crit[best['criterion']])
print(feat[best['max_features']])
print(est[best['n_estimators']])

trainedforest = RandomForestClassifier(criterion = crit[best['criterion']], max_depth = best['max_depth'],
                                       max_features = feat[best['max_features']],
                                       min_samples_leaf = best['min_samples_leaf'],
                                       min_samples_split = best['min_samples_split'],
                                       n_estimators = est[best['n_estimators']]).fit(X_train,y_train)
predictionforest = trainedforest.predict(X_test)
print(metrics.confusion_matrix(y_test,predictionforest))
print(metrics.accuracy_score(y_test,predictionforest))
print(metrics.classification_report(y_test,predictionforest))

dump(trainedforest, 'bayes_baseball_history.joblib')


# files.download('bayes_baseball_history.joblib')
#



# gpu()
