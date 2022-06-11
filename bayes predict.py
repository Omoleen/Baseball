import requests
import json
import lxml
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

df = pd.read_csv('https://projects.fivethirtyeight.com/mlb-api/mlb_elo_latest.csv')


df.drop([ 'elo_prob1', 'elo_prob2', 'season', 'neutral', 'playoff', 'elo1_post', 'elo2_post', 'pitcher1', 'pitcher2', 'rating1_post', 'rating2_post', ], axis=1, inplace=True)

# print(len(df))
# df
df = df[(df['date'] >= '2022-01-18') & (df['date'] < '2022-05-19')].copy()
# df.dropna(inplace=True)
df.reset_index(inplace=True)
df.drop(['index'], axis=1, inplace=True)
original = df.copy()

df.drop(['team1', 'team2', 'date', 'score1','score2' ], axis=1, inplace=True)
print(df)

original['Real results'] = np.where(original['score1'] + original['score2'] > 6, 1, 0)
# original['Real results'] = np.where((original['score2'] < original['score1']) & (original['score2'] + original['score1'] > 6), 1, 0)  # home win
print(original['Real results'].value_counts())
model = load('bayes_baseball_history.joblib')

# # Evaluate Model
# best_grid_eval = evaluate_model(model, df, original['Real results'])

# print('Grid search')
# # Print result
# print('Accuracy:', best_grid_eval['acc'])
# print('Precision:', best_grid_eval['prec'])
# print('Recall:', best_grid_eval['rec'])
# print('F1 Score:', best_grid_eval['f1'])
# print('Cohens Kappa Score:', best_grid_eval['kappa'])
# print('Area Under Curve:', best_grid_eval['auc'])
# print('Confusion Matrix:\n', best_grid_eval['cm'])

from sklearn import metrics



results = model.predict(df)
print(metrics.classification_report(original['Real results'], results))
# results
original['Model results'] = results.tolist()
original['Model results'] = np.where(original['Model results'] == 1, 'Over' , 'Under')
# original['Model results'] = np.where(original['Model results'] == 1, 'Home' , 'Away')
original['Real results'] = np.where(original['score1'] + original['score2'] > 6, 'Over', 'Under')
# original['Real results'] = np.where((original['score2'] < original['score1']) & (original['score2'] + original['score1'] > 6), 'Home', 'Away')  # away win
num_cols = ['elo1_pre', 'elo2_pre', 'rating1_pre', 'rating2_pre', 'pitcher1_rgs', 'pitcher2_rgs', 'pitcher1_adj', 'pitcher2_adj', 'rating_prob1', 'rating_prob2',]
original.drop(num_cols, axis=1, inplace=True)

original['team1'] = np.where(original['team1'] == 'ARI', 'DiamondBacks', original['team1'])
original['team1'] = np.where(original['team1'] == 'LAD', 'Dodgers', original['team1'])
original['team1'] = np.where(original['team1'] == 'MIN', 'Twins', original['team1'])
original['team1'] = np.where(original['team1'] == 'OAK', 'Athletics', original['team1'])
original['team1'] = np.where(original['team1'] == 'SFG', 'Giants', original['team1'])
original['team1'] = np.where(original['team1'] == 'COL', 'Rockies', original['team1'])
original['team1'] = np.where(original['team1'] == 'CHW', 'White Sox', original['team1'])
original['team1'] = np.where(original['team1'] == 'KCR', 'City Royals', original['team1'])
original['team1'] = np.where(original['team1'] == 'TEX', 'Rangers', original['team1'])
original['team1'] = np.where(original['team1'] == 'ANA', 'Angels', original['team1'])
original['team1'] = np.where(original['team1'] == 'MIL', 'Brewers', original['team1'])
original['team1'] = np.where(original['team1'] == 'ATL', 'Braves', original['team1'])
original['team1'] = np.where(original['team1'] == 'CHC', 'Chicago Cubs', original['team1'])
original['team1'] = np.where(original['team1'] == 'PIT', 'Pirates', original['team1'])
original['team1'] = np.where(original['team1'] == 'NYM', 'New York Mets', original['team1'])
original['team1'] = np.where(original['team1'] == 'STL', 'Cardinals', original['team1'])
original['team1'] = np.where(original['team1'] == 'BOS', 'Red Sox', original['team1'])
original['team1'] = np.where(original['team1'] == 'HOU', 'Asros', original['team1'])
original['team1'] = np.where(original['team1'] == 'TOR', 'Blue Jays', original['team1'])
original['team1'] = np.where(original['team1'] == 'SEA', 'Mariners', original['team1'])
original['team1'] = np.where(original['team1'] == 'BAL', 'Orioles', original['team1'])
original['team1'] = np.where(original['team1'] == 'NYY', 'Yankees', original['team1'])
original['team1'] = np.where(original['team1'] == 'TBD', 'Bay Rays', original['team1'])
original['team1'] = np.where(original['team1'] == 'DET', 'Tigers', original['team1'])
# original['team1'] = np.where(original['team1'] == 'COL', 'Rockies', original['team1'])
original['team1'] = np.where(original['team1'] == 'WSN', 'National', original['team1'])
original['team1'] = np.where(original['team1'] == 'FLA', 'Marlins', original['team1'])
original['team1'] = np.where(original['team1'] == 'CIN', 'Reds', original['team1'])
original['team1'] = np.where(original['team1'] == 'CLE', 'Guardians', original['team1'])
original['team1'] = np.where(original['team1'] == 'SDP', 'Padres', original['team1'])
original['team1'] = np.where(original['team1'] == 'PHI', 'Phillies', original['team1'])

original['team2'] = np.where(original['team2'] == 'ARI', 'DiamondBacks', original['team2'])
original['team2'] = np.where(original['team2'] == 'LAD', 'Dodgers', original['team2'])
original['team2'] = np.where(original['team2'] == 'MIN', 'Twins', original['team2'])
original['team2'] = np.where(original['team2'] == 'OAK', 'Athletics', original['team2'])
original['team2'] = np.where(original['team2'] == 'SFG', 'Giants', original['team2'])
original['team2'] = np.where(original['team2'] == 'COL', 'Rockies', original['team2'])
original['team2'] = np.where(original['team2'] == 'CHW', 'White Sox', original['team2'])
original['team2'] = np.where(original['team2'] == 'KCR', 'City Royals', original['team2'])
original['team2'] = np.where(original['team2'] == 'TEX', 'Rangers', original['team2'])
original['team2'] = np.where(original['team2'] == 'ANA', 'Angels', original['team2'])
original['team2'] = np.where(original['team2'] == 'MIL', 'Brewers', original['team2'])
original['team2'] = np.where(original['team2'] == 'ATL', 'Braves', original['team2'])
original['team2'] = np.where(original['team2'] == 'CHC', 'Chicago Cubs', original['team2'])
original['team2'] = np.where(original['team2'] == 'PIT', 'Pirates', original['team2'])
original['team2'] = np.where(original['team2'] == 'NYM', 'New York Mets', original['team2'])
original['team2'] = np.where(original['team2'] == 'STL', 'Cardinals', original['team2'])
original['team2'] = np.where(original['team2'] == 'BOS', 'Red Sox', original['team2'])
original['team2'] = np.where(original['team2'] == 'HOU', 'Asros', original['team2'])
original['team2'] = np.where(original['team2'] == 'TOR', 'Blue Jays', original['team2'])
original['team2'] = np.where(original['team2'] == 'SEA', 'Mariners', original['team2'])
original['team2'] = np.where(original['team2'] == 'BAL', 'Orioles', original['team2'])
original['team2'] = np.where(original['team2'] == 'NYY', 'Yankees', original['team2'])
original['team2'] = np.where(original['team2'] == 'TBD', 'Bay Rays', original['team2'])
original['team2'] = np.where(original['team2'] == 'DET', 'Tigers', original['team2'])
# original['team2'] = np.where(original['team2'] == 'COL', 'Rockies', original['team2'])
original['team2'] = np.where(original['team2'] == 'WSN', 'National', original['team2'])
original['team2'] = np.where(original['team2'] == 'FLA', 'Marlins', original['team2'])
original['team2'] = np.where(original['team2'] == 'CIN', 'Reds', original['team2'])
original['team2'] = np.where(original['team2'] == 'CLE', 'Guardians', original['team2'])
original['team2'] = np.where(original['team2'] == 'SDP', 'Padres', original['team2'])
original['team2'] = np.where(original['team2'] == 'PHI', 'Phillies', original['team2'])


# original.drop(original[original['Model results'] == 'Under'].index, inplace = True)

print(original)