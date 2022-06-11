# import requests
# import json
# import lxml
# from bs4 import BeautifulSoup
import pandas as pd
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from joblib import dump, load
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

def train_model():
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
        clas = metrics.classification_report(y_test, y_pred)

        # Calculate area under curve (AUC)
        y_pred_proba = model.predict_proba(x_test)[::,1]
        fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba)
        auc = metrics.roc_auc_score(y_test, y_pred_proba)

        # Display confussion matrix
        cm = metrics.confusion_matrix(y_test, y_pred)

        return {'acc': acc, 'prec': prec, 'rec': rec, 'f1': f1, 'kappa': kappa,
                'fpr': fpr, 'tpr': tpr, 'auc': auc, 'cm': cm, 'clas': clas}




    df = pd.read_csv('https://projects.fivethirtyeight.com/mlb-api/mlb_elo_latest.csv')
    # print(len(df))
    df.drop([ 'season', 'neutral', 'playoff', 'elo1_post', 'elo2_post', 'pitcher1', 'pitcher2', 'rating1_post', 'rating2_post', 'elo_prob1', 'elo_prob2',], axis=1, inplace=True)
    df.dropna(inplace=True)
    # print(len(df))
    # print(df)
    df['ou'] = np.where(df['score1'] + df['score2'] > 13, 1, 0)
    df.drop(['date','team1', 'team2', 'score1', 'score2'], axis=1, inplace=True)
    print(df)
    # df.drop(['score1','score2', 'total',  'elo_prob1', 'elo_prob2', ], axis=1, inplace=True)
    print(df.describe())
    # standard scaler
    df_ready = df.copy()
    print(df_ready.describe())
    print(df_ready.columns)
    # scaler = StandardScaler()
    # num_cols = ['elo1_pre', 'elo2_pre', 'rating1_pre', 'rating2_pre', 'pitcher1_rgs', 'pitcher2_rgs', 'pitcher1_adj', 'pitcher2_adj', 'rating_prob1', 'rating_prob2']
    # # num_cols = ['elo1_pre', 'elo2_pre', 'elo_prob1', 'elo_prob2', 'rating1_pre', 'rating2_pre', 'pitcher1_rgs', 'pitcher2_rgs', 'pitcher1_adj', 'pitcher2_adj']
    # df_ready[num_cols] = scaler.fit_transform(df[num_cols])
    # # print(df_ready)
    print(df_ready.groupby("ou").size())  # print number of dead or alive
    df_ready['ou'].value_counts()
    print(df_ready.isnull().sum())
    # # upsampling data
    # # using Synthetic Minority Oversampling Technique to upsample
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
                                                        random_state=1)


    from sklearn.ensemble import RandomForestClassifier

    # Building Random Forest model
    rf = RandomForestClassifier(random_state=0)
    rf.fit(X_train, y_train)


    # Evaluate Model
    rf_eval = evaluate_model(rf, X_test, y_test)

    # Print result
    print('Random forest')
    print('Accuracy:', rf_eval['acc'])
    print('Precision:', rf_eval['prec'])
    print('Recall:', rf_eval['rec'])
    print('F1 Score:', rf_eval['f1'])
    print('Cohens Kappa Score:', rf_eval['kappa'])
    print('Area Under Curve:', rf_eval['auc'])
    print('Confusion Matrix:\n', rf_eval['cm'])
    print('Classification Report:\n', rf_eval['clas'])
    #
    # # testing
    preds = rf.predict(X_test)
    print(pd.crosstab(y_test, preds, rownames=['Actual Result'], colnames=['Predicted Result']))
    print(list(zip(X_train, rf.feature_importances_)))
    #
    # dump(rf, 'over 13 baseball.joblib')
    #
    #


# PREDICT
def predict():
    df = pd.read_csv('https://projects.fivethirtyeight.com/mlb-api/mlb_elo_latest.csv')
    df = df[(df['date'] >= '2022-06-12') & (df['date'] < '2022-06-20')].copy()
    # print(len(df))
    df.drop([ 'elo_prob1', 'elo_prob2', 'season', 'neutral', 'playoff', 'elo1_post', 'elo2_post', 'pitcher1', 'pitcher2', 'rating1_post', 'rating2_post', 'score1','score2'], axis=1, inplace=True)
    # df.dropna(inplace=True)
    # print(len(df))
    print(df)

    df.dropna(axis=0, inplace=True)
    df.reset_index(inplace=True)
    df.drop(['index'], axis=1, inplace=True)

    original = df.copy()
    # df.drop(['score1','score2'], axis=1, inplace=True)
    # print(original)
    df.drop(['team1', 'team2', 'date'], axis=1, inplace=True)
    pd.set_option('display.max_columns', None)

    # print(df)
    # print(df.columns)
    # print(df.isnull().sum())
    model = load('over 13 baseball.joblib')
    results = model.predict(df)

    original['Model results'] = results.tolist()
    original['Model results'] = np.where(original['Model results'] == 1, 'Over 13', 'Under')

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


    original.drop(['elo1_pre', 'elo2_pre', 'rating1_pre', 'rating2_pre', 'pitcher1_rgs',
                   'pitcher2_rgs', 'pitcher1_adj', 'pitcher2_adj', 'rating_prob1',
                   'rating_prob2'], axis=1, inplace=True)
    # print(original.loc[original['Model results'] == 'Over 13'])

    result = original.loc[original['Model results'] == 'Over 13'].copy()
    result.reset_index(inplace=True)
    result.drop(['index'], axis=1, inplace=True)
    print(result)



# train_model()
predict()
# x
