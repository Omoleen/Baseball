import json
import lxml
from bs4 import BeautifulSoup
import requests
from sklearn import metrics
from xgboost import XGBClassifier
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
# from imblearn.combine import SMOTEENN, SMOTETomek
# from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from joblib import load,dump


# under 3.5
def evaluate_model(model, x_test, y_test, boundary=0.5):
    # from sklearn import metrics

    # Predict Test Data
    # y_pred = model.predict(x_test)
    y_pred = (model.predict_proba(x_test)[::,1] >= boundary).astype(int)
    # y_pred_proba = model.predict_proba(x_test)[::,1]
    # for i in y_pred_proba:
    #     if i<0.6:
    #         y_pred.append(0)
    #     else:
    #         y_pred.append(1)


    # Calculate accuracy, precision, recall, f1-score, and kappa score
    acc = metrics.accuracy_score(y_test, y_pred)
    prec = metrics.precision_score(y_test, y_pred)
    rec = metrics.recall_score(y_test, y_pred)
    f1 = metrics.f1_score(y_test, y_pred)
    # research on f0.5
    # f1 = metrics.fbeta_score(y_test, y_pred, beta=0.5)

    kappa = metrics.cohen_kappa_score(y_test, y_pred)
    clas = metrics.classification_report(y_test, y_pred)
    mcc = metrics.matthews_corrcoef(y_test, y_pred)


    # Calculate area under curve (AUC)
    y_pred_proba = model.predict_proba(x_test)[::,1]
    fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba)
    auc = metrics.roc_auc_score(y_test, y_pred_proba)  # when both classes are important

    # Precision recall AUC curve
    prauc = metrics.average_precision_score(y_test, y_pred_proba)  # when the positive class is the most important
    # optimal_idx = np.argmax(tpr-fpr)
    # optimal_threshold = _[optimal_idx]
    # print(f'Optimal threshold value is : {optimal_threshold}')
    roc_score = 0
    threshold_value = 0.2
    step_factor = 0.025
    thrsh_score = 0
    while threshold_value <= .8:
        temp_thresh = threshold_value
        predicted = (y_pred_proba >= temp_thresh).astype('int')
        temp_roc = metrics.average_precision_score(y_test, predicted)
        # temp_roc = metrics.roc_auc_score(y_test, y_pred_proba)
        # print(f'Threshold {temp_thresh} -- {temp_roc}')
        if roc_score < temp_roc:
            roc_score = temp_roc
            thrsh_score = threshold_value
        threshold_value = threshold_value + step_factor
    optimal_threshold = thrsh_score

    # Display confussion matrix
    cm = metrics.confusion_matrix(y_test, y_pred)

    return {'acc': acc, 'prec': prec, 'rec': rec, 'f1': f1, 'kappa': kappa,
            'fpr': fpr, 'tpr': tpr, 'auc': auc, 'cm': cm, 'clas': clas, 'mcc': mcc, 'optimal_threshold': optimal_threshold}


def build_model():
    df = pd.read_csv(r'baseball_test.csv')
    df['ou'] = np.where((df.ou == 0), 1, 0)
    X = df.drop(['ou'], axis=1)
    Y = df['ou']
    X_train, X_, y_train, y_ = train_test_split(X, Y,
                                                test_size=.4,
                                                random_state=42)
    X_cv, X_test, y_cv, y_test = train_test_split(X_, y_,
                                                test_size=.5,
                                                random_state=42)
    scaler = RobustScaler()
    X_train = scaler.fit_transform(X_train)
    X_cv = scaler.transform(X_cv)
    X_test = scaler.transform(X_test)
    print(y_train.value_counts())

    # xg = XGBClassifier(random_state=0)
    # xg.fit(X_train, y_train)


    # Building Random Forest model
    rf = RandomForestClassifier(random_state=0)
    rf.fit(X_train, y_train)

    # dump(rf, 'under35_each_baseball.joblib')

    # Evaluate Model
    rf_eval = evaluate_model(rf, X_train, y_train,)

    # Print result
    print('Random forest(train)')
    print('Accuracy:', rf_eval['acc'])
    print('Precision:', rf_eval['prec'])
    print('Recall:', rf_eval['rec'])
    print('F1 Score:', rf_eval['f1'])
    print('Cohens Kappa Score:', rf_eval['kappa'])
    print('Area Under Curve:', rf_eval['auc'])
    print("Mattew's Correlation Coefficient", rf_eval['mcc'])
    print('Optimal Threshold:', rf_eval['optimal_threshold'])
    print('Confusion Matrix:\n', rf_eval['cm'])
    print('Classification Report:\n', rf_eval['clas'])

    # Evaluate Model
    rf_eval = evaluate_model(rf, X_cv, y_cv, .6)

    # Print result
    print('Random forest')
    print('Accuracy:', rf_eval['acc'])
    print('Precision:', rf_eval['prec'])
    print('Recall:', rf_eval['rec'])
    print('F1 Score:', rf_eval['f1'])
    print('Cohens Kappa Score:', rf_eval['kappa'])
    print('Area Under Curve:', rf_eval['auc'])
    print("Mattew's Correlation Coefficient", rf_eval['mcc'])
    print('Optimal Threshold:', rf_eval['optimal_threshold'])
    print('Confusion Matrix:\n', rf_eval['cm'])
    print('Classification Report:\n', rf_eval['clas'])


    # Evaluate Model
    rf_eval = evaluate_model(rf, X_test, y_test,.6)

    # Print result
    print('Random forest(test)')
    print('Accuracy:', rf_eval['acc'])
    print('Precision:', rf_eval['prec'])
    print('Recall:', rf_eval['rec'])
    print('F1 Score:', rf_eval['f1'])
    print('Cohens Kappa Score:', rf_eval['kappa'])
    print('Area Under Curve:', rf_eval['auc'])
    print("Mattew's Correlation Coefficient", rf_eval['mcc'])
    print('Optimal Threshold:', rf_eval['optimal_threshold'])
    print('Confusion Matrix:\n', rf_eval['cm'])
    print('Classification Report:\n', rf_eval['clas'])


def scaler_model():
    saved_model = load('under35_each_baseball.joblib')
    df = pd.read_csv(r'baseball_test.csv')
    df['ou'] = np.where((df.ou == 0), 1, 0)
    X = df.drop(['ou'], axis=1)
    Y = df['ou']
    X_train, X_, y_train, y_ = train_test_split(X, Y,
                                                test_size=.4,
                                                random_state=42)
    scaler = RobustScaler()
    X_train = scaler.fit_transform(X_train)
    return scaler, saved_model


def autopredict(model, record, scaler):
    new = pd.DataFrame(record)
    record = scaler.transform(new)
    result = model.predict_proba(record)[:, 1]
    if result >= .6:
        # print('Under 3.5')
        return 'Under 3.5'
    elif result <= .4:
        # print('Over 3.5')
        return 'Over 3.5'
    else:
        # print('No Output')
        return 'No Output'

def predict():
    scaler, saved_model = scaler_model()
    while True:
        print('TEAM A')
        rating = input('Rating: ')
        pitcher = input('Pitcher: ')
        travel = input('Travel: ')
        adj_rating = input('Adj_Rating: ')
        win_prob = input('Win_Prob(e.g 0.70): ')
        record = {'rating': [rating],
                  'pitcher': [pitcher],
                  'travel': [travel],
                  'adj_rating': [adj_rating],
                  'win_prob': [win_prob],
                  }
        print(f'TEAM A: {autopredict(saved_model, record, scaler)}')
        print('TEAM B')
        rating = input('Rating: ')
        pitcher = input('Pitcher: ')
        travel = input('Travel: ')
        adj_rating = input('Adj_Rating: ')
        win_prob = input('Win_Prob(e.g 0.70): ')
        data = {'rating': [rating],
                'pitcher': [pitcher],
                'travel': [travel],
                'adj_rating': [adj_rating],
                'win_prob': [win_prob],
                }
        print(f'TEAM B: {autopredict(saved_model, record, scaler)}')


def append_multiple_lines(file_name, lines_to_append):
    # Open the file in append & read mode ('a+')
    with open(file_name, "a+") as file_object:
        appendEOL = False
        # Move read cursor to the start of file.
        file_object.seek(0)
        # Check if file is not empty
        data = file_object.read(100)
        if len(data) > 0:
            appendEOL = True
        # Iterate over each string in the list
        for line in lines_to_append:
            # If file is not empty then append '\n' before first line for
            # other lines always append '\n' before appending line
            if appendEOL == True:
                file_object.write("\n")
            else:
                appendEOL = True
            # Append element at the end of file
            file_object.write(line)






def future_matches():
    scaler, saved_model = scaler_model()
    teams = ['angels', 'astros', 'athletics', 'blue-jays', 'braves', 'brewers', 'cardinals', 'cubs', 'diamondbacks', 'dodgers',
             'giants', 'guardians', 'mariners', 'marlins', 'mets', 'nationals', 'orioles', 'padres', 'phillies',
             'pirates', 'rangers', 'rays', 'reds', 'red-sox', 'rockies', 'royals', 'tigers', 'twins', 'white-sox', 'yankees']
    all_teams = []
    games = []
    for team in teams:
        print(f'Generating {team} future matches....')
        script = requests.get(f'https://projects.fivethirtyeight.com/2022-mlb-predictions/{team}/').text
        soup = BeautifulSoup(script, 'lxml')
        upcoming = soup.findAll('tr', class_='tr')
        # print(len(upcoming))
        # print(upcoming)
        count = 0
        for game in upcoming:
            if count == 2:
                break
            # only print for the present day
            # if str(game.find('td', {'class': 'td td-datetime'}).find('span', {'class': 'day long'}).text).find('14') == 0:
            # for game in table:
            if game.find('td', {'class': 'td td-team team'}).find('span', {'class': 'team-name long'}).text.lower() == team.replace('-', ' '):
                try:
                    name = game.find('td', {'class': 'td td-team team'}).find('span', {'class': 'team-name long'}).text
                    date = game.find('td', {'class': 'td td-datetime'}).find('span', {'class': 'day long'}).text + ' ' + game.find('td', {'class': 'td td-datetime'}).find('span', {'class': 'time'}).text
                    pitcher = int(game.find('td', {'class': 'td number td-number pitcher-adj'}).find('span', {'class': 'value'}).text)
                    travel = int(game.find('td', {'class': 'td number td-number travel-adj'}).find('span', {'class': 'value'}).text)
                    i = game.find('td', {'data-game-status': 'upcoming'}).text
                    full = {
                        'name': game.find('td', {'class': 'td td-team team'}).find('span', {'class': 'team-name long'}).text,
                        'date': game.find('td', {'class': 'td td-datetime'}).find('span', {'class': 'day long'}).text + ' ' + game.find('td', {'class': 'td td-datetime'}).find('span', {'class': 'time'}).text
                    }
                    record = {
                        'rating': [int(game.find('td', {'class': 'td number td-number rating'}).text)],
                        'pitcher': [pitcher],
                        'travel': [travel],
                        'adj_rating': [int(game.find('td', {'class': 'td number td-number rating-adj'}).find('span', {'class': 'value'}).text)],
                        'win_prob': [int(game.find('td', {'class': 'td number td-number win-prob'}).text[:-1])/100],
                    }
                    full.update(record)
                    # print(record)
                    print(name)
                    print(date)
                    result_team = autopredict(saved_model, record, scaler)
                    # print(result_team)
                    print('__________________________________________')
                    print('')
                    if result_team != 'No Output':
                        list_of_lines = [name, date, result_team, '__________________________________________', '']
                        append_multiple_lines('future_games.txt', list_of_lines)
                        count += 1
                        games.append(full)
                except:
                    pass
    print(len(games))
    df = pd.DataFrame(games)
    print(df)
    df.to_csv(r"future_games_table.csv", index=False)

    # print(games)
    print('Future games stored in futures_games.txt')


if __name__ == '__main__':
    predict()  # take input manually to predict
    # future_matches()  # scrape matches and predict

# teams = ['angels', 'astros', 'athletics', 'blue-jays', 'braves', 'brewers', 'cardinals', 'cubs', 'diamondbacks', 'dodgers',
#          'giants', 'guardians', 'mariners', 'marlins', 'mets', 'nationals', 'orioles', 'padres', 'phillies',
#          'pirates', 'rangers', 'rays', 'reds', 'red-sox', 'rockies', 'royals', 'tigers', 'twins', 'white-sox', 'yankees']

# script = requests.get(f'https://projects.fivethirtyeight.com/2022-mlb-predictions/{team}/').text