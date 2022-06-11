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


def scrape_data():
    teams = ['angels','astros', 'athletics', 'blue-jays', 'braves', 'brewers', 'cardinals', 'cubs', 'diamondbacks', 'dodgers',
             'giants', 'guardians', 'mariners', 'marlins', 'mets', 'nationals', 'orioles', 'padres', 'phillies',
             'pirates', 'rangers', 'rays', 'reds', 'red-sox', 'rockies', 'royals', 'tigers', 'twins', 'white-sox', 'yankees']
    # teams = ['diamondbacks']
    all_teams = []
    games = []
    for team in teams:
        script = requests.get(f'https://projects.fivethirtyeight.com/2022-mlb-predictions/{team}/').text
        soup = BeautifulSoup(script, 'lxml')
        completed = soup.findAll('tr', class_='tr')
        # ebayproducts = soup.findAll('div', class_='s-item__wrapper clearfix')
        # print(len(completed))
        count = 0  #it is used to scrape the las x number of matches
        for game in completed:
            # if count == 4:
            #     break
            try:
                if game.find('td', {'class': 'td number td-number pitcher-adj'}).text[0] == '+':
                    pitcher = int(game.find('td', {'class': 'td number td-number pitcher-adj'}).text[1:])
                else:
                    pitcher = int(game.find('td', {'class': 'td number td-number pitcher-adj'}).text[1:])*-1
                if game.find('td', {'class': 'td number td-number travel-adj'}).text[0] == '+':
                    travel = int(game.find('td', {'class': 'td number td-number travel-adj'}).text[1:])
                else:
                    travel = int(game.find('td', {'class': 'td number td-number travel-adj'}).text[1:])*-1
                if int(game.find('td', {'data-game-status': 'completed'}).text) >= 3:
                    ou = 1  # 3,4,5...
                else:
                    ou = 0  # 0,1,2
                record = {
                    'rating': int(game.find('td', {'class': 'td number td-number rating'}).text),
                    'pitcher': pitcher,
                    'travel': travel,
                    'adj_rating': int(game.find('td', {'class': 'td number td-number rating-adj'}).text[1:]),
                    'win_prob': int(game.find('td', {'class': 'td number td-number win-prob'}).text[:-1])/100,
                    # 'score': int(game.find('td', {'data-game-status': 'completed'}).text),
                    'ou': ou,
                }

                games.append(record)
                # count += 1
            except:
                pass
        print(f'{team} added to the records')

    df = pd.DataFrame(games)
    print(df)
    df.to_csv(r"baseball_test.csv", index=False)  # used to save the last x number of matches into a csv
    # df.to_csv(r"baseball.csv", index=False)


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


def build_model():
    df = pd.read_csv('baseball.csv')
    # print(df.describe())
    # standard scaler
    df_ready = df.copy()
    scaler = StandardScaler()
    num_cols = ['rating', 'pitcher', 'travel', 'adj_rating', 'win_prob']
    df_ready[num_cols] = scaler.fit_transform(df[num_cols])
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
                                                        test_size=0.1,
                                                        random_state=1)

    # Show the Training and Testing Data
    print('Shape of training feature:', X_train.shape)
    print('Shape of testing feature:', X_test.shape)
    print('Shape of training label:', y_train.shape)
    print('Shape of training label:', y_test.shape)
    #
    #
    from sklearn import tree

    # Building Decision Tree model
    dtc = tree.DecisionTreeClassifier(random_state=0)
    dtc.fit(X_train, y_train)

    from sklearn.ensemble import RandomForestClassifier

    # Building Random Forest model
    rf = RandomForestClassifier(random_state=0)
    rf.fit(X_train, y_train)
    from sklearn.naive_bayes import GaussianNB

    # Building Naive Bayes model
    nb = GaussianNB()
    nb.fit(X_train, y_train)

    from sklearn.neighbors import KNeighborsClassifier

    # Building KNN model
    knn = KNeighborsClassifier()
    knn.fit(X_train, y_train)

    # Evaluate Model
    knn_eval = evaluate_model(knn, X_test, y_test)

    # Print result
    print('K neighbours')
    print('Accuracy:', knn_eval['acc'])
    print('Precision:', knn_eval['prec'])
    print('Recall:', knn_eval['rec'])
    print('F1 Score:', knn_eval['f1'])
    print('Cohens Kappa Score:', knn_eval['kappa'])
    print('Area Under Curve:', knn_eval['auc'])
    print('Confusion Matrix:\n', knn_eval['cm'])

    # Evaluate Model
    nb_eval = evaluate_model(nb, X_test, y_test)

    # Print result
    print('naive bayes')
    print('Accuracy:', nb_eval['acc'])
    print('Precision:', nb_eval['prec'])
    print('Recall:', nb_eval['rec'])
    print('F1 Score:', nb_eval['f1'])
    print('Cohens Kappa Score:', nb_eval['kappa'])
    print('Area Under Curve:', nb_eval['auc'])
    print('Confusion Matrix:\n', nb_eval['cm'])

    # Evaluate Model
    dtc_eval = evaluate_model(dtc, X_test, y_test)

    # Print result
    print('Decision Tree')
    print('Accuracy:', dtc_eval['acc'])
    print('Precision:', dtc_eval['prec'])
    print('Recall:', dtc_eval['rec'])
    print('F1 Score:', dtc_eval['f1'])
    print('Cohens Kappa Score:', dtc_eval['kappa'])
    print('Area Under Curve:', dtc_eval['auc'])
    print('Confusion Matrix:\n', dtc_eval['cm'])

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


    # plotting graph to compare algorithms
    # Intitialize figure with two plots
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle('Model Comparison', fontsize=16, fontweight='bold')
    fig.set_figheight(7)
    fig.set_figwidth(14)
    fig.set_facecolor('white')

    # First plot
    ## set bar size
    barWidth = 0.2
    dtc_score = [dtc_eval['acc'], dtc_eval['prec'], dtc_eval['rec'], dtc_eval['f1'], dtc_eval['kappa']]
    rf_score = [rf_eval['acc'], rf_eval['prec'], rf_eval['rec'], rf_eval['f1'], rf_eval['kappa']]
    nb_score = [nb_eval['acc'], nb_eval['prec'], nb_eval['rec'], nb_eval['f1'], nb_eval['kappa']]
    knn_score = [knn_eval['acc'], knn_eval['prec'], knn_eval['rec'], knn_eval['f1'], knn_eval['kappa']]

    ## Set position of bar on X axis
    r1 = np.arange(len(dtc_score))
    r2 = [x + barWidth for x in r1]
    r3 = [x + barWidth for x in r2]
    r4 = [x + barWidth for x in r3]

    ## Make the plot
    ax1.bar(r1, dtc_score, width=barWidth, edgecolor='white', label='Decision Tree')
    ax1.bar(r2, rf_score, width=barWidth, edgecolor='white', label='Random Forest')
    ax1.bar(r3, nb_score, width=barWidth, edgecolor='white', label='Naive Bayes')
    ax1.bar(r4, knn_score, width=barWidth, edgecolor='white', label='K-Nearest Neighbors')

    ## Configure x and y axis
    ax1.set_xlabel('Metrics', fontweight='bold')
    labels = ['Accuracy', 'Precision', 'Recall', 'F1', 'Kappa']
    ax1.set_xticks([r + (barWidth * 1.5) for r in range(len(dtc_score))], )
    ax1.set_xticklabels(labels)
    ax1.set_ylabel('Score', fontweight='bold')
    ax1.set_ylim(0, 1)

    ## Create legend & title
    ax1.set_title('Evaluation Metrics', fontsize=14, fontweight='bold')
    ax1.legend()

    # Second plot
    ## Comparing ROC Curve
    ax2.plot(dtc_eval['fpr'], dtc_eval['tpr'], label='Decision Tree, auc = {:0.5f}'.format(dtc_eval['auc']))
    ax2.plot(rf_eval['fpr'], rf_eval['tpr'], label='Random Forest, auc = {:0.5f}'.format(rf_eval['auc']))
    ax2.plot(nb_eval['fpr'], nb_eval['tpr'], label='Naive Bayes, auc = {:0.5f}'.format(nb_eval['auc']))
    ax2.plot(knn_eval['fpr'], knn_eval['tpr'], label='K-Nearest Nieghbor, auc = {:0.5f}'.format(knn_eval['auc']))

    ## Configure x and y axis
    ax2.set_xlabel('False Positive Rate', fontweight='bold')
    ax2.set_ylabel('True Positive Rate', fontweight='bold')

    ## Create legend & title
    ax2.set_title('ROC Curve', fontsize=14, fontweight='bold')
    ax2.legend(loc=4)

    plt.show()

    # testing
    preds = rf.predict(X_test)
    print(pd.crosstab(y_test, preds, rownames=['Actual Result'], colnames=['Predicted Result']))
    print(list(zip(X_train, rf.feature_importances_)))
    # Saving model
    # dump(rf, 'random_baseball_fixed.joblib')
    dump(dtc, 'decision_baseball.joblib')


def test_model():
    # testing the accuracy of x number of matches
    df = pd.read_csv('baseball_test.csv')
    df.drop_duplicates(keep='first', inplace=True)
    print(f'{len(df)} records')
    # drop under 2.5 from dataFrame
    # index_names = df[ df['ou'] == 0 ].index
    # df.drop(index_names, inplace = True)
    X_train_orig = df.drop(["ou"], axis=1)
    Y_train_orig = df["ou"]
    # classification report
    from sklearn import metrics
    decision_tree_model = load('decision_baseball.joblib')
    random_forest_model = load('random_baseball_fixed.joblib')
    # Predict Test Data
    y_pred = random_forest_model.predict(X_train_orig)

    # Calculate accuracy, precision, recall, f1-score, and kappa score
    print('************Random Forest**********')
    print('***********************************')
    print(metrics.classification_report(Y_train_orig, y_pred))
    print('___________________________________')
    print('')
    print(metrics.accuracy_score(Y_train_orig, y_pred))
    print('')
    print('')


    # Calculate accuracy, precision, recall, f1-score, and kappa score
    y_pred = decision_tree_model.predict(X_train_orig)
    print('************Decision Trees*********')
    print('***********************************')
    print(metrics.classification_report(Y_train_orig, y_pred))
    print('___________________________________')
    print('')
    print(metrics.accuracy_score(Y_train_orig, y_pred))
    print('***********************************')
    print('***********************************')


def predict():
    saved_model = load('decision_baseball.joblib')
    while True:
        print('TEAM A')
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
        new = pd.DataFrame.from_dict(data)
        result = saved_model.predict(new)
        resultA = result[0]
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
        new = pd.DataFrame.from_dict(data)
        result = saved_model.predict(new)
        resultB = result[0]
        if resultA == 1:
            print('TEAM A: Over 2.5')
        else:
            print('TEAM A: No Output')
        if resultB == 1:
            print('TEAM B: Over 2.5')
        else:
            print('TEAM B: No Output')


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


def autopredict(model, dictgame):
    new = pd.DataFrame.from_dict(dictgame)
    result = model.predict(new)
    result = result[0]
    if result == 1:
        # print('Over 2.5 or 3 or more goals, 3,4,5...')
        return 'Over 2.5'
    else:
        # print('No Output')
        return 'No Output'


def future_matches():
    saved_model = load('decision_baseball.joblib')
    teams = ['angels', 'astros', 'athletics', 'blue-jays', 'braves', 'brewers', 'cardinals', 'cubs', 'diamondbacks', 'dodgers',
             'giants', 'guardians', 'mariners', 'marlins', 'mets', 'nationals', 'orioles', 'padres', 'phillies',
             'pirates', 'rangers', 'rays', 'reds', 'red-sox', 'rockies', 'royals', 'tigers', 'twins', 'white-sox', 'yankees']
    # teams = ['diamondbacks']
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
                    if game.find('td', {'class': 'td number td-number pitcher-adj'}).text[0] == '+':
                        pitcher = int(game.find('td', {'class': 'td number td-number pitcher-adj'}).text[1:])
                    else:
                        pitcher = int(game.find('td', {'class': 'td number td-number pitcher-adj'}).text[1:])*-1
                    if game.find('td', {'class': 'td number td-number travel-adj'}).text[0] == '+':
                        travel = int(game.find('td', {'class': 'td number td-number travel-adj'}).text[1:])
                    else:
                        travel = int(game.find('td', {'class': 'td number td-number travel-adj'}).text[1:])*-1
                    i = game.find('td', {'data-game-status': 'upcoming'}).text
                    record = {
                        'rating': [int(game.find('td', {'class': 'td number td-number rating'}).text)],
                        'pitcher': [pitcher],
                        'travel': [travel],
                        'adj_rating': [int(game.find('td', {'class': 'td number td-number rating-adj'}).text[1:])],
                        'win_prob': [int(game.find('td', {'class': 'td number td-number win-prob'}).text[:-1])/100],
                    }
                    # print(record)
                    print(name)
                    print(date)
                    result_team = autopredict(saved_model, record)
                    print(result_team)
                    print('__________________________________________')
                    print('')
                    list_of_lines = [name, date, result_team, '__________________________________________', '']
                    append_multiple_lines('future_games.txt', list_of_lines)
                    count += 1
                    games.append(record)
                except:
                    pass
    print(len(games))
    # print(games)
    print('Future games stored in futures_games.txt')


if __name__ == '__main__':
    # scrape_data()  # scrape baseball data and save to csv DON'T!!!!!!!
    # build_model()  # to build the model DON'T!!!!!!!
    # predict()  # take input manually to predict
    # future_matches()  # scrape matches and predict
    test_model()  # test the accuracy of n number of matches