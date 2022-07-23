import requests
import json
import lxml
from bs4 import BeautifulSoup
import pandas as pd
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler, RobustScaler
import numpy as np
import matplotlib.pyplot as plt
from joblib import dump, load
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN, SMOTETomek
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
        count = 0  #it is used to scrape the last x number of matches
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
                if int(game.find('td', {'data-game-status': 'completed'}).text) >= 4:
                    ou = 1  # 4,5...
                else:
                    ou = 0  # 0,1,2,3
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
    df.to_csv(r"baseball.csv", index=False)  # used to save the last x number of matches into a csv
    # df.to_csv(r"baseball.csv", index=False)


if __name__ == '__main__':
    # scrape_data()  # scrape baseball data and save to csv DON'T!!!!!!!
    pass