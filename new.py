import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense
import csv
import pandas as pd
import matplotlib.pyplot as plt
from keras.preprocessing.text import Tokenizer
tok = Tokenizer()

#LOADING DATA USING PANDAS
teams = ['ATL','BOS','BRK', 'CHO', 'CHI', 'CLE', 'DAL', 'DEN', 'DET', 'GSW', 'HOU','IND','LAC','LAL','MEM', 'MIA','MIL','MIN','NOP','NYK','OKC','ORL','PHI','PHO','POR','SAC','SAS','TOR','UTA','WAS']
# tok.fit_on_texts(teams)
# tok.texts_to_sequences(teams)
points = pd.read_csv('dataset/nba_games_stats.csv',usecols=['TeamPoints','OpponentPoints']).values
teamStats = pd.read_csv('dataset/nba_games_stats.csv',usecols=['Team','Home','Opponent']).values

#Making the neural network
model = Sequential()
model.add(Dense(30, activation='relu',input_dim=3))
# model.add(Dense(64, activation='relu'))
model.add(Dense(24, activation='relu'))
model.add(Dense(2, activation='relu'))

model.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['mape'])

model.fit(np.asarray(teamStats), np.asarray(points), epochs=10)
predict = np.array(['2','1','5'])
print(model.predict(predict))
