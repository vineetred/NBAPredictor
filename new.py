import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers import Dropout

import csv
import pandas as pd
import matplotlib.pyplot as plt
from keras.preprocessing.text import Tokenizer
tok = Tokenizer()

#LOADING DATA USING PANDAS
teams = ['ATL','BOS','BRK', 'CHO', 'CHI', 'CLE', 'DAL', 'DEN', 'DET', 'GSW', 'HOU','IND','LAC','LAL','MEM', 'MIA','MIL','MIN','NOP','NYK','OKC','ORL','PHI','PHO','POR','SAC','SAS','TOR','UTA','WAS']
# tok.fit_on_texts(teams)
# tok.texts_to_sequences(teams)
points = pd.read_csv('nba_games_stats.csv',usecols=['TeamPoints','OpponentPoints','WINLOSS']).values
teamStats = pd.read_csv('nba_games_stats.csv',usecols=['Team','Home','Opponent']).values

#Making the neural network
model = Sequential()
model.add(Dense(60, activation='relu',input_dim=3))
model.add(Dropout(0.25))
model.add(Dense(36, activation='relu'))
model.add(Dense(3, activation='relu'))

model.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['mse'])

model.fit(np.asarray(teamStats), np.asarray(points), epochs=1000)

hello = np.array([[15,1,1]])
prediction = model.predict(hello)
# print(teams[hello[0][0]])
if(prediction[0][1]>prediction[0][2]):
  print(teams[hello[0][0]],"wins against",teams[hello[0][2]])
  print("Score: ",prediction[0][1], "-",prediction[0][2])

else:
  print(teams[hello[0][2]], " wins against", teams[hello[0][0]])
  print("Score: ",prediction[0][2], "-",prediction[0][1])