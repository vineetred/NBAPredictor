import numpy as np
import keras
from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers import Dropout,Conv2D
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import csv
import pandas as pd
import matplotlib.pyplot as plt


#LOADING DATA USING PANDAS
teams = ['ATL','BOS','BRK', 'CHO', 'CHI', 'CLE', 'DAL', 'DEN', 'DET', 'GSW', 'HOU','IND','LAC','LAL','MEM', 'MIA','MIL','MIN','NOP','NYK','OKC','ORL','PHI','PHO','POR','SAC','SAS','TOR','UTA','WAS']

points = pd.read_csv('dataset/nba_games_stats.csv',usecols=['WINLOSS']).values

#LABELS
home = pd.read_csv('dataset/nba_games_stats.csv',usecols=['Team']).values
home = to_categorical(home)

away = pd.read_csv('dataset/nba_games_stats.csv',usecols=['Opponent']).values
homeaway = pd.read_csv('dataset/nba_games_stats.csv',usecols=['Home']).values
new = np.append(home,away,axis=1)
new2 = np.append(new,homeaway,axis=1)
new2.shape
X_train, X_test, y_train, y_test = train_test_split(new2, points, test_size=0.23)
#Making the neural network
model = Sequential()
keras.optimizers.adam(lr=0.1)
model.add(Dense(60, activation='relu',input_dim=33))
model.add(Dropout(0.25))
# model.add(Dense(46, activation='relu'))
# model.add(Dense(30, activation='relu'))
# model.add(Dense(20, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(20, activation='relu'))



#CNN

model.add(Dense(1, activation = 'relu'))

model.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(np.asarray(X_train),np.asarray(y_train), epochs=50)
model.evaluate(X_test,y_test)