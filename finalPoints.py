#!/usr/bin/env python
# coding: utf-8

import numpy as np
import keras
from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers import Dropout,Conv2D
from sklearn.model_selection import train_test_split
import csv
import pandas as pd
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
encoder = LabelBinarizer()

print("This is the points model")

main = pd.read_csv('dataset/main.csv')
home = pd.read_csv('dataset/main.csv',usecols=['Team'])
away = pd.read_csv('dataset/main.csv',usecols=['Opponent']).values
ha = pd.read_csv('dataset/main.csv',usecols=['Home']).values
#One Hot encoding

home = encoder.fit_transform(home)
away = encoder.fit_transform(away)
homeaway = encoder.fit_transform(ha)

team = np.append(home,away,axis=1)

#Loading stats
#Preparing Data
statistics = pd.read_csv('dataset/main.csv')
statistics = statistics[['FieldGoalsAttempted','FreeThrows','FreeThrowsAttempted','FreeThrows.','OffRebounds','TotalRebounds','Assists','Steals','Blocks','Turnovers','TotalFouls',
                        'Opp.FreeThrows', 
                        'Opp.FreeThrowsAttempted', 'Opp.FreeThrows.', 'Opp.OffRebounds', 'Opp.TotalRebounds', 
                        'Opp.Assists', 'Opp.Steals', 'Opp.Blocks', 'Opp.Turnovers', 'Opp.TotalFouls',
                        'Opp.FieldGoalsAttempted', 'Opp.FieldGoals.', 'Opp.3PointShots']].values
oneHot = np.append(team,homeaway,axis=1)
statistics = np.append(oneHot,statistics,axis=1)
#Output data
winLoss =  pd.read_csv('dataset/main.csv',usecols=['WINorLOSS']).values
winLoss = encoder.fit_transform(winLoss)
points = pd.read_csv('dataset/main.csv',usecols=['TeamPoints','OpponentPoints']).values
points = np.append(points,winLoss,axis=1)

X_train, X_test, y_train, y_test = train_test_split(statistics, points, test_size=0.20)

#MODEL
model = Sequential()
keras.optimizers.adam(lr=0.1)

model.add(Dense(30, activation='relu',input_dim=85))
model.add(Dropout(0.30))
model.add(Dense(25, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(3, activation = 'relu'))

model.compile(loss='mse',
              optimizer='adam',
              metrics=['accuracy'])

history = model.fit(np.asarray(X_train),np.asarray(y_train), epochs=100,batch_size=50)

#Testing
print(model.evaluate(X_test,y_test))
model.summary()
plt.plot(history.history['acc'])
plt.title('Model Accuracy on Training')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.show()
plt.plot(history.history['loss'])
plt.title('Model loss reduction on Training')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.show()


#CLI UI
#CAN ONLY KEYBOARD INTERUPPT!
# while(True):
# #NEED TO LOAD DATA AGAIN!
#     stats = pd.read_csv('dataset/predict.csv')
#     stats
#     home = pd.read_csv('dataset/predict.csv',usecols=['Team'])
#     away = pd.read_csv('dataset/predict.csv',usecols=['Opponent']).values
#     ha = pd.read_csv('dataset/predict.csv',usecols=['Home']).values
#     home = encoder.fit_transform(home)
#     away = encoder.fit_transform(away)
#     homeaway = encoder.fit_transform(ha)

#     t1 = input("Enter home team: ")
#     t2 = input("Enter away team: ")

#     #Finding the row where these teams match
#     location = stats.loc[(stats['Team']==t1) & (stats['Opponent']==t2)&(stats['Home']=='Home')].index
#     teams = np.append(home[location[0]],away[location[0]])
#     teams = np.append(teams,homeaway[location[0]])
#     statsPredict = stats[(stats['Team']==t1) & (stats['Opponent']==t2)&(stats['Home']=='Home')]
#     statsPredict = statsPredict[['FieldGoalsAttempted','FreeThrows','FreeThrowsAttempted','FreeThrows.','OffRebounds','TotalRebounds','Assists','Steals','Blocks','Turnovers','TotalFouls',
#                             'Opp.FreeThrows', 
#                             'Opp.FreeThrowsAttempted', 'Opp.FreeThrows.', 'Opp.OffRebounds', 'Opp.TotalRebounds', 
#                             'Opp.Assists', 'Opp.Steals', 'Opp.Blocks', 'Opp.Turnovers', 'Opp.TotalFouls',
#                             'Opp.FieldGoalsAttempted', 'Opp.FieldGoals.', 'Opp.3PointShots']].values

#     statsPredict = statsPredict.mean(axis=0)
#     statsPredict = np.transpose(statsPredict)
#     teams = np.append(teams,statsPredict)
#     #PREDICTION!!!!!
#     #PRINTING
#     prediction = model.predict(np.asarray([teams]))
#     print(prediction)
    # if(prediction>0.5):
    #     print(t1, "WINS")
    # else:
    #     print(t2,"WINS")


