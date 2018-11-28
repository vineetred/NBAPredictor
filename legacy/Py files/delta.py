
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

home = pd.read_csv('dataset/nba.games.stats - nba_prev.csv',usecols=['Team'])
away = pd.read_csv('dataset/nba.games.stats - nba_prev.csv',usecols=['Opponent']).values
ha = pd.read_csv('dataset/nba.games.stats - nba_prev.csv',usecols=['Home']).values

home = encoder.fit_transform(home)
away = encoder.fit_transform(away)
homeaway = encoder.fit_transform(ha)

new = np.append(home,away,axis=1)
# new2 = pd.read_csv('dataset/nba.games.stats - nba_prev.csv',usecols=['FieldGoals','FieldGoalsAttempted','FieldGoals.','X3PointShots','X3PointShotsAttempted','X3PointShots.','FreeThrows','FreeThrowsAttempted','FreeThrows.','OffRebounds','TotalRebounds','Assists','Steals','Blocks','Turnovers','TotalFouls',
#                         'Opp.FieldGoals', 'Opp.3PointShotsAttempted', 'Opp.3PointShots.', 'Opp.FreeThrows', 
#                         'Opp.FreeThrowsAttempted', 'Opp.FreeThrows.', 'Opp.OffRebounds', 'Opp.TotalRebounds', 
#                         'Opp.Assists', 'Opp.Steals', 'Opp.Blocks', 'Opp.Turnovers', 'Opp.TotalFouls',
#                         'Opp.FieldGoalsAttempted', 'Opp.FieldGoals.', 'Opp.3PointShots']).values

new2 = pd.read_csv('dataset/nba.games.stats - nba_prev.csv',usecols=['FieldGoalsAttempted','FreeThrows','FreeThrowsAttempted','FreeThrows.','OffRebounds','TotalRebounds','Assists','Steals','Blocks','Turnovers','TotalFouls',
                        'Opp.FreeThrows', 
                        'Opp.FreeThrowsAttempted', 'Opp.FreeThrows.', 'Opp.OffRebounds', 'Opp.TotalRebounds', 
                        'Opp.Assists', 'Opp.Steals', 'Opp.Blocks', 'Opp.Turnovers', 'Opp.TotalFouls',
                        'Opp.FieldGoalsAttempted', 'Opp.FieldGoals.', 'Opp.3PointShots']).values


new3 = np.append(new,homeaway,axis=1)
new2 = np.append(new3,new2,axis=1)
winLoss =  pd.read_csv('dataset/nba.games.stats - nba_prev.csv',usecols=['WINorLOSS']).values
winLoss = encoder.fit_transform(winLoss)
points = pd.read_csv('dataset/nba.games.stats - nba_prev.csv',usecols=['TeamPoints','OpponentPoints']).values
points = np.append(points,winLoss,axis=1)
# points.shape
# points = winLoss
X_train, X_test, y_train, y_test = train_test_split(new2, winLoss, test_size=0.40)

#MODEL
model = Sequential()
keras.optimizers.adam(lr=0.1)
model.add(Dense(30, activation='relu',input_dim=85))
model.add(Dropout(0.30))
# model.add(Dense(46, activation='relu'))
# model.add(Dense(30, activation='sigmoid'))
model.add(Dense(25, activation='relu'))
model.add(Dense(10, activation='relu'))

# model.add(Dense(20, activation='relu'))
model.add(Dense(1, activation = 'sigmoid'))

model.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['accuracy'])

history = model.fit(np.asarray(X_train),np.asarray(y_train), epochs=40)
print(model.evaluate(X_test,y_test))



home = pd.read_csv('dataset/nba.games.stats - nba_2018.csv',usecols=['Team'])
away = pd.read_csv('dataset/nba.games.stats - nba_2018.csv',usecols=['Opponent']).values
print("HOME: ", home[5],"AWAY: ", away[5])
ha = pd.read_csv('dataset/nba.games.stats - nba_2018.csv',usecols=['Home']).values
home = encoder.fit_transform(home)
away = encoder.fit_transform(away)
homeaway = encoder.fit_transform(ha)



new = np.append(home,away,axis=1)
new2 = pd.read_csv('dataset/nba.games.stats - nba_2018.csv',usecols=['FieldGoalsAttempted','FreeThrows','FreeThrowsAttempted','FreeThrows.','OffRebounds','TotalRebounds','Assists','Steals','Blocks','Turnovers','TotalFouls',
                        'Opp.FreeThrows', 
                        'Opp.FreeThrowsAttempted', 'Opp.FreeThrows.', 'Opp.OffRebounds', 'Opp.TotalRebounds', 
                        'Opp.Assists', 'Opp.Steals', 'Opp.Blocks', 'Opp.Turnovers', 'Opp.TotalFouls',
                        'Opp.FieldGoalsAttempted', 'Opp.FieldGoals.', 'Opp.3PointShots']).values




new3 = np.append(new,homeaway,axis=1)
new2 = np.append(new3,new2,axis=1)
winLoss =  pd.read_csv('dataset/nba.games.stats - nba_2018.csv',usecols=['WINorLOSS']).values
winLoss = encoder.fit_transform(winLoss)
points = pd.read_csv('dataset/nba.games.stats - nba_2018.csv',usecols=['TeamPoints','OpponentPoints']).values
points = np.append(points,winLoss,axis=1)
# points.shape
# points = winLoss
# X_train, X_test, y_train, y_test = train_test_split(new2, points, test_size=0.40)
scores = model.evaluate(new2,winLoss)
print(scores)
model.summary()
# print(scores)

# predict = np.asarray([new2[5]])
# hello = model.predict(predict)
# print(hello[0][0], "-",hello[0][1])
print(new2[5])
predict = np.asarray([new2[5]])
print(model.predict(predict))
plt.plot(history.history['loss'])
plt.show()