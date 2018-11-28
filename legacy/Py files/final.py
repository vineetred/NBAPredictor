#IMPORTING LIBRARIES
import numpy as np
import keras
from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers import Dropout
from sklearn.model_selection import train_test_split
import csv
import pandas as pd
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer

#CREATING ENCODER OBJECT
encoder = LabelBinarizer()
#LOADING DATASETS USING PANDAS
# main = pd.read_csv('dataset/nba.games.stats.csv')
home = pd.read_csv('dataset/main.csv',usecols=['Team'])
away = pd.read_csv('dataset/main.csv',usecols=['Opponent']).values
ha = pd.read_csv('dataset/main.csv',usecols=['Home']).values
#ONE HOT ENCODING THE DATASET
home = encoder.fit_transform(home)
away = encoder.fit_transform(away)
homeaway = encoder.fit_transform(ha)
#JOINING THE COLUMNS ON THE ENCODED COLUMNS
teamCol = np.append(home,away,axis=1)
# new2 = pd.read_csv('dataset/main.csv',usecols=['FieldGoals','FieldGoalsAttempted','FieldGoals.','X3PointShots','X3PointShotsAttempted','X3PointShots.','FreeThrows','FreeThrowsAttempted','FreeThrows.','OffRebounds','TotalRebounds','Assists','Steals','Blocks','Turnovers','TotalFouls',
#                         'Opp.FieldGoals', 'Opp.3PointShotsAttempted', 'Opp.3PointShots.', 'Opp.FreeThrows', 
#                         'Opp.FreeThrowsAttempted', 'Opp.FreeThrows.', 'Opp.OffRebounds', 'Opp.TotalRebounds', 
#                         'Opp.Assists', 'Opp.Steals', 'Opp.Blocks', 'Opp.Turnovers', 'Opp.TotalFouls',
#                         'Opp.FieldGoalsAttempted', 'Opp.FieldGoals.', 'Opp.3PointShots']).values
#LOADING STATS INTO A DF
teamStats = pd.read_csv('dataset/main.csv',usecols=['FieldGoalsAttempted','FreeThrows','FreeThrowsAttempted','FreeThrows.','OffRebounds','TotalRebounds','Assists','Steals','Blocks','Turnovers','TotalFouls',
                        'Opp.FreeThrows', 
                        'Opp.FreeThrowsAttempted', 'Opp.FreeThrows.', 'Opp.OffRebounds', 'Opp.TotalRebounds', 
                        'Opp.Assists', 'Opp.Steals', 'Opp.Blocks', 'Opp.Turnovers', 'Opp.TotalFouls',
                        'Opp.FieldGoalsAttempted', 'Opp.FieldGoals.']).values

#JOINING COLUMNS
teamColLoc = np.append(teamCol,homeaway,axis=1)
stats = np.append(teamColLoc,teamStats,axis=1)
winLoss =  pd.read_csv('dataset/main.csv',usecols=['WINorLOSS']).values
winLoss = encoder.fit_transform(winLoss)
points = pd.read_csv('dataset/main.csv',usecols=['TeamPoints','OpponentPoints']).values
points = np.append(points,winLoss,axis=1)
# points.shape
# points = winLoss
X_train, X_test, y_train, y_test = train_test_split(stats, winLoss, test_size=0.0001)

#MODEL
model = Sequential()
keras.optimizers.adam(lr=0.1)
model.add(Dense(60, activation='relu',input_dim=84))
model.add(Dropout(0.30))
model.add(Dense(25, activation='relu'))
model.add(Dropout(0.30))
model.add(Dense(10, activation='relu'))
model.add(Dropout(0.30))
model.add(Dense(1, activation = 'sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

#FITING
history = model.fit(np.asarray(X_train),np.asarray(y_train), epochs=50,batch_size=300)
print(model.evaluate(X_test,y_test))
predict = np.asarray([stats[674]])

#TESTING
mainTest = pd.read_csv('dataset/main copy.csv')
home = pd.read_csv('dataset/main copy.csv',usecols=['Team'])
away = pd.read_csv('dataset/main copy.csv',usecols=['Opponent']).values
ha = pd.read_csv('dataset/main copy.csv',usecols=['Home']).values
home = encoder.fit_transform(home)
away = encoder.fit_transform(away)
homeaway = encoder.fit_transform(ha)

testTeamCol = np.append(home,away,axis=1)
finalStats = pd.read_csv('dataset/main copy.csv',usecols=['FieldGoalsAttempted','FreeThrows','FreeThrowsAttempted','FreeThrows.','OffRebounds','TotalRebounds','Assists','Steals','Blocks','Turnovers','TotalFouls',
                        'Opp.FreeThrows', 
                        'Opp.FreeThrowsAttempted', 'Opp.FreeThrows.', 'Opp.OffRebounds', 'Opp.TotalRebounds', 
                        'Opp.Assists', 'Opp.Steals', 'Opp.Blocks', 'Opp.Turnovers', 'Opp.TotalFouls',
                        'Opp.FieldGoalsAttempted', 'Opp.FieldGoals.']).values

testTeam = np.append(testTeamCol,homeaway,axis=1)
finalStats = np.append(testTeam,finalStats,axis=1)
winLoss =  pd.read_csv('dataset/main copy.csv',usecols=['WINorLOSS']).values
winLoss = encoder.fit_transform(winLoss)
points = pd.read_csv('dataset/main copy.csv',usecols=['TeamPoints','OpponentPoints']).values
points = np.append(points,winLoss,axis=1)

#TESTING EVAL ON 2018 DATASET
print("2018 Dataset")
scores = model.evaluate(finalStats,winLoss)
print(scores)
model.summary()
model.predict(predict)

#RANDOM TEST
predict = np.asarray([finalStats[104]])
print(finalStats[104])
print(model.predict(predict))
# PREDICTION ANALYSIS

winLoss =  pd.read_csv('dataset/main copy.csv',usecols=['WINorLOSS']).values
for i in range(0,len(finalStats)):
    predict = np.asarray([finalStats[i]])
    hello = model.predict(predict)
    if(hello>0.5):
       print("Game No: ",i,"Home: ",mainTest['Team'][i],"Away: ",mainTest['Opponent'][i],"Prediction: W"," Actual: ",winLoss[i],"ScoreLine: ",mainTest['TeamPoints'][i],"-",mainTest['OpponentPoints'][i])
       
    else:
        print("Game No: ",i,"Home: ",mainTest['Team'][i],"Away: ",mainTest['Opponent'][i],"Prediction: L"," Actual: ",winLoss[i],"ScoreLine: ",mainTest['TeamPoints'][i],"-",mainTest['OpponentPoints'][i])
#TESTING!!!! DEVLOPENET VERY BROKEN!
