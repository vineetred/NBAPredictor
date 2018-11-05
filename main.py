import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense
import csv
import pandas as pd



#LOADING SHIT INTO LIST
# with open('NBA_Rankings - 2016-17.csv', 'r') as f:
#     reader = csv.reader(f)
#     allStuff = list(reader)
# print(allStuff)
# offensivenumberRating = []
# defensiveRating = []
# rating = []
# for i in range(1,31):
#     offensivenumberRating.append(allStuff[i][10])
#     defensiveRating.append(allStuff[i][11])
#     rating.append(float(allStuff[i][12]))
# print(rating)
# offensivenumberRating = np.asarray(offensivenumberRating)
# defensiveRating = np.asarray(defensiveRating)
# rating = np.asarray(rating)
# print(rating)
# print(offensivenumberRating.shape)



#SAME LOADING USING PANDAS
rating = pd.read_csv('NBA_Rankings - 2016-17.csv',usecols=['Rating']).values
offensiveRating = pd.read_csv('NBA_Rankings - 2016-17.csv',usecols=['Offensive Rating','Defensive Rating']).values/110
defensiveRating = pd.read_csv('NBA_Rankings - 2016-17.csv',usecols=['Defensive Rating']).values
#     allStuff = list(reader)



rating = np.array([i for [i] in rating])

print(rating)
model = Sequential()

model.add(Dense(20, activation='relu',input_dim=2))
model.add(Dense(20, activation='relu'))
model.add(Dense(1, activation='relu'))

model.compile(loss='mean_squared_error',
              optimizer='sgd',
              metrics=['accuracy'])



model.fit(np.asarray(offensiveRating), np.asarray(rating), epochs=500)

