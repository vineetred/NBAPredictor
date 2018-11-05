import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense
import csv

model = Sequential()

model.add(Dense(units=64, activation='relu'))
model.add(Dense(units=10, activation='softmax'))

model.compile(loss='mean_squared_error',
              optimizer='sgd',
              metrics=['accuracy'])

#LOADING SHIT INTO LIST
with open('dataset/NBA_Rankings - 2016-17.csv', 'r') as f:
    reader = csv.reader(f)
    allStuff = list(reader)
offensivenumberRating = []
defensiveRating = []
rating = []
for i in range(1,31):
    offensivenumberRating.append(allStuff[i][10])
    defensiveRating.append(allStuff[i][11])
    rating.append(allStuff[i][12])

offensivenumberRating = np.array(offensivenumberRating)
defensiveRating = np.array(defensiveRating)
rating = np.array(rating)


model.fit([offensivenumberRating,defensiveRating], rating, epochs=5)

