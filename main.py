import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense
import csv


#LOADING SHIT INTO LIST
with open('NBA_Rankings - 2016-17.csv', 'r') as f:
    reader = csv.reader(f)
    allStuff = list(reader)
print(allStuff)
offensivenumberRating = []
defensiveRating = []
rating = []
for i in range(1,31):
    offensivenumberRating.append(allStuff[i][10])
    defensiveRating.append(allStuff[i][11])
    rating.append(allStuff[i][12])

offensivenumberRating = np.asarray(offensivenumberRating)
defensiveRating = np.asarray(defensiveRating)
rating = np.asarray(rating)
print(rating)
print(offensivenumberRating.shape)

model = Sequential()

model.add(Dense(units=64, activation='relu',input_shape=(30,)))
model.add(Dense(units=10, activation='softmax'))

model.compile(loss='mean_squared_error',
              optimizer='sgd',
              metrics=['accuracy'])



model.fit([offensivenumberRating,defensiveRating], rating, epochs=5)

