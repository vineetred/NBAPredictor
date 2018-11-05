import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense
import csv
import pandas as pd



#LOADING DATA USING PANDAS
rating = pd.read_csv('NBA_Rankings - 2017-18.csv',usecols=['Rating']).values/	110.25
offensiveRating = pd.read_csv('NBA_Rankings - 2017-18.csv',usecols=['Offensive Rating','Defensive Rating']).values/110.25


print(offensiveRating)

#NEURAL NETWORK
model = Sequential()

model.add(Dense(64, activation='relu',input_dim=2))
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='relu'))

model.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['mape'])



model.fit(np.asarray(offensiveRating), np.asarray(rating), epochs=500)

#PREDICTION!
rating_18 = pd.read_csv('NBA_Rankings - 2016-17.csv',usecols=['Rating']).values/110.25
offensiveRating_18 = pd.read_csv('NBA_Rankings - 2016-17.csv',usecols=['Offensive Rating','Defensive Rating']).values/110.25
print(model.predict(offensiveRating_18)*110.25)
model.layers[1].output

