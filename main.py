import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense
import csv
import pandas as pd
import matplotlib.pyplot as plt




#LOADING DATA USING PANDAS
rating = pd.read_csv('dataset/NBA_Rankings - 2016-17.csv',usecols=['Rating']).values
offensiveRating = pd.read_csv('dataset/NBA_Rankings - 2016-17.csv',usecols=['Pace','Assist Ratio','Turnover Ratio','Offensive Rebound Ratio','Defensive Rebound Ratio','Rebound Rate','Effective Field Goal Percentage','Shooting Percentage','Offensive Rating','Defensive Rating','Rating']).values


print(offensiveRating)

#NEURAL NETWORK
model = Sequential()

model.add(Dense(24, activation='relu',input_dim=11))
model.add(Dense(34, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['mape'])



history = model.fit(np.asarray(offensiveRating), np.asarray(rating), epochs=1000)

#PREDICTION!
rating_18 = pd.read_csv('dataset/NBA_Rankings - 2017-18.csv',usecols=['Rating']).values
offensiveRating_18 = pd.read_csv('dataset/NBA_Rankings - 2017-18.csv',usecols=['Pace','Assist Ratio','Turnover Ratio','Offensive Rebound Ratio','Defensive Rebound Ratio','Rebound Rate','Effective Field Goal Percentage','Shooting Percentage','Offensive Rating','Defensive Rating','Rating']).values
print("2017-18")
print(model.predict(offensiveRating_18)-rating_18)
# print(model.predict(offensiveRating_18)*110.25)
rating_19 = pd.read_csv('dataset/NBA_Rankings - 2018-19.csv',usecols=['Rating']).values
teamName = pd.read_csv('dataset/NBA_Rankings - 2018-19.csv',usecols=['TEAM'])
print(teamName)
offensiveRating_19 = pd.read_csv('dataset/NBA_Rankings - 2018-19.csv',usecols=['Pace','Assist Ratio','Turnover Ratio','Offensive Rebound Ratio','Defensive Rebound Ratio','Rebound Rate','Effective Field Goal Percentage','Shooting Percentage','Offensive Rating','Defensive Rating','Rating']).values
print("2018-19")
print(model.predict(offensiveRating_19)-rating_19,teamName)
plt.plot(history.history['loss'])
plt.show()
