import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras

from keras.layers import Dense
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
from keras.models import load_model
from keras.utils import to_categorical
from numpy import argmax


df = pd.read_csv('mnist_train.csv', header = None)
print(df.shape)
X_train = df.drop(df.columns[[0]],axis=1).values
print(X_train.shape)
y_train = df.iloc[:,0].values
y_train = to_categorical(y_train)
print(y_train.shape)




num_pixels = 784
X_train = X_train.reshape(X_train.shape[0], num_pixels).astype('float32')
X_train = X_train / 255





early_stopping_monitor = EarlyStopping(patience = 3)
n_cols = X_train.shape[1]
model = Sequential()
model.add(Dense(num_pixels,activation = 'relu',input_shape = (n_cols,)))
model.add(Dense(10, activation = 'softmax'))
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
model.fit(X_train,y_train,epochs = 1, callbacks=[early_stopping_monitor])

print(y_train)

df_test = pd.read_csv('mnist_test.csv', header = None)
X_test = df_test.drop(df_test.columns[[0]], axis = 1).values
y_test = df_test.iloc[:,0].values
y_test = to_categorical(y_test)

y_preds = model.predict(X_test)
print(np.argmax(y_preds, axis = 1))




model.save('model_file.h5py')




