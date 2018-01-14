
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








model = load_model('model_file.h5py')

df_test = pd.read_csv('mnist_test.csv', header = None)
X_test = df_test.drop(df_test.columns[[0]], axis = 1).values
y_test = df_test.iloc[:,0].values
y_test = to_categorical(y_test)


def num_recognize(n) :
    ans = model.predict(X_test[n].reshape(-1,784).astype('float32'))
    print(np.argmax(ans, axis = 1))
    temp = X_test[n].reshape([28,28])
    plt.imshow(temp, interpolation = 'nearest',cmap=plt.cm.gray_r)
    #plt.gray()
    plt.savefig('num_fig.pdf')




num_recognize(200)
