from __future__ import print_function
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
import numpy as np
import random
import sys

class RNNKeras:
    
    def __init__(self, sentenceLen, vector_size, output_size, hidden_dim=100):
        # Assign instance variables
        self.sentenceLen = sentenceLen
        self.vector_size = vector_size
        self.output_size = output_size
        self.hidden_dim = hidden_dim

        self.__model_build__()
    
    def __model_build__(self):
        self.model = Sequential()
        self.model.add(LSTM(self.output_size, input_shape=(self.sentenceLen, self.vector_size)))
        self.model.add(Dense(self.vector_size)))
        self.model.add(Activation('softmax'))

        optimizer = RMSprop(lr=0.01)
        self.model.compile(loss='categorical_crossentropy', optimizer=optimizer)
    
    def train_model(self, X, y, batchSize=128, nepoch=1):
        self.model.fit(X, y, batch_size=batchSize, nb_epoch=nepoch)

    def predict(self, x):
        return model.predict(x, verbose=0)[0]