import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
from keras.layers import Input, LSTM, GRU, Dense, LayerNormalization
from keras.models import Model
from keras.optimizers import Nadam
import keras_tuner as kt
from sklearn.model_selection import train_test_split

class HyperModel(kt.HyperModel):
    def build(hp):
        """
        Paramaters to tune:
        - Number of units per LSTM layer
        - No of LSTM layers
        - No of units per feedforward hidden layer
        - No of feedforward hidden layers
        - Batch normalisation 
        - Drop out
        """
        obj_input = Input(shape=(14,8))
        lstm = LSTM(182)(obj_input)
        ln = LayerNormalization()(lstm)

        out = Dense(1, activation="sigmoid")(ln)
        model = Model(inputs=obj_input, outputs=out)
        model.compile(loss="binary_crossentropy", optimizer=Nadam(), metrics=[keras.metrics.AUC(), keras.metrics.Precision(), keras.metrics.Recall()])  
        return model
    
    def fit(self, hp, model, x, y, validation_data, callbacks=None, **kwargs):
        ...