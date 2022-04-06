import tensorflow as tf
from tensorflow import keras
from Preprocessing import PreProcess
from tensorflow.keras.layers import (
    Input,
    Dense,
    LSTM,
    AlphaDropout,
    Concatenate,
    LayerNormalization,
    GRU,
    Conv1D,
    Flatten,
    BatchNormalization,
    Conv2D,
    AveragePooling2D,
    MaxPooling2D,
)
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras import Model
from tensorflow.keras.metrics import (
    AUC,
    BinaryAccuracy,
    Precision,
    Recall,
    FalsePositives,
    TruePositives,
    CategoricalAccuracy,
)
from tensorflow.keras.regularizers import l1, l2, l1_l2
from functools import partial
from abc import ABCMeta, abstractmethod


class NN(metaclass=ABCMeta):
    """
    Base class for building different neural network architectures
    - Contains @property model attribute that checks model has been built before returning
    - Abstract method build_model implemented by subclasses, must take at least input and output shapes as kwargs
    - compile_model method takes learning rate and other keras model compilation kwargs
    """

    def __init__(self):
        self._model = None

    @property
    def model(self):
        if self._model is None:
            print("Model not built yet!")
        else:
            return self._model

    def compile_model(self, learning_rate=0.001, **kwargs):
        if self._model is None:
            print("model not built yet")
        else:
            opt = Nadam(learning_rate=learning_rate)
            self._model.compile(optimizer=opt, **kwargs)

    @abstractmethod
    def build_model(self, input_shape=None, n_outputs=None, **kwargs):
        pass


class MLP(NN):
    """
    Multi layer perceptron model to take only event level inputs
    """

    def build_model(
        self, n_inputs=10, n_outputs=1, n_layers=3, n_neurons=300, dropout=0.2
    ):
        inp = Input(shape=n_inputs)
        prev = inp
        for _ in range(n_neurons):
            prev = Dense(
                n_neurons, activation="selu", kernel_initializer="lecun_normal"
            )(prev)
            if dropout > 0:
                prev = AlphaDropout(rate=dropout)(prev)
        if n_outputs == 1:
            out = Dense(n_outputs, activation="sigmoid")(prev)
        else:
            out = Dense(n_outputs, activation="softmax")
        self._model = Model(inputs=inp, outputs=out)


class RNN(NN):
    """
    RNN model to take only object level inputs (expects jets fed in sequence ordered by pt)
    - RNN cell of choice is GRU which was found to have best performance
    - Recurrent dropout adds a small performance boost but use with care as it is not supported by cudnn so will cripple performance on gpus (can mitigate by increasing batch size by ~8x)
    """

    def build_model(
        self,
        input_shape=[],
        n_outputs=1,
        rnn_layers=3,
        rnn_neurons=200,
        recurrent_dropout=0,
        dense_layers=0,
        dense_neurons=40,
        dropout=0.5,
    ):
        inp = Input(shape=input_shape)
        prev = inp
        if rnn_layers == 1:
            prev = GRU(rnn_neurons, recurrent_dropout=recurrent_dropout)(prev)
        else:
            for _ in range(rnn_layers - 1):
                prev = GRU(
                    rnn_neurons,
                    recurrent_dropout=recurrent_dropout,
                    return_sequences=True,
                )(prev)
        prev = GRU(rnn_neurons, recurrent_dropout=recurrent_dropout)(prev)
        for _ in range(dense_layers):
            if dropout > 0:
                prev = AlphaDropout(rate=dropout)(prev)
            prev = Dense(
                dense_neurons, activation="selu", kernel_initializer="lecun_normal"
            )(prev)
        if n_outputs == 1:
            out = Dense(n_outputs, activation="sigmoid")(prev)
        else:
            out = Dense(n_outputs, activation="softmax")
        self._model = Model(inputs=inp, outputs=out)


class RNNCombined(NN):
    """
    RNN model to take both object level inputs (expects jets fed in sequence ordered by pt) and event level tabular inputs
    - RNN cell of choice is GRU which was found to have best performance
    - Recurrent dropout adds a small performance boost but use with care as it is not supported by cudnn so will cripple performance on gpus (can mitigate by increasing batch size by ~8x)

    """

    def build_model(
        self,
        input_shape=[],
        n_outputs=1,
        rnn_layers=3,
        rnn_neurons=200,
        recurrent_dropout=0,
        dense_layers=3,
        dense_neurons=40,
        dropout=0.5,
    ):
        obj_inp = Input(shape=input_shape[0])
        prev = obj_inp
        if rnn_layers == 1:
            prev = GRU(rnn_neurons, recurrent_dropout=recurrent_dropout)(prev)
        else:
            for _ in range(rnn_layers - 1):
                prev = GRU(
                    rnn_neurons,
                    recurrent_dropout=recurrent_dropout,
                    return_sequences=True,
                )(prev)
        prev = GRU(rnn_neurons, recurrent_dropout=recurrent_dropout)(prev)
        event_inp = Input(shape=input_shape[1])
        dense = Dense(
            rnn_neurons, activation="selu", kernel_initializer="lecun_normal"
        )(event_inp)
        conc = Concatenate()([prev, dense])
        prev = conc
        for _ in range(dense_layers - 1):
            if dropout > 0:
                prev = AlphaDropout(rate=dropout)(prev)
            prev = Dense(
                dense_neurons, activation="selu", kernel_initializer="lecun_normal"
            )(prev)
        if n_outputs == 1:
            out = Dense(n_outputs, activation="sigmoid")(prev)
        else:
            out = Dense(n_outputs, activation="softmax")
        self._model = Model(inputs=[obj_inp, event_inp], outputs=out)


class CNN(NN):
    """
    CNN model expects only jet image input (no event level features)
    - Eta-phi plane of ~ 40x40 pixels was found to work well
    - Rotating events to centre the leading jet at phi=0 improves performance
    """

    def build_model(
        self,
        input_shape=None,
        n_outputs=None,
        conv_layers=2,
        pooling="max",
        dense_layers=2,
        dense_neurons=200,
        dropout=0.5,
        **conv_kwargs
    ):
        inp = Input(shape=input_shape)
        prev = Conv2D(**conv_kwargs)(inp)
        for _ in range(conv_layers - 1):
            prev = Conv2D(**conv_kwargs)(prev)
        if pooling == "max":
            pool = MaxPooling2D(pool_size=(2, 2))(prev)
        else:
            pool = AveragePooling2D(pool_size=(2, 2))(prev)
        flat = Flatten()(pool)
        prev = flat
        for _ in range(dense_layers):
            if dropout > 0:
                prev = AlphaDropout(rate=dropout)(prev)
            prev = Dense(
                dense_neurons, activation="selu", kernel_initializer="lecun_normal"
            )(prev)
        if n_outputs == 1:
            out = Dense(n_outputs, activation="sigmoid")(prev)
        else:
            out = Dense(n_outputs, activation="softmax")
        self._model = Model(inputs=inp, outputs=out)
