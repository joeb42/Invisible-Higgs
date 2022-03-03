import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
from keras.layers import (
    Input,
    LSTM,
    GRU,
    Dense,
    LayerNormalization,
    AlphaDropout,
    Concatenate,
)
from keras.models import Model
from keras.optimizers import Nadam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.metrics import AUC
import keras_tuner as kt
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
import sys
import argparse



class HyperModel(kt.HyperModel):
    def __init__(self, output_neurons=1):
        self.output_neurons = output_neurons

    def build(self, hp):
        """
        Paramaters to tune:
        - Number of units per LSTM layer
        - No of LSTM layers
        - No of units per feedforward hidden layer
        - No of feedforward hidden layers
        - Batch normalisation
        - Drop out
        """
        obj_input = Input(shape=(14, 8))
        gru = GRU(hp.Int("RNN units", 100, 300, 25))(obj_input)
        ln = LayerNormalization()(gru)
        units = hp.Int("mlp neurons per layer", 100, 400, 50)
        dropout_rate = hp.Float("dropout rate", 0, 0.4, 0.05)
        n_layers = hp.Int("mlp layers", 1, 10, 1)
        combine_layer = hp.Int("combine position", 1, n_layers)
        event_input = Input(shape=(10,))
        prev = event_input
        for n in range(n_layers):
            layer = Dense(units, activation="selu", kernel_initializer="lecun_normal")(
                prev
            )
            if n == combine_layer - 1:
                layer = Concatenate()([ln, layer])
            prev = AlphaDropout(rate=dropout_rate)(layer)
        if self.output_neurons == 1:
            act = "sigmoid"
        else:
            act = "softmax"
        out = Dense(self.output_neurons, activation=act)(prev)
        model = Model(inputs=[obj_input, event_input], outputs=out)
        lr = hp.Float("learning rate", 0.00001, 0.001, 0.000005)
        beta_1 = hp.Float("beta 1", 0.9, 0.999, 0.001)
        beta_2 = 0.999
        if self.output_neurons == 1:
            loss = "binary_crossentropy"
        else:
            loss = "categorical_crossentropy"
        model.compile(
            loss=loss,
            optimizer=Nadam(learning_rate=lr, beta_1=beta_1, beta_2=beta_2),
            metrics=[
                keras.metrics.AUC(),
                keras.metrics.Precision(),
                keras.metrics.Recall(),
                keras.metrics.BinaryAccuracy(),
            ],
        )
        return model

    def fit(self, hp, model, *args, **kwargs):
        return model.fit(*args, **kwargs)


def main(output_neurons=1):
    full_data = np.load("./data/combined_data.npy", allow_pickle=True)
    X_event, X_obj, y = full_data[0], full_data[2], full_data[-2]
    (
        X_train_event,
        X_test_event,
        X_train_obj,
        X_test_obj,
        y_train,
        y_test,
    ) = train_test_split(
        X_event, X_obj, y, test_size=0.15, random_state=42, shuffle=True, stratify=y
    )
    # Imbalanced dataset so want to adjusts weights of signal and background training examples
    y_train, y_test = y_train.values, y_test.values  # np arrays for ease
    if output_neurons == 1:
        y_train, y_test = y_train[:,-1], y_test[:,-1]
        class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
        d_class_weights = dict(enumerate(class_weights))
        train_weights = np.array([d_class_weights[i] for i in y_train])
        test_weights = np.array([d_class_weights[i] for i in y_test])
    else:
        y_integers = np.argmax(y_train, axis=1)
        class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y_integers), y=y_integers)
        d_class_weights = dict(enumerate(class_weights))
        train_weights = np.array([d_class_weights[i] for i in np.argmax(y_train, axis=1)])
        test_weights = np.array([d_class_weights[i] for i in np.argmax(y_test, axis=1)])
    # Bayesian opt
    tuner = kt.BayesianOptimization(
        HyperModel(output_neurons),
        objective='val_loss',
        max_trials=100,
        overwrite=True,
        directory=f"kt_RNN_MLP_{output_neurons}_classes",
        project_name="mar_1_tuning",
        beta=3,
    )
    # Reduce learning rate if val loss start to increase + stop early if it doesn't decrease after 15 epochs
    callbacks = [EarlyStopping(patience=15), ReduceLROnPlateau(patience=1)]
    tuner.search(
        [X_train_obj, X_train_event],
        y_train,
        validation_data=([X_test_obj, X_test_event], y_test, test_weights),
        callbacks=callbacks,
        sample_weight=train_weights,
        epochs=50,
        batch_size=64
    )
    best_HP = tuner.get_best_hyperparameters()[0]
    print(best_HP.values)
    best_model = HyperModel(output_neurons).build(best_HP)
    best_model.save(f"./models/mar1_tuned_{output_neurons}_classes")

if __name__ == "__main__":
    # gpu setup
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
    # Restrict TensorFlow to only use the second GPU
        try:
            tf.config.set_visible_devices(gpus[1], 'GPU')
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
        except RuntimeError as e:
            # Visible devices must be set before GPUs have been initialized
            print(e)

    if len(sys.argv) == 2:
        main(int(sys.argv[1]))
    else:
        main()
