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
        lstm = LSTM(hp.Int("RNN units", 100, 200, 10))(obj_input)
        ln = LayerNormalization()(lstm)
        units = hp.Int("mlp neurons per layer", 50, 200, 25)
        dropout_rate = hp.Float("dropout rate", 0, 0.4, 0.05)
        n_layers = hp.Int("mlp layers", 1, 5, 1)
        combine_layer = hp.Int("combine position", 1, n_layers)
        event_input = Input(shape=(11,))
        prev = event_input
        for n in range(n_layers):
            layer = Dense(units, activation="selu", kernel_initializer="lecun_normal")(
                prev
            )
            if n == combine_layer - 1:
                layer = Concatenate()([ln, layer])
            prev = AlphaDropout(rate=dropout_rate)(layer)
        out = Dense(self.output_neurons, activation="sigmoid")(prev)
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
        return model.fit(*args, batch_size=hp.Choice("Batch size", [16, 32, 64, 128]), **kwargs)


def main(output_neurons=1):
    full_data = np.load("./data/full_data.npy", allow_pickle=True)
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
    # Oversample hadronics to deal with massive class imbalance
    y_train, y_test = y_train.values, y_test.values  # np arrays for ease
    hadronics_obj, hadronics_event, hadronics_labels = (
        np.repeat(X_train_obj[y_train[:, 1] == 1], 50, axis=0),
        np.repeat(X_train_event[y_train[:, 1] == 1], 50, axis=0),
        np.repeat(y_train[y_train[:, 1] == 1], 50, axis=0),
    )
    X_train_obj, X_train_event, y_train = (
        np.concatenate([X_train_obj, hadronics_obj]),
        np.concatenate([X_train_event, hadronics_event]),
        np.concatenate([y_train, hadronics_labels]),
    )
    indices = np.random.permutation(len(X_train_obj))
    X_train_obj, X_train_event, y_train = (
        X_train_obj[indices],
        X_train_event[indices],
        y_train[indices],
    )
    # weight positives more than negatives
    sample_weights = class_weight.compute_sample_weight(
        class_weight="balanced",
        y=y_train,
    )
    test_weights = class_weight.compute_sample_weight(class_weight="balanced", y=y_test)
    tuner = kt.BayesianOptimization(
        HyperModel(output_neurons),
        objective=kt.Objective("val_auc", "max"),
        max_trials=80,
        overwrite=True,
        directory=f"kt_RNN_MLP_{output_neurons}_classes",
        project_name="feb_24_tuning",
        beta=3,
    )
    callbacks = [EarlyStopping(patience=10), ReduceLROnPlateau(patience=5)]
    if output_neurons == 1:
        y_train, y_test = y_train[:, -1], y_test[:, -1]
    tuner.search(
        [X_train_obj, X_train_event],
        y_train,
        validation_data=([X_test_obj, X_test_event], y_test, test_weights),
        callbacks=callbacks,
        sample_weight=sample_weights,
        epochs=30
    )
    best_HP = tuner.get_best_hyperparameters()[0]
    print(best_HP.values)
    best_model = HyperModel(output_neurons).build(best_HP)
    best_model.save(f"./models/feb24_tuned_{output_neurons}_classes")

if __name__ == "__main__":
    if len(sys.argv) == 2:
        main(int(sys.argv[1]))
    else:
        main()
