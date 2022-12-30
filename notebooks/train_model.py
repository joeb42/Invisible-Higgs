import numpy as np
import tensorflow as tf
from tensorflow import keras
from utilities.Preprocessing import Data
from utilities.DNN import *
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.metrics import AUC
from sklearn.model_selection import train_test_split
import sklearn.utils 
import matplotlib.pyplot as plt
from utilities.Analysis import Evaluate
import os
import time
import argparse


def main(model_type):
    # data = np.load(f"./data/{data_path}", allow_pickle=True)
    data = Data(path="/usersc/ac18804/new_data/ml_vars/", seed=5, exclude=[])
    X_train_obj, X_test_obj = data.sequential()
    X_train_image, X_test_image = data.image()
    X_train_event, X_test_event = data.event()
    y_train, y_test = data.labels()
    train_xs_weights, test_xs_weights = data.xs_weight()
    (
        X_train_obj,
        X_valid_obj,
        X_train_image,
        X_valid_image,
        X_train_event,
        X_valid_event,
        y_train,
        y_valid,
    ) = train_test_split(
        X_train_obj,
        X_train_image,
        X_train_event,
        y_train,
        test_size=0.1,
        random_state=42,
        stratify=y_train,
    )
    run_logdir = get_run_logdir()
    TB = TensorBoard(run_logdir)

    callbacks = [
        EarlyStopping(patience=10, monitor="val_auc", restore_best_weights=True, mode="max"),
        ReduceLROnPlateau(patience=3, monitor="val_auc", mode="max"),
        TB,
    ]
    tot = len(y_train)
    pos = np.sum(y_train["ttH125"])
    neg = tot - pos
    # weight positives more than negatives
    weight_for_0 = (1 / neg) * (tot / 2.0)
    weight_for_1 = (1 / pos) * (tot / 2.0)
    class_weight = {0: weight_for_0, 1: weight_for_1}
    sample_weights = np.array([class_weight[i] for i in y_valid.values[:, -1]])
    if model_type.lower() == "combined":
        model = RNNCombined()
        model.build_model(
            input_shape=[X_train_obj.shape[1:], X_train_event.shape[1:]],
            n_outputs=1,
            rnn_layers=1,
            dense_layers=2,
            recurrent_dropout=0.5,
            dense_neurons=200,
            dropout=0.2,
        )
        model.compile(
            learning_rate=0.0005, loss="binary_crossentropy", metrics=AUC(name="auc")
        )
        print(model.summary())
        # train model
        history = model.fit(
            [X_train_obj, X_train_event],
            y_train.values[:, -1],
            validation_data=(
                [X_valid_obj, X_valid_event],
                y_valid.values[:, -1],
                sample_weights,
            ),
            epochs=80,
            shuffle=True,
            batch_size=256,
            callbacks=callbacks,
            class_weight=class_weight,
        )
        path = "./models/trained_models/final_combined_new_data"
        model.model.save(path)
        analyser = Evaluate(
            model,
            [X_test_obj, X_test_event],
            y_test.values[:, -1],
            test_xs_weights,
            path=path + "/",
        )
    elif model_type.lower() == "rnn":
        model = RNN()
        model.build_model(input_shape=X_train_obj.shape[1:])
        model.compile(
            learning_rate=0.00025, loss="binary_crossentropy", metrics=AUC(name="auc")
        )
        print(model.summary())
        # train model
        history = model.fit(
            [X_train_obj],
            y_train.values[:, -1],
            validation_data=(
                [X_valid_obj],
                y_valid.values[:, -1],
                sample_weights,
            ),
            epochs=100,
            shuffle=True,
            batch_size=64,
            callbacks=callbacks,
            class_weight=class_weight,
        )
        path = "./models/trained_models/final_rnn_new_data_smallerbatch"
        model.model.save(path)
        analyser = Evaluate(
            model,
            [X_test_obj],
            y_test.values[:, -1],
            test_xs_weights,
            path=path + "/",
        )
    elif model_type.lower() == "rnn_multi":
        model = RNN()
        model.build_model(input_shape=X_train_obj.shape[1:], n_outputs=4)
        model.compile(
            learning_rate=0.0005, loss="categorical_crossentropy"
        )
        print(model.summary())
        # weights
        y_integers = np.argmax(y_train.values, axis=1)
        class_weights = sklearn.utils.class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y_integers), y=y_integers)
        d_class_weights = dict(enumerate(class_weights))
        train_weights = np.array([d_class_weights[i] for i in np.argmax(y_train.values, axis=1)])
        valid_weights = np.array([d_class_weights[i] for i in np.argmax(y_valid.values, axis=1)])
        callbacks = [EarlyStopping(patience=10), ReduceLROnPlateau(patience=10), TB]
        # train model
        history = model.fit(
            [X_train_obj],
            y_train.values,
            validation_data=(
                [X_valid_obj],
                y_valid.values,
                valid_weights,
            ),
            epochs=80,
            shuffle=True,
            batch_size=256,
            callbacks=callbacks,
            sample_weight=train_weights
        )
        path = "./models/trained_models/final_rnn_new_data_multi"
        model.model.save(path)
    
    elif model_type.lower() == "rnn_multi_all":
        data = Data(datasets="all", exclude=[], seed=5)
        X_train, X_test = data.sequential()
        y_train, y_test = data.labels()
        X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.1, random_state=42, stratify=y_train)
        model = RNN()
        model.build_model(input_shape=X_train.shape[1:], n_outputs=7)
        model.compile(
            learning_rate=0.0005, loss="categorical_crossentropy"
        )
        print(model.summary())
        # weights
        y_integers = np.argmax(y_train.values, axis=1)
        class_weights = sklearn.utils.class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y_integers), y=y_integers)
        d_class_weights = dict(enumerate(class_weights))
        train_weights = np.array([d_class_weights[i] for i in np.argmax(y_train.values, axis=1)])
        valid_weights = np.array([d_class_weights[i] for i in np.argmax(y_valid.values, axis=1)])
        callbacks = [EarlyStopping(patience=10), ReduceLROnPlateau(patience=5), TB]
        # train model
        history = model.fit(
            [X_train],
            y_train.values,
            validation_data=(
                [X_valid],
                y_valid.values,
                valid_weights,
            ),
            epochs=80,
            shuffle=True,
            batch_size=256,
            callbacks=callbacks,
            sample_weight=train_weights
        )
        path = "./models/trained_models/final_rnn_new_data_multi_all"
        model.model.save(path)


    elif model_type.lower() == "mlp":
        model = MLP()
        model.build_model(input_shape=X_train_event.shape[1:])
        model.compile(
            learning_rate=0.0005, loss="binary_crossentropy", metrics=AUC(name="auc")
        )
        print(model.summary())
        history = model.fit(
            [X_train_event],
            y_train.values[:, -1],
            validation_data=([X_valid_event], y_valid.values[:,-1], sample_weights),
            epochs=80,
            shuffle=True,
            batch_size=128,
            callbacks=callbacks,
            class_weight=class_weight,
        )
        path = "./models/trained_models/final_mlp_new_data"
        model.model.save(path)
        analyser = Evaluate(
            model,
            [X_test_event],
            y_test.values[:, -1],
            test_xs_weights,
            path=path + "/",
        )
    elif model_type.lower() == "cnn":
        model = CNN()
        model.build_model(input_shape=X_train_image.shape[1:], n_outputs=1, filters=64, kernel_size=(4, 4), strides=(2, 2), activation="selu", padding="SAME")
        model.compile(
            learning_rate=0.0005, loss="binary_crossentropy", metrics=AUC(name="auc")
        )
        print(model.summary())
        history = model.fit([X_train_image], y_train.values[:,-1],validation_data=(
                [X_valid_image],
                y_valid.values[:, -1],
                sample_weights,
            ),
            epochs=80,
            shuffle=True,
            batch_size=64,
            callbacks=callbacks,
            class_weight=class_weight,
        )
        path = "./models/trained_models/final_cnn_new_data"
        model.model.save(path)
        analyser = Evaluate(
            model, 
            [X_test_image], 
            y_test.values[:,-1],
            test_xs_weights,
            path=path+"/"
        )
    analyser.plot_discriminator(loc="upper center")
    analyser.ROC(log=True)
    ams = analyser.significance()
    # plot learning curves
    epochs = range(1, len(history.history["loss"]) + 1)
    for k, v in history.history.items():
        if "val" in k:
            continue
        if k == "lr":
            plt.plot(epochs, v)
        else:
            fig, ax = plt.subplots()
            ax.plot(epochs, v, label="Training")
            ax.plot(epochs, history.history["val_" + k], label="Validation")
        ax.set_xlabel("epochs")
        ax.set_ylabel(k)
        ax.legend()
        fig.savefig(f"{path}/{k}_learning_curve.png", dpi=200)
        plt.show()
def get_run_logdir():
    root_logdir = "./models/tensorboard_logs/"
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    return os.path.join(root_logdir, run_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_type", help="Which model to train", choices=["mlp", "rnn", "cnn", "combined", "rnn_multi", "rnn_multi_all"])
    parser.add_argument("-g", "--gpu", help="GPU number on which to train model", type=int, choices=range(6), default=5)
    args = parser.parse_args()
    # GPU setup
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        # Restrict TensorFlow to only use the fourth GPU
        try:
            tf.config.set_visible_devices(
                gpus[args.gpu], "GPU"
            )  # change to n-1 to use nth gpu
            logical_gpus = tf.config.list_logical_devices("GPU")
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
        except RuntimeError as e:
            # Visible devices must be set before GPUs have been initialized
            print(e)
    main(args.model_type)