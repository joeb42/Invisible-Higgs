import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
<<<<<<< HEAD
from keras.layers import Input, LSTM, GRU, Dense, LayerNormalization
from keras.models import Model, Sequential
import keras_tuner as kt
from sklearn.model_selection import train_test_split
=======
from keras.layers import Input, LSTM, Dense, Concatenate, Flatten, Dropout
from keras.models import Model, Sequential
import keras_tuner as kt
>>>>>>> 678335accac650b61d7df38897595cb3c1b53e47

def build_model(hp):
    """
    Paramaters to tune:
    - Number of units per LSTM layer
    - No of LSTM layers
    - No of units per feedforward hidden layer
    - No of feedforward hidden layers
    - Activation functions for feedforward (RELU vs elu vs selu)
    - Batch normalisation 
    - Drop out
    - Optmiser (sgd vs Adam vs Nadam)

    """
    obj_input = Input(shape=X_train_obj.shape[1:])
    prev = obj_input
<<<<<<< HEAD
    RNN_layer_type = hp.Choice("RNN layer kind", ["LSTM", "GRU"])
    RNN_units = hp.Int("RNN units", 50, 300, 50)
    for _ in range(hp.Int("RNN layers", 1, 3, 1)-1):
        if RNN_layer_type == "LSTM":
            prev = LSTM(RNN_units, return_sequences=True)(prev)
        elif RNN_layer_type == "GRU":
            prev = GRU(RNN_units, return_sequences=True)(prev)
        prev = LayerNormalization()(prev)
    if RNN_layer_type == "LSTM":
        RNN_out = LSTM(RNN_units)(prev)
    elif RNN_layer_type == "GRU":
        RNN_out = GRU(RNN_units)(prev)
    ln = LayerNormalization()(RNN_out)
    out = Dense(1, activation="sigmoid")(ln)
    model = Model(inputs=obj_input, outputs=out)
    model.compile(loss="binary_crossentropy", optimizer="Nadam", metrics=[keras.metrics.AUC(), keras.metrics.Precision(), keras.metrics.Recall()])  
=======
    for _ in range(hp.Int("LSTM layers", 1, 4, 1)-1):
        prev = LSTM(hp.Int("LSTM units", 16, 64, 16), return_sequences=True)(prev)
    lstm_out = LSTM(hp.Int("LSTM units", 16, 64, 16))(prev)
    event_input = Input(shape=X_train_event.shape[1])
    x = Concatenate()([lstm_out, event_input])
    prev = x 
    activation = hp.Choice("activation", ["relu", "selu"])
    for _ in range(hp.Int("Feedforward layers", 2, 5, 1)):
        prev = Dense(hp.Int("Feedfoward units", 100, 400, 100), activation=activation)(prev)
        if activation == "selu":
            prev.kernel_initializer = keras.initializers.LecunNormal()
        if hp.Boolean("dropout"):
            prev = Dropout(rate=0.2)(prev)
    out = Dense(1, activation="sigmoid")(prev)
    model = Model(inputs=[obj_input, event_input], outputs=out)
    model.compile(loss="binary_crossentropy", optimizer=hp.Choice("optimiser", ["Adam", "Nadam"]), metrics=[keras.metrics.AUC(), keras.metrics.Precision(), keras.metrics.Recall()])  
>>>>>>> 678335accac650b61d7df38897595cb3c1b53e47
    return model

# Load data
full_data = np.load('./full_data.npy', allow_pickle=True)
<<<<<<< HEAD
X_event, X_obj, y = full_data[0], full_data[2], full_data[4]
X_train_event, X_valid_event, X_train_obj, X_valid_obj, y_train, y_valid = train_test_split(X_event, X_obj, y, train_size=0.8, stratify=y, random_state=42)
# print(f'training ratio: {np.sum(y_train)/len(y_train)} \nvalidation ratio {np.sum(y_valid)/len(y_valid)}')
# assert np.sum(y_train)/len(y_train) == np.sum(y_valid)/len(y_valid), "Should be same proportion of signal in training and test set"

tot = len(y_train)
pos = np.sum(y_train['ttH125'])
=======
X_train_event, X_test_event, X_train_obj, X_test_obj, y_train, y_test = full_data

tot = len(y_train)
pos = np.sum(y_train == 1)
>>>>>>> 678335accac650b61d7df38897595cb3c1b53e47
neg = tot - pos
# weight positives more than negatives
weight_for_0 = (1 / neg) * (tot / 2.0)
weight_for_1 = (1 / pos) * (tot / 2.0)
class_weight = {0: weight_for_0, 1: weight_for_1}

<<<<<<< HEAD
# # GPU set up
# all_devices = len(tf.config.list_physical_devices('GPU'))
# print("Num GPUs Available: ", all_devices)
# physical_devices=tf.config.list_physical_devices('GPU')
# gpus= tf.config.experimental.list_physical_devices('GPU')
# for i in range(0,all_devices):
#     tf.config.experimental.set_memory_growth(gpus[i], True)

# mirrored_strategy = tf.distribute.MirroredStrategy(devices=[f"/GPU:{GPU_id}" for GPU_id in range (0,6)])

# Tuning
tuner = kt.BayesianOptimization(build_model, objective='val_loss', max_trials=100, overwrite=True)
tuner.search(X_train_obj, y_train['ttH125'], epochs=10, validation_data=(X_valid_obj, y_valid['ttH125']),class_weight=class_weight, batch_size=64)
best_HP = tuner.get_best_hyperparameters()[0]
print(best_HP.values)
best_model = tuner.hypermodel.build(best_HP)
best_model.save('./RNN_tuned_model')
=======
# GPU set up
all_devices = len(tf.config.list_physical_devices('GPU'))
print("Num GPUs Available: ", all_devices)
physical_devices=tf.config.list_physical_devices('GPU')
gpus= tf.config.experimental.list_physical_devices('GPU')
for i in range(0,all_devices):
    tf.config.experimental.set_memory_growth(gpus[i], True)

mirrored_strategy = tf.distribute.MirroredStrategy(devices=[f"/GPU:{GPU_id}" for GPU_id in range (0,6)])

# Tuning
tuner = kt.RandomSearch(build_model, objective='val_loss', max_trials=50, overwrite=True)
tuner.search([X_train_obj, X_train_event], y_train, epochs=20, validation_data=([X_test_obj, X_test_event], y_test),class_weight=class_weight)
best_HP = tuner.get_best_hyperparameters()[0]
print(best_HP.values)
best_model = tuner.hypermodel.build(best_HP)
best_model.save('./RNN_tuned_model')


>>>>>>> 678335accac650b61d7df38897595cb3c1b53e47
