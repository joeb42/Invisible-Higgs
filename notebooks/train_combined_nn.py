import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
from keras.layers import Input, LSTM, Dense, Concatenate, Dropout
from keras.models import Model

# # GPU memory management
# all_devices = len(tf.config.experimental.list_physical_devices('GPU'))
# # print("Num GPUs Available: ", all_devices)
# physical_devices=tf.config.experimental.list_physical_devices('GPU')
# gpus= tf.config.experimental.list_physical_devices('GPU')
# for i in range(0,all_devices):
#     tf.config.experimental.set_memory_growth(gpus[i], True)

# mirrored_strategy = tf.distribute.MirroredStrategy(devices=[f"/GPU:{GPU_id}" for GPU_id in range (0,6)])

# Load data
full_data = np.load('./full_data.npy', allow_pickle=True)
X_train_event, X_test_event, X_train_obj, X_test_obj, y_train, y_test = full_data

# Imbalanced dataset so want to adjusts weights of signal and background training examples
tot = len(y_train)
pos = np.sum(y_train == 1)
neg = tot - pos
# print(f'Total training samples:  {tot}\npositives:  {pos}\nnegatives:  {neg}')

# weight positives more than negatives
weight_for_0 = (1 / neg) * (tot / 2.0)
weight_for_1 = (1 / pos) * (tot / 2.0)

class_weight = {0: weight_for_0, 1: weight_for_1}
# print(f'Postive weight:  {weight_for_1} \nNegative weight:  {weight_for_0}')

checkpoint_path = "./Combined_RNN_checkpoints"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

# model = keras.models.load_model('./RNN_model')
obj_input = Input(shape=X_train_obj.shape[1:])
lstm = LSTM(64)(obj_input)
event_input = Input(shape=X_train_event.shape[1])
x = Concatenate()([lstm, event_input])
prev = x
for _ in range(5):
    prev = Dense(300, activation="selu")(prev)
    prev = Dropout(rate=0.2)(prev)
out = Dense(1, activation="sigmoid")(prev)
model = Model(inputs=[obj_input, event_input], outputs=out)
model.compile(loss="binary_crossentropy", optimizer="Nadam", metrics=[keras.metrics.AUC(), keras.metrics.Precision(), keras.metrics.Recall()])  
history = model.fit([X_train_obj, X_train_event], y_train, epochs=100, validation_data=([X_test_obj, X_test_event], y_test), class_weight=class_weight, callbacks=[cp_callback])
model.save('./RNN_model_100')