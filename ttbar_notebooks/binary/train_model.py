import numpy as np
from Preprocessing import PreProcess
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Input, Dense, Flatten, AveragePooling2D, AlphaDropout
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.metrics import AUC, Precision, Recall, SensitivityAtSpecificity, BinaryAccuracy, CategoricalAccuracy
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
import sys
from Preprocessing import PreProcess

def main(model_path, data="image", batch_size=64, epochs=50):
    model = keras.models.load_model(f"./models/{model_path}")
    # data = np.load(f"./data/{data_path}", allow_pickle=True)
    processer = PreProcess("all")
    X_train, X_test = processer.image()
    y_train, y_test = processer.labels()
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)
    callbacks = [EarlyStopping(patience=10, monitor="val_loss", restore_best_weights=True), ReduceLROnPlateau(patience=3, monitor="val_loss")]
    class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(np.argmax(y_train.values, axis=1)), y=np.argmax(y_train.values, axis=1))
    d_class_weights = dict(enumerate(class_weights))
    train_weights = np.array([d_class_weights[i] for i in np.argmax(y_train.values, axis=1)])
    valid_weights = np.array([d_class_weights[i] for i in np.argmax(y_valid.values, axis=1)])
    print(train_weights)
    # test_weights = np.array([class_weights[i] for i in np.argmax(y_test.values, axis=1)])
    model.fit(X_train, y_train, validation_data=(X_valid, y_valid, valid_weights), epochs=int(epochs), batch_size=int(batch_size), sample_weight=train_weights, callbacks=callbacks)
    model.save(f"./models/trained_models/trained_{model_path}")

if __name__ == "__main__":
    # GPU setup
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        # Restrict TensorFlow to only use the fourth GPU
        try:
            tf.config.set_visible_devices(gpus[3], 'GPU') # change to n-1 to use nth gpu
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
        except RuntimeError as e:
            # Visible devices must be set before GPUs have been initialized
            print(e)
    main(*sys.argv[1:])