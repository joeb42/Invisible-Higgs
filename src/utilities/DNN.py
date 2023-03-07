from tensorflow.keras.layers import (
    Dense,
    AlphaDropout,
    Concatenate,
    LayerNormalization,
    GRU,
    Flatten,
    Conv2D,
    AveragePooling2D,
)
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras import Model
from typing import Union


class MLP(Model):
    """
    Muli-layer perceptron model, designed to take tabular event level features as input
    """

    def __init__(
        self,
        *,
        n_outputs: int,
        n_layers: int,
        neurons: Union[tuple, int],
        dropout_rate: float = 0.2,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.dense_layers = [Dense(neurons) for layer in range(n_layers)]
        self.dropout_rate = dropout_rate
        activation = "sigmoid" if n_outputs == 1 else "softmax"
        self.out = Dense(n_outputs, activation=activation)

    def compile(self, *args, learning_rate: float = 0.001, **kwargs) -> None:
        opt = Nadam(learning_rate=learning_rate)
        super().compile(*args, optimizer=opt, **kwargs)

    def call(self, x):
        for idx, layer in enumerate(self.dense_layers):
            x = layer(x)
            if idx < len(self.dense_layers) - 1:
                x = AlphaDropout(rate=self.dropout_rate)(x)
        return self.out(x)


class CNN(Model):
    """Convolutional neural network (CNN) model designed to take an image like input representative of the jets in an event"""

    def __init__(
        self,
        *,
        input_shape: tuple[int, int],
        n_outputs: int = 1,
        n_conv_layers: int = 2,
        n_conv_filters: int = 64,
        filter_sizes: Union[list[tuple[int, int]], tuple[int, int]] = (2, 2),
        n_dense_layers: int = 2,
        n_dense_neurons: int = 200,
        dropout: float = 0.5,
        **kwargs,
    ) -> None:
        super().__init__(self, **kwargs)
        self.conv_layers = [
            Conv2D(n_conv_filters, filter_sizes, activation="relu")
            for layer in range(n_conv_layers)
        ]
        self.dense_layers = [
            Dense(n_dense_neurons, activation="selu", kernel_initializer="lecun_normal")
            for layer in range(n_dense_layers)
        ]
        self.dropout_rate = dropout
        activation = "sigmoid" if n_outputs == 1 else "softmax"
        self.out = Dense(n_outputs, activation=activation)

    def compile(self, *args, learning_rate: float = 0.001, **kwargs) -> None:
        opt = Nadam(learning_rate=learning_rate)
        super().compile(*args, optimizer=opt, **kwargs)

    def call(self, x):
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
            x = AveragePooling2D(pool_size=(2, 2))(x)
        x = Flatten()(x)
        for dense_layer in self.dense_layers[:-1]:
            x = dense_layer(x)
            if self.dropout_rate > 0:
                x = AlphaDropout(self.dropout_rate)(x)
        x = self.Dense_layers[-1](x)
        return self.out(x)


class RNN(Model):
    """
    Recurrent neural network model (RNN) designed to take into sequences of jet level features
    """

    def __init__(
        self,
        *,
        n_outputs: int = 1,
        rnn_layers=3,
        rnn_neurons=200,
        recurrent_dropout=0,
        dense_layers=0,
        dense_neurons=40,
        dropout=0.5,
        **kwargs,
    ) -> None:
        super().__init__(self, **kwargs)
        self.rnn_layers = [
            GRU(rnn_neurons, recurrent_dropout=recurrent_dropout)
            for layer in range(rnn_layers)
        ]
        self.dense_layers = [
            Dense(dense_neurons, activation="selu", kernel_initializer="lecun_normal")
            for layer in range(dense_layers)
        ]
        self.dropout_rate = dropout
        activation = "sigmoid" if n_outputs == 1 else "softmax"
        self.out = Dense(n_outputs, activation=activation)

    def compile(self, *args, learning_rate: float = 0.001, **kwargs) -> None:
        opt = Nadam(learning_rate=learning_rate)
        super().compile(*args, optimizer=opt, **kwargs)

    def call(self, x):
        # GRU layers
        for layer in self.rnn_layers:
            x = layer(x)
            x = LayerNormalization(x)
        # Dense Layers
        for layer in self.dense_layers:
            if self.dropout_rate > 0 and layer > 0:
                x = AlphaDropout(rate=self.dropout_rate)(x)
            x = layer(x)
        return self.out(x)


class RNN_Combined(Model):
    """Combination of RNN and MLP models"""

    def __init__(
        self,
        *,
        n_outputs: int = 1,
        rnn_layers=3,
        rnn_neurons=200,
        recurrent_dropout=0,
        dense_layers=0,
        dense_neurons=40,
        dropout=0.5,
        **kwargs,
    ) -> None:
        super().__init__(self, **kwargs)
        self.rnn = RNN(
            rnn_layers=rnn_layers,
            rnn_neurons=rnn_neurons,
            recurrent_dropout=recurrent_dropout,
            n_outputs=dense_neurons,
        )
        self.mlp = MLP(
            n_outputs=n_outputs,
            n_layers=dense_layers,
            neurons=dense_neurons,
            dropout_rate=dropout,
        )

    def call(self, x_jet, x_event):
        x_jet = self.rnn(x_jet)
        x_jet = Flatten()(x_jet)
        x = Concatenate()([x_jet, x_event])
        x = self.mlp(x)
        return x
