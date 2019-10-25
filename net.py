import keras
import keras.backend as K
from keras import layers as L


def _add_lstm_layer(
        input_layer,
        n_layers,
        hidden_size,
        dropout=0.4,
        bidirectional=False):
    layer = input_layer

    for i in range(n_layers):
        lstm = L.LSTM(
            hidden_size,
            recurrent_dropout=dropout,
            return_sequences=True)

        if bidirectional:
            layer = L.Bidirectional(lstm)(layer)
        else:
            layer = lstm(layer)
    return layer


def open_unmix(
        input_layer,
        hidden_size=128,
        n_layers=1,
        unidirectional=False):

    if n_layers <= 0:
        raise ValueError("n_layers must be greater than zero")

    if unidirectional:
        lstm_hidden_size = hidden_size
    else:
        lstm_hidden_size = hidden_size // 2

    _, nb_channels, _, nb_bins = K.int_shape(input_layer)

    # (nb_samples, nb_channels, nb_frames, nb_bins) to (nb_samples, nb_frames, nb_bins * nb_channels)
    x = L.Reshape((-1, nb_bins * nb_channels), name="umx_reshape_1")(input_layer)

    x = L.Dense(hidden_size, use_bias=False, name="umx_fc1")(x)
    x = L.BatchNormalization()(x)
    # tanh
    x = L.Activation("tanh")(x)

    lstm = _add_lstm_layer(
        x,
        n_layers=n_layers,
        hidden_size=lstm_hidden_size,
        dropout=0.25,
        bidirectional=not unidirectional)
    x = L.Concatenate()([x, lstm])

    x = L.Dense(hidden_size, use_bias=False, name="umx_fc2")(x)
    x = L.BatchNormalization()(x)
    x = L.Activation("relu")(x)
    # x = L.ReLU(6.0)(x)

    x = L.Dense(nb_bins * nb_channels, use_bias=False, name="umx_fc3")(x)
    x = L.BatchNormalization()(x)
    x = L.Activation("relu")(x)
    # x = L.ReLU(1.0)(x)
    x = L.Reshape((nb_channels, -1, nb_bins), name="umx_reshape_2")(x)
    return x


def create_model(
        input_shape=(2, None, 2049),
        hidden_size=512,
        n_layers=3,
        unidirectional=False
):
    """[summary]

    Args:
        input_shape (n_samples, n_channels, n_frames, n_bins)
        hidden_size
        n_layers
        unidirectional

    """

    input_layer = L.Input(shape=input_shape)
    _, nb_channels, _, nb_bins = K.int_shape(input_layer)

    x = open_unmix(
        input_layer,
        hidden_size=hidden_size,
        n_layers=n_layers,
        unidirectional=unidirectional)

    x = L.Multiply(name="umx_multiply")([x, input_layer])
    
    return keras.Model(inputs=[input_layer], outputs=[x])
