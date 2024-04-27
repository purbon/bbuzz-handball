from keras import Model, Sequential, saving, layers
from keras.src.layers import Dense, LSTM, RepeatVector, TimeDistributed
from keras.src.optimizers import Adam


class lstm_masked(layers.Layer):
    def __init__(self, lstm_units, **kwargs):
        self.lstm_units = lstm_units
        self.lstm_layer = LSTM(lstm_units, return_sequences=False)
        super(lstm_masked, self).__init__(**kwargs)

    def call(self, inputs):
        # just call the two initialized layers
        return self.lstm_layer(inputs)

    def compute_mask(self, inputs, mask=None):
        # return the input_mask directly
        return mask


class lstm_repeatvector(layers.Layer):
    def __init__(self, time_steps, **kwargs):
        self.time_steps = time_steps
        self.repeat_layer = RepeatVector(time_steps)
        super(lstm_repeatvector, self).__init__(**kwargs)

    def call(self, inputs):
        return self.repeat_layer(inputs)

    def compute_mask(self, inputs, mask=None):
        # return the input_mask directly
        return mask


def lstm_autoencoder_em(timesteps, n_features, lstm_units=128, levels=2):
    encoder_input = layers.Input(shape=(timesteps, n_features), name="encoder_input")
    x = layers.Masking(mask_value=0)(encoder_input)

    units_count = lstm_units
    for i in range(levels - 1):
        x = LSTM(units_count, activation='relu', return_sequences=True)(x)
        units_count = int(units_count / 2)

    encoder_output = lstm_masked(lstm_units=units_count)(x)
    encoder = Model(encoder_input, encoder_output)

    the_decoder = Sequential()
    the_decoder.add(layers.Input(shape=(units_count,), name="encoder_input"))
    the_decoder.add(lstm_repeatvector(time_steps=timesteps))
    for i in range(levels - 1):
        the_decoder.add(LSTM(units_count, activation='relu', return_sequences=True))
        units_count = int(units_count * 2)
    the_decoder.add(LSTM(lstm_units, activation='relu', return_sequences=True))
    the_decoder.add(TimeDistributed(Dense(n_features)))

    autoencoder = Model(inputs=[encoder_input], outputs=the_decoder(encoder_output))
    autoencoder.compile(optimizer="adam", loss='mean_squared_error', metrics=["mae", "acc"])

    return encoder, the_decoder, autoencoder
