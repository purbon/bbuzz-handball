import keras
from keras import Input
from keras.src.layers import Embedding, Dense, LSTM, Conv1D, Dropout, Flatten, \
    BatchNormalization, GlobalAveragePooling1D, LayerNormalization, MultiHeadAttention, Reshape


def get_lstm_model(input_size):
    model = keras.Sequential()
    model.add(Embedding(10_001, 10,
                        input_length=input_size,
                        mask_zero=True))
    model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
    model.add(LSTM(100, dropout=0.25, recurrent_dropout=0.25))
    model.add(Dense(1, activation='sigmoid'))
    return model


def get_lstm2_model(input_shape):
    model = keras.Sequential()
    model.add(LSTM(8, input_shape=input_shape, dropout=0.2, recurrent_dropout=0.2, return_sequences=True))
    model.add(BatchNormalization())
    model.add(LSTM(8, dropout=0.2, recurrent_dropout=0.2, return_sequences=True))
    model.add(BatchNormalization())
    model.add(LSTM(8, dropout=0.2, recurrent_dropout=0.2, return_sequences=False))
    model.add(BatchNormalization())
    model.add(Dense(1, activation='sigmoid'))
    return model


def get_dense_model(input_size):
    model = keras.Sequential()
    model.add(Embedding(10_001, 40,
                        input_length=input_size,
                        mask_zero=True))
    model.add(Flatten())
    model.add(Dense(32, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid', bias_initializer='zeros'))
    return model


def get_dense2_model(input_size):
    model = keras.Sequential()
    model.add(Dense(32, input_dim=input_size, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    return model


def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Attention and Normalization
    x = MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(inputs, inputs)
    x = Dropout(dropout)(x)
    x = LayerNormalization(epsilon=1e-6)(x)
    res = x + inputs

    # Feed Forward Part
    x = Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(res)
    x = Dropout(dropout)(x)
    x = Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    x = LayerNormalization(epsilon=1e-6)(x)
    return x + res


def get_transformer_model(
        input_shape,
        head_size,
        num_heads,
        ff_dim,
        num_transformer_blocks,
        mlp_units,
        dropout=0.0,
        mlp_dropout=0.0,
        n_classes=1,
):
    inputs = keras.Input(shape=input_shape)
    x = inputs
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    x = GlobalAveragePooling1D(data_format="channels_last")(x)
    for dim in mlp_units:
        x = Dense(dim, activation="relu")(x)
        x = Dropout(mlp_dropout)(x)
    outputs = Dense(1, activation="sigmoid")(x)
    return keras.Model(inputs, outputs)


def get_ann_model(model_type, input_size, timesteps=30, n_fields=12):
    if model_type == "lstm":
        model = get_lstm_model(input_size=input_size)
    elif model_type == "lstm2":
        input_shape = (timesteps, n_fields)
        model = get_lstm2_model(input_shape)
    elif model_type == "dense":
        model = get_dense_model(input_size=input_size)
    elif model_type == "dense2":
        model = get_dense_model(input_size=input_size)
    elif model_type == "transformer":
        input_shape = (timesteps, n_fields)

        model = get_transformer_model(
            input_shape,
            head_size=256,
            num_heads=4,
            ff_dim=4,
            num_transformer_blocks=4,
            mlp_units=[128],
            mlp_dropout=0.2,
            dropout=0.25,
        )
    else:
        raise Exception("Fuck!")
    return model
