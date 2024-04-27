import math
import os
import pickle

import pandas as pd
from keras import saving
from sklearn.preprocessing import MinMaxScaler

from thinker.ann.autoencoders import lstm_autoencoder_em
from thinker.ann.data import DataGameGenerator


class AutoEmbeddings:

    def __init__(self):
        self.n_features = 12
        self.scaler = MinMaxScaler()
        self.model = None
        self.encoder = None

    # model_name = f"keras.{lse_size}_{lstm_units}-{levels}-{timesteps}.model"
    def train(self, df, model_name, lstm_units, levels, timesteps=25, epochs=500):

        if os.path.exists(model_name):
            model, history = self.__load_trained_model(model_name)
            print(model.summary())
            print(model.layers[-1].summary())
            self.model = model

        total_size = df.groupby(["game", "possession"]).count().shape[0]

        trainSize = int(math.ceil(0.7 * total_size * 2))
        testSize = int(math.ceil(0.2 * total_size * 2))
        valSize = int(math.ceil(0.1 * total_size * 2))

        fields = []
        for i in range(6):
            fields += [f"player_{i}_x", f"player_{i}_y"]

        df[fields] = self.scaler.fit_transform(df[fields])

        lse_size = int(lstm_units / pow(2, levels - 1))

        trainGenerator = DataGameGenerator(df=df, timesteps=timesteps,
                                           start=0, end=trainSize,
                                           augment=True)
        testGenerator = DataGameGenerator(df=df, timesteps=timesteps,
                                          start=trainSize, end=(trainSize + testSize),
                                          augment=True)
        valGenerator = DataGameGenerator(df=df, timesteps=timesteps,
                                         start=(trainSize + testSize),
                                         end=(trainSize + testSize + valSize),
                                         display_labels=False,
                                         augment=True)

        encoder, decoder, model = lstm_autoencoder_em(timesteps=timesteps, n_features=self.n_features,
                                                      lstm_units=lstm_units, levels=levels)

        model.summary()
        trainHistory = model.fit(trainGenerator,
                                 epochs=epochs,
                                 batch_size=32,
                                 validation_data=valGenerator,
                                 verbose=1)

        score = model.evaluate(testGenerator, verbose=0)
        print(model.metrics_names)
        print(score)
        self.__save_trained_model(model_name, model, trainHistory, overwrite=True)
        self.model = model
        self.encoder = encoder

    def eval(self, df, handball_df, lstm_units, levels, timesteps=25):
        mdf = pd.DataFrame()

        lse_size = int(lstm_units / pow(2, levels - 1))
        total_size = df.groupby(["game", "possession"]).count().shape[0]
        dataGenerator = DataGameGenerator(df=df, timesteps=timesteps, start=0, end=total_size)

        def map_row(row, i):
            return row.origin.split("_")[i]

        for j in range(len(dataGenerator)):
            X, y = dataGenerator.__getitem__(j)
            embeddings = self.encoder.predict(X)
            O = dataGenerator.__getorigins__(j)
            ope = pd.DataFrame(embeddings)
            oo = pd.DataFrame(O, columns=['origin'])
            oo["game"] = oo.apply(lambda row: map_row(row, 0), axis=1)
            oo["possession"] = oo.apply(lambda row: map_row(row, 1), axis=1)
            oo.drop(columns='origin', inplace=True)
            idf = ope.join(oo)
            mdf = pd.concat([mdf, idf])
            mdf["game"] = mdf["game"].astype(int)

        def filter_columns(column):
            pcount = [f"p{i}" for i in range(6)]
            team = ["team_x_centroid", "team_y_centroid"]
            return not column.startswith("player_") and column[:2] not in pcount and column not in team

        columns = list(filter(filter_columns, handball_df.columns))
        handball_df = handball_df[columns]
        mdf = pd.merge(handball_df.reset_index(), mdf, on=['game', 'possession'], how='inner')
        return mdf

    def __save_trained_model(self, fileName, theModel, trainHistory, overwrite=False):
        if not overwrite and os.path.exists(fileName):
            return None
        print(f'Overwriting the model {fileName}')
        saving.save_model(theModel, fileName, overwrite=True)
        with open('keras.history', 'wb') as file:
            pickle.dump(trainHistory.history, file)

    def __load_trained_model(self, fileName="model.keras"):
        model = saving.load_model(fileName)
        with open('keras.history', "rb") as file:
            history = pickle.load(file)
        return model, history
