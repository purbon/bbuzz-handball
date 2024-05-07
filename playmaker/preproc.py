import os

import pandas as pd

from thinker.embeddings import AutoEmbeddings
from thinker.meta import Schema
from thinker.processor import CentroidsEmbedding, FlattenEmbedding


def build_manual_embeddings(df, key):
    if key == "flatten":
        preproc = FlattenEmbedding(raw_df=df, avg_length=30)
    else:
        preproc = CentroidsEmbedding(raw_df=df)

    for agg_key in ["possession", "sequence_label"]:
        view = preproc.process(aggregation_key=agg_key)  # sequence_label possession
        view_key = f"{key}s" if agg_key == "sequence_label" else f"{key}"
        view.to_excel(os.path.join("dumps", f"{view_key}.xlsx"), index=False)
        view.to_hdf(os.path.join("dumps", "dataset.h5"), key=view_key)


def build_autoencoder_embeddings(df):
    autokey = AutoEmbeddings()
    lstm_units = 512 #256
    levels = 2
    timesteps = 35

    lse_size = int(lstm_units / pow(2, levels - 1))
    model_name = f"keras.{lse_size}_{lstm_units}-{levels}-{timesteps}.keras"

    autokey.train(df=df, model_name=model_name, lstm_units=lstm_units, timesteps=timesteps, levels=levels, epochs=50)

    handball_df = pd.read_hdf(path_or_buf=f"dumps/dataset.h5", key="centroids")
    mdf = autokey.eval(df=df, lstm_units=lstm_units, levels=levels, timesteps=timesteps, handball_df=handball_df)

    mdf.to_excel(f"lstm-autoencoder-embeddings-{lse_size}-{timesteps}.xlsx", index=False)
    mdf.to_hdf(path_or_buf="dumps/dataset.h5", key=f"lstm_autoenc_{lse_size}-{timesteps}")


if __name__ == "__main__":

    df = pd.read_hdf(path_or_buf=f"dumps/dataset.h5", key="pos")

    key = "les"  # centroids flatten les

    if key == "les":

        df = df[df[Schema.ACTIVE_PLAYERS_COLUMN] == 6]
        build_autoencoder_embeddings(df=df)
    else:
        build_manual_embeddings(df=df, key=key)
