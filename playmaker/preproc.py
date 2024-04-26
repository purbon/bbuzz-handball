import os

import pandas as pd

from thinker.processor import CentroidsEmbedding, FlattenEmbedding

if __name__ == "__main__":

    df = pd.read_hdf(path_or_buf=f"dumps/dataset.h5", key="pos")

    key = "centroids" # centroids flatten

    if key == "flatten":
        preproc = FlattenEmbedding(raw_df=df, avg_length=25)
    else:
        preproc = CentroidsEmbedding(raw_df=df)

    view = preproc.process(aggregation_key="possession") # sequence_label possession

    view.to_excel(os.path.join("dumps", f"{key}.xlsx"), index=False)
    view.to_hdf(os.path.join("dumps", "dataset.h5"), key=key)