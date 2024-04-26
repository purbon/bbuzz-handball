import pandas as pd

from thinker.processor import CentroidsEmbedding, FlattenEmbedding

if __name__ == "__main__":

    df = pd.read_hdf(path_or_buf=f"dumps/dataset.h5", key="pos")

    preproc = FlattenEmbedding(raw_df=df, avg_length=25)
    view = preproc.process(aggregation_key="possession") # sequence_label possession
    print(view)