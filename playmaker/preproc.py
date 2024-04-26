import pandas as pd

from thinker.processor import CentroidsEmbedding

if __name__ == "__main__":

    df = pd.read_hdf(path_or_buf=f"dumps/dataset.h5", key="pos")

    centroids = CentroidsEmbedding(raw_df=df)
    view = centroids.process(aggregation_key="sequence_id") # sequence_id possession
    print(view)