import os

import pandas as pd


def extract_coordinates_from(df, io_path="dumps/"):
    os.makedirs(io_path, exist_ok=True)

    df_copy = df.copy()
    # "score_diff", "live_possession_duration_in_sec"
    columns_to_remove = ["Unnamed: 14",
                         "POS. B", "POS. B.1", "analysis",
                         'TIME_PER_HALF_IN_SECONDS', 'STOPWATCH_TIME_IN_SECONDS', 'EFFECTIVE_TIME_IN_SECONDS']

    columns_to_remove += [f"player_{i}" for i in range(6)]
    columns_to_remove += [f"prev{i}_possession_result" for i in range(3, 6, 1)]
    columns_to_remove += [f"prev{i}_score_diff" for i in range(3, 6, 1)]

    df_copy.drop(columns=columns_to_remove, inplace=True)
    df_copy.rename(mapper=lambda x: x.lower(), axis=1, inplace=True)

    df_copy.to_excel(os.path.join(io_path, "dataset.xlsx"), index_label=False)
    df_copy.to_hdf(os.path.join(io_path, "dataset.h5"), key="pos")


if __name__ == '__main__':
    df = pd.read_hdf(path_or_buf=f"data/handball.h5", key="pos")

    extract_coordinates_from(df, io_path="dumps")
