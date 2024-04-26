from thinker.meta import Schema
import numpy as np
import pandas as pd

from thinker.models import CentroidsModel, Kinetics


class CentroidsEmbedding:

    def __init__(self, raw_df):
        self.raw_df = raw_df

    def process(self, aggregation_key):
        df = self.raw_df.copy()
        df = df[df[Schema.ACTIVE_PLAYERS_COLUMN] == 6]

        new_df_columns = ["game", aggregation_key]
        extra_attributes = df.columns.to_list()

        try:
            extra_attributes.remove(aggregation_key)
        except ValueError:
            pass

        for i in range(6):
            new_df_columns.append(f"p{i}_x_centroid")
            new_df_columns.append(f"p{i}_y_centroid")
        new_df_columns += extra_attributes

        flatten_view_df = pd.DataFrame()#([], columns=new_df_columns)


        cm = CentroidsModel()
        km = Kinetics()

        agg_fields = ["misses", "throws", "goal"]

        for game, data_points in df.groupby(["game", aggregation_key]):
            print(f"Processing {game[0]} {game[1]}")
            first_record_in_series = data_points.head(1)
            last_record_in_series = data_points.tail(1)
            phase = first_record_in_series["game_phases"].values[0]

            cdf = cm.fit(possession_array=data_points)
            kdf = km.fit(poss_array=data_points)
            cdf = pd.concat((cdf, kdf), axis=1)

            cdf["game"] = game[0]
            cdf[aggregation_key] = game[1]

            for extra_attribute in extra_attributes:
                if extra_attribute == "game_phases":
                    val = first_record_in_series[extra_attribute].iloc[0]
                else:
                    if aggregation_key == "possession" and extra_attribute in agg_fields:
                        values_arr = data_points.groupby("sequence_label")[extra_attribute].unique().values
                        val = np.array([value_arr[0] for value_arr in values_arr]).sum()
                    elif aggregation_key == "possession" and phase == "AT" and extra_attribute == "organized_game":
                        values_arr = data_points.groupby("sequence_label")[extra_attribute].unique().values
                        val_arr = np.array([value_arr[0] for value_arr in values_arr])
                        val = "yes" if "yes" in val_arr else "no"
                    else:
                        val = last_record_in_series[extra_attribute].iloc[0]
                cdf[extra_attribute] = val

            flatten_view_df = pd.concat((flatten_view_df, cdf), ignore_index=True)

        return flatten_view_df