from thinker.meta import Schema
import numpy as np
import pandas as pd

from thinker.models import CentroidsModel, Kinetics, Normalizer, flatten


class EmbeddingPreProc:

    def __init__(self, raw_df):
        self.raw_df = raw_df

    def process(self, aggregation_key, coordinate_processor):
        df = self.raw_df.copy()
        df = df[df[Schema.ACTIVE_PLAYERS_COLUMN] == 6]

        new_df_columns = ["game", aggregation_key]
        extra_attributes = [attr for attr in df.columns.to_list() if not attr.startswith("player_")]

        try:
            extra_attributes.remove(aggregation_key)
        except ValueError:
            pass

        if isinstance(coordinate_processor, CentroidsProcessor):
            for i in range(6):
                new_df_columns.append(f"p{i}_x_centroid")
                new_df_columns.append(f"p{i}_y_centroid")
        new_df_columns += extra_attributes

        flatten_view_df = pd.DataFrame()

        agg_fields = ["misses", "throws", "goal"]

        for game, data_points in df.groupby(["game", aggregation_key]):
            print(f"Processing {game[0]} {game[1]}")
            first_record_in_series = data_points.head(1)
            last_record_in_series = data_points.tail(1)
            phase = first_record_in_series["game_phases"].values[0]

            cdf = coordinate_processor.apply(data_points=data_points)

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

            flatten_view_df = pd.concat((cdf, flatten_view_df), ignore_index=True)

        return flatten_view_df


class CentroidsProcessor:

    def __init__(self):
        self.cm = CentroidsModel()
        self.km = Kinetics()

    def apply(self, data_points):
        cdf = self.cm.fit(possession_array=data_points)
        kdf = self.km.fit(poss_array=data_points)
        return pd.concat((cdf, kdf), axis=1)


class FlattenProcessor:

    def __init__(self, n_players, avg_length):
        self.nm = Normalizer()
        self.km = Kinetics()
        self.n_players = n_players
        self.avg_length = avg_length

    def apply(self, data_points):
        normalized_possession = self.nm.fit(poss_array=data_points, cutoff_size=self.avg_length, number_of_players=self.n_players)
        kdf = self.km.fit(poss_array=normalized_possession)
        flattened_version = flatten(poss_array=normalized_possession, number_of_players=self.n_players)
        return pd.concat((flattened_version, kdf), axis=1)


class CentroidsEmbedding(EmbeddingPreProc):

    def __init__(self, raw_df):
        super().__init__(raw_df=raw_df)

    def process(self, aggregation_key):
        return super().process(aggregation_key=aggregation_key, coordinate_processor=CentroidsProcessor())


class FlattenEmbedding(EmbeddingPreProc):

    def __init__(self, raw_df, avg_length, n_players=6):
        super().__init__(raw_df=raw_df)
        self.avg_length = avg_length
        self.n_players = n_players

    def process(self, aggregation_key):
        coordinate_processor = FlattenProcessor(n_players=self.n_players, avg_length=self.avg_length)
        return super().process(aggregation_key=aggregation_key, coordinate_processor=coordinate_processor)