import numpy as np
import pandas as pd
from pandas.core.dtypes.common import is_string_dtype
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

from thinker.meta import Schema


class DatasetConfig:

    def __init__(self,
                 id=0,
                 include_centroids=False,
                 include_distance=False,
                 include_vel=False,
                 include_acl=False,
                 include_offense_metadata=False,
                 include_fk_counts=False,
                 include_scoring=False,
                 include_time=False,
                 include_prev_possession_result=False,
                 include_prev_score_diff=False
                 ):
        self.id = id
        self.include_centroids = include_centroids
        self.include_distance = include_distance
        self.include_vel = include_vel
        self.include_acl = include_acl
        self.include_offense_metadata = include_offense_metadata
        self.include_fk_counts = include_fk_counts
        self.include_scoring = include_scoring
        self.include_time = include_time
        self.include_prev_possession_result = include_prev_possession_result
        self.include_prev_score_diff = include_prev_score_diff


def flattened_handball_possessions(data_home=f"dumps/dataset.h5",
                                   n_players=6,
                                   target_attribute=None,
                                   phase=None,
                                   include_sequences=False,
                                   length=20,
                                   filter=DatasetConfig(),
                                   game_column="game"):
    key = f"flattens" if include_sequences else f"flatten"
    df = pd.read_hdf(path_or_buf=data_home, key=key)
    if phase is not None:
        df = df[df["game_phases"] == phase]

    if target_attribute is None:
        return df, None, None
    else:
        columns = []
        for column in df.columns:
            if column.startswith('player_'):
                columns.append(column)
        filter.include_centroids = False
        columns += __cget_classification_data(filter=filter, n_players=0)
        try:
            columns.remove(target_attribute)
        except ValueError:
            pass
        G = df[game_column]
        return df[columns], df[target_attribute], G


def centroid_handball_possession(data_home=f"dumps/dataset.h5",
                                 n_players=6,
                                 include_sequences=False,
                                 target_attribute=None,
                                 phase=None,
                                 filter=DatasetConfig(),
                                 game_column="game"):
    key = "centroidss" if include_sequences else "centroids"
    df = pd.read_hdf(path_or_buf=data_home, key=key)
    if phase == "AT" and "offense_type" in df.columns:
        df.dropna(subset="offense_type", inplace=True)

    if phase is not None:
        df = df[df["game_phases"] == phase]
    if target_attribute is None:
        return df, None, None
    else:
        y = df[target_attribute]
        x_columns = __cget_classification_data(n_players=n_players, filter=filter)
        try:
            x_columns.remove(target_attribute)
        except ValueError:
            pass
        X = df[x_columns]
        G = df[game_column]
        return X, y, G


def latent_space_encoded_possession(method_type="lstm", lse_size=128,
                                    data_home=f"dumps/dataset.h5",
                                    target_attribute=None,
                                    phase=None,
                                    timesteps=25,
                                    filter=DatasetConfig(),
                                    game_column="game"):
    key_label = f"{method_type}_autoenc_{lse_size}-{timesteps}"
    df = pd.read_hdf(path_or_buf=data_home, key=key_label)
    if phase == "AT" and "offense_type" in df.columns:
        df.dropna(subset="offense_type", inplace=True)

    if phase is not None:
        df = df[df["game_phases"] == phase]
    if target_attribute is None:
        return df, None, None
    else:
        y = df[target_attribute]
        filter.include_centroids = False
        filter.include_distance = False
        filter.include_vel = False
        filter.include_acl = False
        x_columns = __cget_classification_data(n_players=0, filter=filter)

        try:
            x_columns.remove(target_attribute)
        except ValueError:
            pass
        lse_columns = [i for i in range(lse_size)]
        x_columns += lse_columns

        X = df[x_columns]
        X.columns = X.columns.astype(str)
        G = df[game_column]
        return X, y, G




def __cget_classification_data(filter=DatasetConfig(), n_players=6):
    columns = []
    for i in range(n_players):
        if filter.include_centroids:
            columns += [f"p{i}_x_centroid", f"p{i}_y_centroid"]
        if filter.include_distance:
            columns += [f"p{i}_dist_to_center", f"p{i}_dist"]
        if filter.include_vel:
            columns += [f"p{i}_avg_vel", f"p{i}_p90_vel"]
        if filter.include_acl:
            columns += [f"p{i}_avg_acc", f"p{i}_p90_acc"]
    if filter.include_offense_metadata:
        columns += ["offense_type", "misses", "throws", "throw_zone", "tactical_situation", "sequences"]
    if filter.include_centroids:
        columns += ["team_x_centroid", "team_y_centroid"]
    if filter.include_scoring:
        columns += ["score_team_a", "score_team_b"]
    if filter.include_time:
        columns += ["live_possession_duration_in_sec"]
    if filter.include_prev_possession_result:
        columns += ["prev1_possession_result", "prev2_possession_result"]
    if filter.include_prev_score_diff:
        columns += ["prev1_score_diff", "prev2_score_diff"]
    if filter.include_fk_counts:
        columns += ["fk_count",	"pen_count", "tm_count",	"prev_tm",	"post_tm"]
    return list(set(columns))

def raw_handball_possessions(target_class,
                             timesteps,
                             game_phase=None,
                             augment=False,
                             normalizer=False,
                             population_field=None):
    df = pd.read_hdf(path_or_buf=f"dumps/dataset.h5", key="pos")
    df = df[df[Schema.ACTIVE_PLAYERS_COLUMN] == 6]

    scaler = MinMaxScaler()
    label_encoder = LabelEncoder()

    data = []
    truth = []
    filter = []
    games = []
    sensitive_attribute = []
    for i in range(6):
        labels = [f"player_{i}_x",
                  f"player_{i}_y"]
        for label in labels:
            filter.append(label)

    df[filter] = scaler.fit_transform(df[filter])
    df[target_class] = label_encoder.fit_transform(df[target_class])

    def field_encode(df_value):
        if is_string_dtype(df_value):
            return label_encoder.fit_transform(df_value)
        return scaler.fit_transform(df_value)

    if population_field:
        df[population_field].fillna('', inplace=True)
    #    df[population_field] = field_encode(df_value=df[population_field])

    group_df = df.groupby(["game", "possession"])
    for game, data_points in group_df:
        my_game_phase = data_points["game_phases"].head(1).iloc[0]
        if game_phase is not None and my_game_phase != game_phase:
            continue
        possession = data_points[filter]
        games.append(data_points["game"].to_numpy()[-1])
        np_data, np_truth = __map_a_possession(possession=possession, timesteps=timesteps)
        data.append(np_data)
        truth.append(data_points[target_class].tail(1))
        if population_field:
            sensitive_attribute.append(data_points[population_field].tail(1))
        if augment:
            augmented_possession = __rotate_a_possession(possession=possession)
            np_data, np_truth = __map_a_possession(possession=augmented_possession, timesteps=timesteps)
            data.append(np_data)
            truth.append(data_points[target_class].tail(1))
            if population_field:
                sensitive_attribute.append(data_points[population_field].tail(1))
            games.append(data_points["game"].to_numpy()[-1])

    X, y, G = np.array(data), np.array(truth), np.array(games)
    if normalizer:
        X = (np.rint(X * 10_000)).astype('int')
        shape_x = X.shape[0]
        X = X.reshape((shape_x, timesteps * len(filter)))
        y = y.reshape((-1, 1))
        G = G.reshape((-1, 1))
    return X, y, G, np.array(sensitive_attribute)


def __map_a_possession(possession, timesteps):
    np_array = possession.to_numpy()
    np_data = np_array  # [:-1]
    np_truth = np_array[-1:]
    if np_data.shape[0] < timesteps:
        diff = timesteps - np_data.shape[0]
        np_data = np.pad(np_data, ((0, diff), (0, 0)), 'constant', constant_values=(0,))
    elif np_data.shape[0] > timesteps:
        np_data = np_data[0:timesteps]
    return np_data, np_truth


def __rotate_a_possession(possession):
    rotated = possession.copy()
    original_columns = rotated.columns
    new_columns = []
    i = 0
    while i < len(original_columns):
        first_column = original_columns[i + 1]
        second_column = original_columns[i]
        new_columns.append(first_column)
        new_columns.append(second_column)
        i += 2
    return rotated[new_columns]
