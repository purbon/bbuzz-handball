import numpy as np
import pandas as pd
from scipy.spatial import distance
from scipy.interpolate import CubicSpline
from sklearn.metrics import nan_euclidean_distances


class CentroidsModel:

    def __init__(self):
        self.team_centroid = [0.0, 0.0]
        self.players_centroid = []
        self.players_distances = []

    # "player_{i}_x" "player_{i}_y", for i=[0, num_of_players)
    def fit(self, possession_array, number_of_players=6):
        self.team_centroid = [0.0, 0.0]
        self.players_centroid = []
        self.players_distances = []
        for i in range(number_of_players):
            xc = self.__trajectory_centroid(possession_array[f"player_{i}_x"])
            yc = self.__trajectory_centroid(possession_array[f"player_{i}_y"])
            self.players_centroid.append([xc, yc])
        team_x = np.array([e[0] for e in self.players_centroid])
        team_y = np.array([e[1] for e in self.players_centroid])

        self.team_centroid[0] = self.__trajectory_centroid(X=team_x)
        self.team_centroid[1] = self.__trajectory_centroid(X=team_y)

        for i in range(number_of_players):
            player_xy = self.players_centroid[i]
            self.players_distances.append(distance.euclidean(player_xy, self.team_centroid))
        return self.get_df()

    def get_df(self):
        columns = []
        data = []
        for i in range(len(self.players_centroid)):
            columns += [f"p{i}_x_centroid", f"p{i}_y_centroid", f"p{i}_dist_to_center"]
            data += self.players_centroid[i]
            data.append(self.players_distances[i])
        data += self.team_centroid
        columns += ["team_x_centroid", "team_y_centroid"]
        return pd.DataFrame(np.array([data]), columns=columns)

    @staticmethod
    def __trajectory_centroid(X):
        return np.sum(X) / X.size


class Kinetics:

    def __init__(self):
        self.__reset()

    def __reset(self):
        self.cumdist = []
        self.avg_velocity = []
        self.p90_velocity = []
        self.avg_acceleration = []
        self.p90_acceleration = []

    def fit(self, poss_array, number_of_players=6):
        self.__reset()
        poss_array.reset_index(inplace=True)
        poss_array = poss_array.aggregate(lambda row: self.__append(row=row, df=poss_array, tail=2), axis=1)
        for player_id in range(number_of_players):
            self.cumdist.append(poss_array[f"p{player_id}_dist"].sum())

        x = poss_array.index.values
        for player_id in range(number_of_players):
            try:
                y = poss_array[f"p{player_id}_dist"].cumsum().values
                if len(x) > 1:
                    cs = CubicSpline(x=x, y=y)  # vel = cs(y, 1) -> S' | accel = cs(y, 2) -> S''
                    velocities = np.fromiter((cs(vx, 1) for vx in x), float)
                    accelerations = np.fromiter((cs(vx, 2) for vx in x), float)
                else:
                    velocities = np.array([0.])
                    accelerations = np.array([0.])

                self.avg_velocity.append(np.percentile(a=velocities, q=50))
                self.p90_velocity.append(np.percentile(a=velocities, q=90))
                self.avg_acceleration.append(np.percentile(a=accelerations, q=50))
                self.p90_acceleration.append(np.percentile(a=accelerations, q=90))
            except ValueError:
                print(f"Something happened with player {player_id}")
                self.avg_velocity.append(np.nan)
                self.p90_velocity.append(np.nan)
                self.avg_acceleration.append(np.nan)
                self.p90_acceleration.append(np.nan)

        return self.get_df()

    def get_df(self):
        columns = []
        data = []
        for player_id in range(len(self.cumdist)):
            columns.append(f"p{player_id}_dist")
            data.append(self.cumdist[player_id])
            columns.append(f"p{player_id}_avg_vel")
            data.append(self.avg_velocity[player_id])
            columns.append(f"p{player_id}_p90_vel")
            data.append(self.p90_velocity[player_id])
            columns.append(f"p{player_id}_avg_acc")
            data.append(self.avg_acceleration[player_id])
            columns.append(f"p{player_id}_p90_acc")
            data.append(self.p90_acceleration[player_id])
        return pd.DataFrame(np.array([data]), columns=columns)

    def __append(self, row, df, tail=5):
        table = df[max(row.name - tail + 1, 0):row.name + 1]
        target = table.tail(1).iloc[0]
        origin = table.head(1).iloc[0]

        for player_id in range(6):
            v = [origin[f"player_{player_id}_x"], origin[f"player_{player_id}_y"]]
            u = [target[f"player_{player_id}_x"], target[f"player_{player_id}_y"]]
            # row[f"p{player_id}_dist"] = distance.euclidean(u=u, v=v)
            x = np.array(u).flatten().reshape(1, -1)
            y = np.array(v).flatten().reshape(1, -1)
            try:
                row[f"p{player_id}_dist"] = nan_euclidean_distances(X=x, Y=y)[0][0]
            except ValueError:
                print(x, y)
        return row


class Normalizer:

    def __init__(self):
        pass

    def fit(self, poss_array, cutoff_size=20, number_of_players=6):
        """
        A sampler game possession normalizer
        :param poss_array : np.array()
            A matrix containing the multivariate representation of the possession
        :param cutoff_size : int
            The maximum size the possession results would have, if the actual data is bigger
            the result will be cut, if lower, an interpolation method is going to be used and sampled
            at the requested rate.  (default is 20, number of frames)
        :param number_of_players : int, optional
            The number of active players in the possession (default is 6)
        :return:
        """
        columns = []
        for i in range(number_of_players):
            columns += [f"player_{i}_x", f"player_{i}_y"]
        poss_array = poss_array[columns]
        nrows = poss_array.shape[0]
        if nrows > cutoff_size:  # should cut the possession to the maximum size
            if isinstance(poss_array, pd.DataFrame):
                poss_array = poss_array[:cutoff_size]
            else:
                poss_array = poss_array[:cutoff_size, :]
        elif nrows < cutoff_size:  # should be using interpolation to have as much required data points
            # target shape will be (cutoff_size, number_of_players*2)
            m = np.zeros((cutoff_size, number_of_players * 2))
            for i in range(number_of_players):
                t = range(cutoff_size)
                px = poss_array[f"player_{i}_x"].values
                py = poss_array[f"player_{i}_y"].values
                m[:, 2 * i] = resample(x=t, y=px, cutoff_size=cutoff_size)
                m[:, 2 * i + 1] = resample(x=t, y=py, cutoff_size=cutoff_size)
            poss_array = pd.DataFrame(data=m, columns=columns)

        return poss_array


def flatten(poss_array, number_of_players):
    columns = []
    n_rows = poss_array.shape[0]
    for n_player in range(number_of_players):
        for i in range(n_rows):
            columns += [f"player_{n_player}_x{i}", f"player_{n_player}_y{i}"]
    m = np.zeros((1, n_rows * 2 * number_of_players))
    for n_player in range(number_of_players):
        for i in range(n_rows):
            pointer = 2 * i + (n_player * 2 * n_rows)
            next_pointer = pointer + 1
            m[0][pointer] = poss_array[f"player_{n_player}_x"].iloc[i]
            m[0][next_pointer] = poss_array[f"player_{n_player}_y"].iloc[i]
    return pd.DataFrame(data=m, columns=columns)


def resample(x, y, cutoff_size):
    if len(y) < 2:
        y = np.array([0, y[0]])
    cs = CubicSpline(x=range(len(y)), y=y, extrapolate='periodic')
    u = np.linspace(0, cutoff_size, cutoff_size)
    return cs(u)
    # return [cs(n) for n in u]
