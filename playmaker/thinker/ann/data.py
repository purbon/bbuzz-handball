import keras
import numpy as np


class DataGameGenerator(keras.utils.Sequence):

    def __init__(self, df,
                 timesteps=60,
                 batch_size=10,
                 start=0,
                 end=-1,
                 display_labels=False,
                 augment=False,
                 normalizer=False,
                 game_phase=None):
        self.batch_size = batch_size
        self.normalizer = normalizer
        self.timesteps = timesteps
        self.data = []
        self.origins = []
        self.labels = []
        self.truth = []
        self.filter = []
        for i in range(6):
            labels = [f"player_{i}_x",
                      f"player_{i}_y"]  # , f"player_{i}_dist_last_5sec", f"player_{i}_speed_last_5sec" ]
            for label in labels:
                self.filter.append(label)

        group_df = df.groupby(["game", "possession"])
        if game_phase is not None:
            end = -1
        for game, data_points in group_df:
            my_game_phase = data_points["game_phases"].head(1).iloc[0]
            if game_phase is not None and my_game_phase != game_phase:
                continue
            self.origins.append(f"{game[0]}_{game[1]}")
            self.labels.append(data_points["organized_game"].tail(1))
            possession = data_points[self.filter]
            np_data, np_truth = self.map_a_possession(possession=possession, timesteps=timesteps)
            self.data.append(np_data)
            self.truth.append(np_truth)
            if augment:
                augmented_possession = self.rotate_a_possession(possession=possession)
                np_data, np_truth = self.map_a_possession(possession=augmented_possession, timesteps=timesteps)
                self.data.append(np_data)
                self.truth.append(np_truth)

        self.truth = np.array(self.data[start:end])
        self.data = np.array(self.data[start:end])
        self.origins = np.array(self.origins[start:end])
        self.labels = np.array(self.labels)

        self.indices = np.arange(self.data.shape[0])

        if display_labels:
            for origin in self.origins:
                print(origin)
        self.numPossessions = len(self.data)
        self.on_epoch_end()

    def map_a_possession(self, possession, timesteps):
        np_array = possession.to_numpy()
        np_data = np_array  # [:-1]
        np_truth = np_array[-1:]
        if np_data.shape[0] < timesteps:
            diff = timesteps - np_data.shape[0]
            np_data = np.pad(np_data, ((0, diff), (0, 0)), 'constant', constant_values=(0,))
        elif np_data.shape[0] > timesteps:
            np_data = np_data[0:timesteps]
        return np_data, np_truth

    def rotate_a_possession(self, possession):
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
        df = rotated[new_columns]

        def column_rename(column_name):
            if column_name.endswith('_y'):
                return column_name.replace('_y', '_x')
            else:
                return column_name.replace('_x', '_y')

        df.rename(lambda x: column_rename(x), axis='columns', inplace=True)
        return df

    def __getitem__(self, index):
        # lower_bound = index * self.batch_size
        # upper_bound = (index + 1) * self.batch_size
        inds = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        x = self.data[inds].astype('float32')  # data
        y = self.truth[inds].astype('float32')  # ground truth
        if self.normalizer:
            x = (np.rint(x * 10_000)).astype('int')
            x = x.reshape((self.batch_size, self.timesteps * len(self.filter)))
        return x, y

    def __getorigins__(self, index):
        # lower_bound = index * self.batch_size
        # upper_bound = (index + 1) * self.batch_size
        inds = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        return self.origins[inds]

    def __getlabels__(self, index):
        inds = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        return self.labels[inds].flatten()

    def on_epoch_end(self):
        np.random.shuffle(self.indices)

    def __len__(self):
        return int(np.ceil(self.numPossessions / float(self.batch_size)))