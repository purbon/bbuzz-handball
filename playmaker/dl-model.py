import os
import time

import keras
import numpy as np
from keras_tuner.src.backend.io import tf
from pandas.core.dtypes.common import is_numeric_dtype, is_string_dtype
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, GroupKFold, LeaveOneGroupOut, train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from thinker.ann.models import get_ann_model
from thinker.datasets import centroid_handball_possession, flattened_handball_possessions, \
    latent_space_encoded_possession, DatasetConfig, raw_handball_possessions
from thinker.ml.experimenter import ml_methods_configs, grid_search_estimator, train_and_test

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NeighbourhoodCleaningRule, RandomUnderSampler


def get_dataset(config, mode, target_attribute, normalizer=False, keras_config=None, include_sequence=False, phase="AT", timesteps=25, lse_size=128):
    data, classes, games = None, None, None
    if mode == "centroids":
        data, classes, games = centroid_handball_possession(target_attribute=target_attribute,
                                                            phase=phase,
                                                            include_sequences=include_sequence,
                                                            filter=config)
    if mode == "flatten":
        data, classes, games = flattened_handball_possessions(target_attribute=target_attribute,
                                                              length=timesteps,
                                                              phase=phase,
                                                              include_sequences=include_sequence,
                                                              filter=config)
    if mode == "ls":
        data, classes, games = latent_space_encoded_possession(method_type="lstm",
                                                               lse_size=lse_size,
                                                               phase=phase,
                                                               timesteps=timesteps,
                                                               target_attribute=target_attribute,
                                                               filter=config)
    if mode == "raw":
        data, classes, games, _ = raw_handball_possessions(target_class=target_attribute,
                                                           timesteps=keras_config.timesteps,
                                                           game_phase=phase,
                                                           augment=keras_config.do_augment,
                                                           normalizer=normalizer)

    return data, classes, games


def transform_dataset(data, classes):
    label_encoder = LabelEncoder()
    std_encoder = MinMaxScaler()  # StandardScaler()
    X = data
    #na_columns = ["sequences", "passive_alert", "tactical_situation", "misses", "throws"]
    #for na_column in na_columns:
    #    if na_column in X.columns:
    #        X[na_column].fillna(0, inplace=True)

    if "throw_zone" in X.columns:
        X["throw_zone"].fillna("", inplace=True)
    #if "offense_type" in X.columns:
    #    X["offense_type"].fillna("", inplace=True)

    for column in X.columns:
        if is_numeric_dtype(X[column]):
            if X[column].isnull().any():
                X[column].fillna(0, inplace=True)
            X[[column]] = std_encoder.fit_transform(X[[column]])
        else:
            print(column)
            if X[column].isnull().any():
                X[column].fillna("", inplace=True)
            X[column] = label_encoder.fit_transform(X[column])

    y = label_encoder.fit_transform(classes)

    return X.astype(float), y


def dataset_resample(X, y, action):
    if action == "down":
        #rus = NeighbourhoodCleaningRule(sampling_strategy="not minority", n_neighbors=3)
        rus = RandomUnderSampler(sampling_strategy="not minority")
        X, y = rus.fit_resample(X, y)
    elif action == "up":
        rus = SMOTE(sampling_strategy=1)
        X, y = rus.fit_resample(X, y)
    return X, y

class KerasConfig:

    def __init__(self,
                 id,
                 timesteps,
                 batch_size,
                 epochs,
                 kfold_strategy="group",
                 n_splits=9,
                 do_augment=False,
                 class_action="none"):
        self.id = id
        self.timesteps = timesteps
        self.batch_size = batch_size
        self.epochs = epochs
        self.kfold_strategy = kfold_strategy
        self.n_splits = n_splits
        self.do_augment = do_augment
        self.class_action = class_action

    def to_dict(self):
        data = {
            "id": self.id,
            "timesteps": self.timesteps,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "kfoldstrategy": self.kfold_strategy,
            "n_splits": self.n_splits,
            "do_augment": self.do_augment,
            "class_action": self.class_action
        }
        return data

    def __str__(self):
        return str(self.to_dict())


if __name__ == "__main__":
    mode = "raw"
    experiment_id = time.time()
    k_fold_strategy = "group"
    resample_action = None # down up None
    method_name = "dense" # dense dense2 lstm2 transformer
    target_attribute = "organized_game" # organized_game possession_result

    n_splits = 9

    include_sequences = False

    output_path = os.path.join("outputs", "dl", f"{experiment_id}")
    os.makedirs(output_path, exist_ok=True)

    data_config = DatasetConfig(
        include_centroids=True,
        include_distance=True,
        include_vel=False,
        include_acl=False,
        include_fk_counts=False,
        include_offense_metadata=False,
        include_scoring=False,
        include_time=False,
        include_prev_possession_result=False,
        include_prev_score_diff=False
    )

    config = KerasConfig(
        id=1,
        epochs=50,
        batch_size=32,
        timesteps=30,
        do_augment=False
    )

    normalizer = (method_name != "lstm2" and method_name != "transformer" and mode == "raw")

    data, classes, games = get_dataset(config=data_config,
                                       mode=mode,
                                       lse_size=256,
                                       timesteps=35,
                                       normalizer=normalizer,
                                       keras_config=config,
                                       target_attribute=target_attribute)

    if mode != "raw":
        X, y = transform_dataset(data=data, classes=classes)
        print(X.shape)
        print(X.columns)
    else:
        X, y = data, classes

    k_folder = StratifiedKFold(n_splits=n_splits, shuffle=True)
    if k_fold_strategy != "stratified":
        k_folder = GroupKFold(n_splits=n_splits)

    fold_id = 0
    acc_scores = {"auc": 0, "precision": 0, "recall": 0, "f1": 0}

    callbacks = [keras.callbacks.EarlyStopping(patience=10, monitor="loss", restore_best_weights=True)]


    for train_index, test_index in k_folder.split(X=X, y=y, groups=games):
        X_train = X.iloc[train_index, :] if mode != "raw" else X[train_index, :]
        y_train = y[train_index]
        X_test = X.iloc[test_index, :] if mode != "raw" else X[test_index, :]
        y_test = y[test_index]

        print(X_train.shape)

        X_train, y_train = dataset_resample(X=X_train, y=y_train, action=resample_action)
        input_size = X.shape[1]
        model = get_ann_model(model_type=method_name, timesteps=config.timesteps, n_fields=12, input_size=input_size)
        model.compile(
            loss="binary_crossentropy",
            optimizer='adam',
            # optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=1e-4),
            metrics=["binary_accuracy",
                     tf.keras.metrics.AUC(from_logits=False),
                     tf.keras.metrics.Precision(),
                     tf.keras.metrics.Recall()],
        )
        # model.summary()

        model.fit(
            X_train,
            y_train,
            epochs=config.epochs,
            batch_size=config.batch_size,
            verbose=0
        )

        score = model.evaluate(X_test,
                               y_test,
                               batch_size=config.batch_size,
                               verbose=0)
        print(model.metrics_names)
        print(score)

        precision_num = score[3]
        recall_num = score[4]
        if (precision_num + recall_num) == 0:
            f1_num = 0
        else:
            f1_num = 2 * (precision_num * recall_num) / (precision_num + recall_num)
        auc_num = score[1]

        acc_scores["auc"] += auc_num
        acc_scores["precision"] += precision_num
        acc_scores["recall"] += recall_num
        acc_scores["f1"] += f1_num

        with open(f"{output_path}/{method_name}-train_and_test.txt", 'a+') as f:
            print(f"Model {method_name} Precision {precision_num} Recall {recall_num} F1-score {f1_num} "
                  f"FoldId={fold_id} BinAcu={auc_num}",
                  file=f)

        fold_id += 1

    avg_auc = acc_scores["auc"] / fold_id
    avg_prec = acc_scores["precision"] / fold_id
    avg_rec = acc_scores["recall"] / fold_id
    avg_f1 = acc_scores["f1"] / fold_id

    with open(f"{output_path}/{method_name}-avg-score.txt", 'a+') as f:
        print(f"Avg score for {fold_id} folds is f1={avg_f1}, precision={avg_prec}, recall={avg_rec}, BinAcu={avg_auc}",
              file=f)
    print(f"Avg score for {fold_id} folds is {avg_f1}")

