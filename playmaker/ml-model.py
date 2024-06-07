import os
import time

from pandas.core.dtypes.common import is_numeric_dtype, is_string_dtype
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from thinker.datasets import centroid_handball_possession, flattened_handball_possessions, \
    latent_space_encoded_possession, DatasetConfig
from thinker.ml.experimenter import ml_methods_configs, grid_search_estimator, train_and_test

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NeighbourhoodCleaningRule


def get_dataset(config, mode, target_attribute, include_sequence=False, phase="AT", timesteps=35, lse_size=128):
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

    return data, classes, games


def transform_dataset(data, classes):
    label_encoder = LabelEncoder()
    std_encoder = MinMaxScaler()  # StandardScaler()
    X = data

    na_columns = ["sequences", "passive_alert", "tactical_situation", "misses", "prev1_score_diff", "prev2_score_diff"]
    for na_column in na_columns:
        if na_column in X.columns:
            X[na_column].fillna(0, inplace=True)

    if "throw_zone" in X.columns:
        X["throw_zone"].fillna("", inplace=True)
        X["offense_type"].fillna("", inplace=True)

    for column in X.columns:
        if is_numeric_dtype(X[column]):
            X[[column]] = std_encoder.fit_transform(X[[column]])
        if is_string_dtype(X[column]):
            X[column] = label_encoder.fit_transform(X[column])

    X.dropna(how="any", inplace=True)
    y = label_encoder.fit_transform(classes)

    return X, y


def dataset_resample(X, y, action):
    if action == "down":
        rus = NeighbourhoodCleaningRule(sampling_strategy="majority")
        X, y = rus.fit_resample(X, y)
    elif action == "up":
        rus = SMOTE()
        X, y = rus.fit_resample(X, y)
    return X, y


if __name__ == "__main__":
    mode = "centroids"  # centroids, flatten, ls
    experiment_id = time.time()
    k_fold_strategy = "group"
    resample_action = None  # up down None

    n_splits = 9

    include_sequences = True

    output_path = os.path.join("outputs", f"{experiment_id}")
    os.makedirs(output_path, exist_ok=True)

    config = DatasetConfig(
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
    target_attribute = "organized_game"  # possession_result organized_game
    method_name = "lgbm"  # lgbm randomforest knn

    data, classes, games = get_dataset(config=config,
                                       mode=mode,
                                       lse_size=256,
                                       include_sequence=include_sequences,
                                       target_attribute=target_attribute)

    X, y = transform_dataset(data=data, classes=classes)

    print(X.shape)
    print(X.columns)

    k_folder = StratifiedKFold(n_splits=n_splits, shuffle=True)
    if k_fold_strategy != "stratified":
        k_folder = GroupKFold(n_splits=n_splits)

    method_configs = ml_methods_configs()
    fold_id = 0
    acc_score = 0
    for train_index, test_index in k_folder.split(X=X, y=y, groups=games):
        X_train = X.iloc[train_index, :]
        y_train = y[train_index]
        X_test = X.iloc[test_index, :]
        y_test = y[test_index]

        print(X_train.shape)

        X_train, y_train = dataset_resample(X=X_train, y=y_train, action=resample_action)

        estimator = None
        if method_name == "lgbm":
            estimator = HistGradientBoostingClassifier()
        elif method_name == "randomforest":
            estimator = RandomForestClassifier()
        elif method_name == "knn":
            estimator = KNeighborsClassifier()
        else:
            raise Exception(f"Wrong method: {method_name}")

        method_config = method_configs[method_name]
        model = grid_search_estimator(estimator=estimator, params=method_config["params"],
                                      name=f"{method_name}-{fold_id}", output_path=output_path,
                                      X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                                      doTest=False)
        fscore = train_and_test(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                                fold_id=fold_id,
                                model=model,
                                name=method_name,
                                output_path=output_path)

        acc_score += fscore
        fold_id += 1

    avg_score = acc_score / fold_id
    print(f"Avg score for {fold_id} folds is {avg_score}")
