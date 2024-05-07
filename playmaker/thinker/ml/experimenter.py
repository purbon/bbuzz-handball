import json

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, PrecisionRecallDisplay
from sklearn.model_selection import GridSearchCV, permutation_test_score

import matplotlib.pyplot as plt


def grid_search_estimator(estimator, params, name,
                          X_train, y_train, X_test, y_test,
                          doTest=False, output_path="./", log=True):
    print(f"Running estimator: {estimator}")
    clf = GridSearchCV(estimator, params,
                       cv=5,
                       scoring="f1_weighted",  # f1_weighted
                       refit="f1_weighted",
                       n_jobs=10)
    clf.fit(X=X_train, y=y_train)

    if doTest:
        score, perm_scores, pvalue = permutation_test_score(estimator=clf.best_estimator_,
                                                            X=X_train, y=y_train,
                                                            cv=5,
                                                            scoring="f1_weighted",
                                                            n_permutations=1000)
    if log:
        with open(f"{output_path}/{name}-scores.json", 'a+') as f:
            scores_dict = {
                'best_estimator': str(clf.best_estimator_),
                'best_score': clf.best_score_,
                'test_score': clf.score(X=X_test, y=y_test)
            }
            if doTest:
                scores_dict['perm_test'] = {'score': score, 'pvalue': pvalue}
            json.dump(scores_dict, f)
    return clf

def train_and_test(X_train, y_train, X_test, y_test, fold_id, output_path="./", model=None, name="kfold"):
    if model is None:
        model = RandomForestClassifier(n_estimators=150)
        name = "randomforest"
    model.fit(X=X_train, y=y_train)
    y_pred = model.predict(X=X_test)
    precision_num = precision_score(y_true=y_test, y_pred=y_pred, average='weighted')
    recall_num = recall_score(y_true=y_test, y_pred=y_pred, average='weighted')
    f1_num = f1_score(y_true=y_test, y_pred=y_pred, average='weighted')
    auc_num = roc_auc_score(y_true=y_test, y_score=model.predict_proba(X_test)[:, 1], average='weighted')
    with open(f"{output_path}/{name}-train_and_test.txt", 'a+') as f:
        print(f"Model {name} Precision {precision_num} Recall {recall_num} F1-score {f1_num} FoldId={fold_id} Auc={auc_num}",
              file=f)

    dump_precision_recall_curve(model, X_test=X_test, y_test=y_test, name=f"{name}-prc-{fold_id}",
                                output_path=output_path)
    return f1_num


def dump_precision_recall_curve(estimator, X_test, y_test, name, output_path="./"):
    PrecisionRecallDisplay.from_estimator(estimator=estimator, X=X_test, y=y_test)
    plt.savefig(f"{output_path}/{name}.png")

def ml_methods_configs():
    configs = {"knn": {
        'params': {'n_neighbors': [5, 7, 10, 15, 20, 25],
                   'weights': ['distance', 'uniform'],
                   'p': [1, 2, 3]},
        'name': 'knn'
    }, "randomforest": {
        'params': {'n_estimators': [100, 150, 200, 300, 400],
                   'criterion': ["gini", "entropy", "log_loss"],
                   'max_depth': [None, 2, 3, 6]},
        'name': 'randomforest'
    }, "lgbm": {
        "params": {'learning_rate': [0.5, 0.1, 0.05, 0.01, 0.001]},
        "name": "lgbm"
    }}
    return configs