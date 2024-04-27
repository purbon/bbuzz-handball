from sklearn.model_selection import GridSearchCV, permutation_test_score


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