import sklearn.linear_model
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

def cross_validated_estimator(x_train, y_train):
    """
    Return the best estimator resulting from cross-validation.
    Es passiert oft, dass mehrere Estimatoren einen roc-aur score von 1.0 haben. Dann wird z.B. ein LOGR mit C=1.0
    zurückgegeben.
    Im Ergebnis reinigt das aber schlechter, als wenn ich immer mit ABC(n_estimators=100) arbeite.
    Darum lasse ich zuerst das baseline-Modell durchlaufen. Wenn das schon einen perfekten Score erzielt, wird es
    direkt genommen.
    """
    cv = 2 if sum(y_train) < 4 else 3
    classifiers = {
        'baseline': {
            'name': 'Baseline',
            'estimator': sklearn.ensemble.AdaBoostClassifier(),
            'parameters': {
                'n_estimators': [100],
            }
        },
        'abc': {
            'name': 'AdaBoost Classifier',
            'estimator': sklearn.ensemble.AdaBoostClassifier(),
            'parameters': {
                'n_estimators': [10, 20, 50, 200],
            }
        },
        'logr': {
            'name': 'Logistic Regression',
            'estimator': sklearn.linear_model.LogisticRegression(),
            'parameters': {
                'C': [00.1, 0.1, 10, 100],
            },
        },
    }

    best_score = 0
    clfs = []
    best_clf = None
    for classifier in classifiers:
        est = classifiers[classifier]['estimator']
        params = classifiers[classifier]['parameters']
        # DEBUG: Warum geht scoring='precision' nicht?
        grid_search = GridSearchCV(estimator=est, param_grid=params, cv=cv, n_jobs=1, scoring='f1')
        gs_clf = grid_search.fit(x_train, y_train)
        clfs.append(gs_clf)
        if gs_clf.best_score_ > best_score:
            best_score = gs_clf.best_score_
            best_clf = gs_clf

        if best_clf is None:
            best_clf = clfs[0]
    return best_clf