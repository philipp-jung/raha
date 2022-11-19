import math
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
import warnings
warnings.filterwarnings('ignore')  # "error", "ignore", "always", "default", "module" or "once"


def cross_validated_estimator(x_train, y_train):
    """
    Return the best estimator resulting from cross-validation.
    Es passiert oft, dass mehrere Estimatoren einen roc-aur score von 1.0 haben. Dann wird z.B. ein LOGR mit C=1.0
    zurückgegeben.
    Im Ergebnis reinigt das aber schlechter, als wenn ich immer mit ABC(n_estimators=100) arbeite.
    Darum lasse ich zuerst das baseline-Modell durchlaufen. Wenn das schon einen perfekten Score erzielt, wird es
    direkt genommen.
    """
    cv = 2 if sum(y_train) < 4 else math.floor(math.log2(sum(y_train)))
    params = {'n_estimators': [10, 100, 200],
              'learning_rate': [0.1, 1]}

    grid_search = GridSearchCV(estimator=AdaBoostClassifier(),
                               param_grid=params,
                               cv=cv,
                               n_jobs=1,
                               scoring='precision')
    gs_clf = grid_search.fit(x_train, y_train)
    return gs_clf
