from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


def train_logistic_regression(X_train, y_train, C=1.0):
    model = LogisticRegression(max_iter=2000, random_state=42, C=C)
    model.fit(X_train, y_train)
    return model


def train_svm(X_train, y_train, C=1.0, kernel="rbf", gamma="scale"):
    model = SVC(probability=True, random_state=42, C=C, kernel=kernel, gamma=gamma)
    model.fit(X_train, y_train)
    return model


def train_decision_tree(X_train, y_train, max_depth=None, min_samples_split=2, criterion="gini"):
    model = DecisionTreeClassifier(
        random_state=42,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        criterion=criterion
    )
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train, n_estimators=100, max_depth=None, min_samples_split=2):
    model = RandomForestClassifier(
        random_state=42,
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split
    )
    model.fit(X_train, y_train)
    return model


def tune_logistic_regression(X_train, y_train):
    param_grid = {
        "C": [0.01, 0.1, 1, 10],
        "solver": ["lbfgs"]
    }

    grid = GridSearchCV(
        LogisticRegression(max_iter=2000, random_state=42),
        param_grid,
        cv=5,
        scoring="accuracy",
        n_jobs=1
    )
    grid.fit(X_train, y_train)
    return grid.best_estimator_, grid.best_params_


def tune_svm(X_train, y_train):
    param_grid = {
        "C": [0.1, 1, 10],
        "kernel": ["linear", "rbf"],
        "gamma": ["scale", "auto"]
    }

    grid = GridSearchCV(
        SVC(probability=True, random_state=42),
        param_grid,
        cv=5,
        scoring="accuracy",
        n_jobs=1
    )
    grid.fit(X_train, y_train)
    return grid.best_estimator_, grid.best_params_


def tune_decision_tree(X_train, y_train):
    param_grid = {
        "max_depth": [3, 5, 10, None],
        "min_samples_split": [2, 5, 10],
        "criterion": ["gini", "entropy"]
    }

    grid = GridSearchCV(
        DecisionTreeClassifier(random_state=42),
        param_grid,
        cv=5,
        scoring="accuracy",
        n_jobs=1
    )
    grid.fit(X_train, y_train)
    return grid.best_estimator_, grid.best_params_


def tune_random_forest(X_train, y_train):
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5]
    }

    grid = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid,
        cv=5,
        scoring="accuracy",
        n_jobs=1
    )
    grid.fit(X_train, y_train)
    return grid.best_estimator_, grid.best_params_