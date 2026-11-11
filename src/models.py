"""
src/models.py
-------------
Model definitions, hyperparameter grids, and GridSearchCV wrappers.
Every model is wrapped in an sklearn Pipeline with a preprocessor.
"""

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from src.preprocessing import build_preprocessor

RANDOM_STATE = 42

# -----------------------------------------------------------------------
# Hyperparameter grids (prefixed with 'clf__' for pipeline compat)
# -----------------------------------------------------------------------
PARAM_GRIDS = {
    "LogisticRegression": {
        "clf__C": [0.01, 0.1, 1, 10, 100],
        "clf__solver": ["liblinear", "lbfgs"],
    },
    "KNN": {
        "clf__n_neighbors": [3, 5, 7, 9, 11, 15],
        "clf__weights": ["uniform", "distance"],
        "clf__metric": ["euclidean", "manhattan"],
    },
    "DecisionTree": {
        "clf__max_depth": [2, 3, 4, 5, 6, None],
        "clf__min_samples_split": [2, 5, 10],
        "clf__min_samples_leaf": [1, 2, 4],
        "clf__criterion": ["gini", "entropy"],
    },
    "SVM": {
        "clf__C": [0.1, 1, 10, 100],
        "clf__kernel": ["linear", "rbf"],
        "clf__gamma": ["scale", 0.01, 0.1],
    },
    "RandomForest": {
        "clf__n_estimators": [100, 200, 300],
        "clf__max_depth": [3, 5, 8, None],
        "clf__min_samples_split": [2, 5],
        "clf__max_features": ["sqrt", "log2"],
    },
}

ESTIMATORS = {
    "LogisticRegression": LogisticRegression(
        max_iter=1000, random_state=RANDOM_STATE
    ),
    "KNN": KNeighborsClassifier(),
    "DecisionTree": DecisionTreeClassifier(random_state=RANDOM_STATE),
    "SVM": SVC(probability=True, random_state=RANDOM_STATE),
    "RandomForest": RandomForestClassifier(random_state=RANDOM_STATE),
}


def build_pipeline(model_name, feature_set):
    """Build a Pipeline(preprocessor + classifier) for a given model."""
    preprocessor = build_preprocessor(feature_set)
    clf = ESTIMATORS[model_name]
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("clf", clf),
    ])
    return pipeline


def tune_model(model_name, feature_set, X_train, y_train, verbose=1):
    """
    Run GridSearchCV on training data and return the fitted GridSearchCV object.
    Uses StratifiedKFold(5), scoring='f1', refit=True.
    """
    pipeline = build_pipeline(model_name, feature_set)
    param_grid = PARAM_GRIDS[model_name]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring="f1",
        refit=True,
        n_jobs=-1,
        verbose=verbose,
        return_train_score=True,
    )
    grid_search.fit(X_train, y_train)

    if verbose:
        print(f"\n[{model_name}] Best params: {grid_search.best_params_}")
        print(f"[{model_name}] Best CV F1:  {grid_search.best_score_:.4f}")

    return grid_search

