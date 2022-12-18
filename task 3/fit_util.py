import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import accuracy_score


def cv_estimate(clf, X, y, n_splits=5, scoring='accuracy', n_jobs=8):
    cv_results = cross_validate(clf, X, y, cv=n_splits, n_jobs=n_jobs, scoring=scoring)
    scores = cv_results['test_score']
    return scores


def fit_validate(clf, X_train, X_test, y_train, y_test, n_splits=5, scoring='accuracy', n_jobs=8):
    clf.fit(X_train, y_train)
    return cv_estimate(clf, X_test, y_test, n_jobs=n_jobs, n_splits=n_splits, scoring=scoring)


def fit_predict(clf, X, y, train_size=5000, test_size=5000, metric=accuracy_score):
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, test_size=test_size, random_state=0)
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    accuracy = metric(y_test, predictions)
    print(f"Accuracy score: {accuracy}")

def gsearch(pipe_clf, X, y, train_size=5000, test_size=5000):
    X_train, _, y_train, _ = train_test_split(X, y, train_size=train_size, test_size=test_size, random_state=0)
    pipe_clf.fit(X_train, y_train)
    
    best_params = pipe_clf[-1].best_params_
    best_score = pipe_clf[-1].best_score_
    
    print(f"Best params: {best_params}")
    print(f"Best score: {best_score}")
    return 