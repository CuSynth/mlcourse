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


def fit_predict(clf, X_train, X_test, y_train, y_test, metric=accuracy_score):
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    return metric(y_test, predictions)