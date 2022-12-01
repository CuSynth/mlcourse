import numpy as np
from sklearn.utils.validation import check_X_y
from sklearn.utils.multiclass import unique_labels

class PotentialClassifier:

    def __init__(self, window_size, kernel_func=lambda x: 1. / (x + 1.) * (np.abs(x) <= 1), num_epoch=10):
        self.kernel_func = kernel_func
        self.window_size = window_size
        self.num_epoch = num_epoch

    def __set_classifier_parameters(self, X, y):
        self.X = X
        self.y = y
        self.classes_ = unique_labels(y)
        self.pots = np.zeros_like(y, dtype=np.int32)
        # self.pots_count = len(self.pots)

    def __get_reference_samples(self):
        self.pot_ids = (self.pots > 0)
        self.X = self.X[self.pot_ids]
        self.y = self.y[self.pot_ids]
        self.pots = self.pots[self.pot_ids]
        self.pots_count = len(self.pots)

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.__set_classifier_parameters(X, y)

        for _ in range(self.num_epoch):
            for i, x_sample in enumerate(self.X):
                if self.predict(x_sample) != self.y[i]:
                    self.pots[i] += 1

        self.__get_reference_samples()

        return self

    def predict(self, X):
        if len(X.shape) < 2:
            X = X.copy()
            X = X[None, :]

        diffs = X[:, None] - self.X[None, :]
        assert diffs.shape[0] == X.shape[0] and diffs.shape[1] == self.X.shape[0] \
            and diffs.shape[2] == X.shape[1] and X.shape[1] == self.X.shape[1]

        dists = np.sqrt(np.sum(diffs ** 2, axis=-1))
        assert dists.shape[0] == X.shape[0] and dists.shape[1] == self.X.shape[0]

        weight = self.pots * self.kernel_func(dists / self.window_size)
        assert weight.shape[0] == X.shape[0] and weight.shape[1] == self.X.shape[0]

        result_predictions = np.zeros((X.shape[0], self.classes_.size))

        for class_ in self.classes_:
            result_predictions[:, class_] = np.sum(weight[:, self.y == class_], axis=-1)

        return np.argmax(result_predictions, axis=-1)

    def get_params(self, deep=True):
        return {"window_size": self.window_size,
                "kernel_func": self.kernel_func,
                "num_epoch" : self.num_epoch}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
