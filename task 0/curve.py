import numpy as np
import matplotlib.pyplot as plt

from metrics import TPR, FPR, precision, recall


def plot_curve(x, y,  title=None, label=None, xlabel=None, ylabel=None, show_legend=False, plot=None):
    if plot is None:
        plt.figure(dpi=100)
        plt.plot(x, y, label=label)

        plt.title(title)
        plt.xlabel(xlabel, fontsize=12)
        plt.ylabel(ylabel, fontsize=12)

        if show_legend:
            plt.legend()
        plt.show()
        return

    plot.plot(x, y, label=label)
    plot.set_title(title)
    plot.set_xlabel(xlabel, fontsize=12)
    plot.set_ylabel(ylabel, fontsize=12)

    if show_legend:
        plot.legend()


def roc_curve(y_test, y_score):
    fprs, tprs, thresholds = [], [], np.unique(y_score)[::-1]
    thresholds = np.append(1.1, thresholds)

    for th in thresholds:
        tprs.append(TPR(probs=y_score, ground=y_test, threshold=th))
        fprs.append(FPR(probs=y_score, ground=y_test, threshold=th))

    return np.asarray(fprs), np.asarray(tprs), np.asarray(thresholds)


def precision_recall_curve(y_test, y_score):
    precisions, recalls, thresholds = [], [], np.unique(y_score)

    for th in thresholds:
        recalls.append(recall(probs=y_score, ground=y_test, threshold=th))
        precisions.append(precision(probs=y_score, ground=y_test, threshold=th))

    recalls.append(0)
    precisions.append(1)

    return np.asarray(precisions), np.asarray(recalls), np.asarray(thresholds)