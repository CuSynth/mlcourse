import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

def plot_image(ax, image:np.ndarray, cmap=plt.cm.binary_r, title:str=None, label:str=None):
    ax.set_axis_off()
    showed_image = ax.imshow(image.reshape(28, 28), cmap=cmap, interpolation="nearest")
    
    if label is not None:
        ax.text(1, 3, label, bbox={'facecolor': 'white', 'pad': 5})
        
    if title is not None:
        ax.set_title(title)
    
    return showed_image

def plot_samples(X:np.ndarray, y:np.ndarray, cmap=plt.cm.binary_r, rows:int=1, cols:int=5, figsize:tuple=(10, 3)):
    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=figsize, constrained_layout=True)

    for ax, image, label in zip(axes.flat, X, y):
        showed_image = plot_image(ax=ax, image=image, label=str(label), cmap=cmap)
    
    plt.show()


def plot_errors(X: np.ndarray, y: np.ndarray, preds: np.ndarray, num_samples: int=5):
    fig, axes = plt.subplots(nrows=1, ncols=num_samples, figsize=(10, 3))
    
    wrong_preds_indexes = np.where(preds != y)[0]
    
    X_plot:np.ndarray = X[wrong_preds_indexes][:num_samples]
    preds_plot:np.ndarray = preds[wrong_preds_indexes][:num_samples]
    gt_plot = y[wrong_preds_indexes][:num_samples]

    plt.suptitle("Wrong predictions of the classifier")
    for ax, img, label, gt in zip(axes, X_plot, preds_plot, gt_plot):
        plot_image(ax, img, label=str(label), title=("gt: " + str(gt)))
    
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(estimator, predictions, y_test, figsize: tuple=(15, 12)):
    cm = confusion_matrix(y_test, predictions, labels=estimator.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=estimator.classes_)
    _, ax = plt.subplots(figsize=figsize)
    disp.plot(ax=ax)