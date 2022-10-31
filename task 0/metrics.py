import numpy as np

def TP(probs, ground, threshold):
    return np.sum((probs >= threshold) & (ground == 1))

def FP(probs, ground, threshold):
    return np.sum((probs >= threshold) & (ground == 0))

def TN(probs, ground, threshold):
    return np.sum((probs < threshold) & (ground == 0))

def FN(probs, ground, threshold):
    return np.sum((probs < threshold) & (ground == 1))



def recall(probs, ground, threshold):
    tp = TP(probs, ground, threshold)
    fn = FN(probs, ground, threshold)
    if tp == 0:
        return 0
    return tp/(tp+fn)

def precision(probs, ground, threshold):
    tp = TP(probs, ground, threshold)
    fp = FP(probs, ground, threshold)
    if tp == 0:
        return 0
    return tp/(tp+fp)



def TPR(probs, ground, threshold):
    return recall(probs, ground, threshold)

def FPR(probs, ground, threshold):
    fp = FP(probs, ground, threshold)
    tn=TN(probs, ground, threshold)
    if fp == 0:
        return 0
    return fp/(fp+tn)
