#!/usr/bin/env python
import sys
import os
import os.path
import numpy as np

import matplotlib.pyplot as plt
import sklearn
import sklearn.metrics

from evalution import compute_min_cost

def main():

    a = []
    for i in range(500):
        a.append(np.random.random() * 0.6)
    b = []
    for i in range(500):
        b.append(np.random.random() * 0.6 + 0.4)
    print(np.mean(a),np.mean(b))
    c = [i for i in range(500)]
    
    # plt.scatter(c,a)
    # plt.scatter(c,b)
    # plt.show()

    score = []
    label = []
    for i in range(500):
        score.append(a[i])
        label.append(0)

    for i in range(500):
        score.append(b[i])
        label.append(1)


    print(len(score),len(label))
    score = np.array(score)
    label = np.array(label)

    eer,minDcf = compute_min_cost(score, label, p_target=0.01)

    print(eer,minDcf)


    fpr, tpr, threshold = sklearn.metrics.roc_curve(label, score, 1)
    fnr = 1 - tpr
    eer_threshold = threshold[np.nanargmin(np.absolute((fnr - fpr)))]
    eer_1 = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    eer_2 = fnr[np.nanargmin(np.absolute((fnr - fpr)))]
    # return the mean of eer from fpr and from fnr
    eer = (eer_1 + eer_2) / 2
    print("等错误率：",eer,eer_threshold)



if __name__ == '__main__':
    main()