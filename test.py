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
    # the threshold of fnr == fpr
    eer_threshold = threshold[np.nanargmin(np.absolute((fnr - fpr)))]
    # theoretically eer from fpr and eer from fnr should be identical but they can be slightly differ in reality
    eer_1 = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    eer_2 = fnr[np.nanargmin(np.absolute((fnr - fpr)))]

    # return the mean of eer from fpr and from fnr
    eer = (eer_1 + eer_2) / 2
    print("等错误率：",eer,eer_threshold)



    ## 使用sklearn中的代码计算EER,判断是否一致。


    # input_dir = sys.argv[1]
    # output_dir = sys.argv[2]
    #
    # submit_dir = os.path.join(input_dir, 'res')
    # reference_dir = os.path.join(input_dir, 'ref')
    #
    # if not os.path.isdir(submit_dir):
    #     print("%s doesn't exist" % submit_dir)
    #     exit(1)
    #
    # if os.path.isdir(reference_dir):
    #     if not os.path.exists(output_dir):
    #         os.makedirs(output_dir)
    #
    #     progress = np.loadtxt(os.path.join(reference_dir, "progress.txt"), dtype=int)
    #     print("Reading progress indices are finished.")
    #     keys = np.loadtxt(os.path.join(reference_dir, "keys.txt"), dtype=int)[progress]
    #     print("Reading keys are finished.")
    #     scores = np.loadtxt(os.path.join(submit_dir, "answer.txt"), dtype=float)[progress]
    #     print("Reading scores are finished.")
    #
    #     eer, min_c = compute_min_cost(scores, keys)
    #     print("Scores are calculated.")
    #
    #     output_filename = os.path.join(output_dir, 'scores.txt')
    #     output_file = open(output_filename, 'wb')
    #     output_file.write("mindcf: %f\n" % min_c)
    #     output_file.write("eer: %f\n" % (eer * 100))
    #     output_file.close()
    #     print("Scoring is finished successfully.")
    #     exit(0)
    # else:
    #     print("%s doesn't exist" % truth_dir)
    #     exit(2)


if __name__ == '__main__':
    main()