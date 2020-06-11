
import numpy as np

## 给定一些类数据，计算等错误率，
def compute_eer(fnr, fpr):
    """ computes the equal error rate (EER) given FNR and FPR values calculated
        for a range of operating points on the DET curve
    """

    diff_pm_fa = fnr - fpr
    x1 = np.flatnonzero(diff_pm_fa >= 0)[0]
    x2 = np.flatnonzero(diff_pm_fa < 0)[-1]
    a = (fnr[x1] - fpr[x1]) / (fpr[x2] - fpr[x1] - (fnr[x2] - fnr[x1]))
    return fnr[x1] + a * (fnr[x2] - fnr[x1])


def compute_pmiss_pfa(scores, labels):
    """ computes false positive rate (FPR) and false negative rate (FNR)
    given trial scores and their labels. A weights option is also provided
    to equalize the counts over score partitions (if there is such
    partitioning).
    """

    sorted_ndx = np.argsort(scores)
    labels = labels[sorted_ndx]

    tgt = (labels == 1).astype('f8')
    imp = (labels == 0).astype('f8')

    fnr = np.cumsum(tgt) / np.sum(tgt)
    fpr = 1 - np.cumsum(imp) / np.sum(imp)
    return fnr, fpr


def compute_min_cost(scores, labels, p_target=0.01):
    fnr, fpr = compute_pmiss_pfa(scores, labels)
    eer = compute_eer(fnr, fpr)
    min_c = compute_c_norm(fnr, fpr, p_target)
    return eer, min_c


def compute_c_norm(fnr, fpr, p_target, c_miss=1, c_fa=1):
    """ computes normalized minimum detection cost function (DCF) given
        the costs for false accepts and false rejects as well as a priori
        probability for target speakers
    """
    dcf = c_miss * fnr * p_target + c_fa * fpr * (1 - p_target)
    c_det = np.min(dcf)
    c_def = min(c_miss * p_target, c_fa * (1 - p_target))
    return c_det/c_def