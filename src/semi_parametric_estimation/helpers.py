import numpy as np
import numpy as np
from scipy.special import logit

import sklearn.linear_model as lm

def calibrate_g(g, t):
    """
    Improve calibation of propensity scores by fitting 1 parameter (temperature) logistic regression on heldout data

    :param g: raw propensity score estimates
    :param t: treatment assignments
    :return:
    """

    logit_g = logit(g).reshape(-1,1)
    calibrator = lm.LogisticRegression(fit_intercept=False, C=1e6)  # no intercept or regularization
    calibrator.fit(logit_g, t)
    calibrated_g = calibrator.predict_proba(logit_g)[:,1]
    return calibrated_g


def truncate_by_g(attribute, g, level=0.01):
    keep_these = np.logical_and(g >= level, g <= 1.-level)

    return attribute[keep_these]


def truncate_all_by_g(q_t0, q_t1, g, t, y, truncate_level=0.05):
    """
    Helper function to clean up nuisance parameter estimates.

    """

    orig_g = np.copy(g)

    q_t0 = truncate_by_g(np.copy(q_t0), orig_g, truncate_level)
    q_t1 = truncate_by_g(np.copy(q_t1), orig_g, truncate_level)
    g = truncate_by_g(np.copy(g), orig_g, truncate_level)
    t = truncate_by_g(np.copy(t), orig_g, truncate_level)
    y = truncate_by_g(np.copy(y), orig_g, truncate_level)

    return q_t0, q_t1, g, t, y



def cross_entropy(y, p):
    return -np.mean((y*np.log(p) + (1.-y)*np.log(1.-p)))


def mse(x, y):
    return np.mean(np.square(x-y))
