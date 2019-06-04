import numpy as np
from scipy.special import logit, expit
from scipy.optimize import minimize

from .helpers import truncate_all_by_g, cross_entropy, mse


def _perturbed_model(q_t0, q_t1, g, t, q, eps):
    # helper function for psi_tmle

    h1 = t / q - ((1 - t) * g) / (q * (1 - g))
    full_q = (1.0 - t) * q_t0 + t * q_t1
    perturbed_q = full_q - eps * h1

    def q1(t_cf, epsilon):
        h_cf = t_cf * (1.0 / g) - (1.0 - t_cf) / (1.0 - g)
        full_q = (1.0 - t_cf) * q_t0 + t_cf * q_t1  # predictions from unperturbed model
        return full_q - epsilon * h_cf

    psi_init = np.mean(t * (q1(np.ones_like(t), eps) - q1(np.zeros_like(t), eps))) / q
    h2 = (q_t1 - q_t0 - psi_init) / q
    perturbed_g = expit(logit(g) - eps * h2)

    return perturbed_q, perturbed_g


def psi_tmle(q_t0, q_t1, g, t, y, prob_t, truncate_level=0.05):
    """
    Near canonical van der Laan TMLE, except we use a
    1 dimension epsilon shared between the Q and g update models

    """

    q_t0, q_t1, g, t, y = truncate_all_by_g(q_t0, q_t1, g, t, y, truncate_level)

    def _perturbed_loss(eps):
        pert_q, pert_g = _perturbed_model(q_t0, q_t1, g, t, prob_t, eps)
        loss = (np.square(y - pert_q)).mean() + cross_entropy(t, pert_g)
        return loss

    eps_hat = minimize(_perturbed_loss, 0.)
    eps_hat = eps_hat.x[0]

    def q2(t_cf, epsilon):
        h_cf = t_cf * (1.0 / g) - (1.0 - t_cf) / (1.0 - g)
        full_q = (1.0 - t_cf) * q_t0 + t_cf * q_t1  # predictions from unperturbed model
        return full_q - epsilon * h_cf

    psi_tmle = np.mean(t * (q2(np.ones_like(t), eps_hat) - q2(np.zeros_like(t), eps_hat))) / prob_t
    return psi_tmle


def make_one_step_tmle(prob_t, deps_default=0.001):
    "Make a function that computes the 1-step TMLE ala https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4912007/"

    def _perturb_q(q_t0, q_t1, g, t, deps=deps_default):
        h1 = t / prob_t - ((1 - t) * g) / (prob_t * (1 - g))

        full_q = (1.0 - t) * q_t0 + t * q_t1
        perturbed_q = full_q - deps * h1
        # perturbed_q= expit(logit(full_q) - deps*h1)
        return perturbed_q

    def _perturb_g(q_t0, q_t1, g, deps=deps_default):
        h2 = (q_t1 - q_t0 - _psi(q_t0, q_t1, g)) / prob_t
        perturbed_g = expit(logit(g) - deps * h2)
        return perturbed_g

    def _perturb_g_and_q(q0_old, q1_old, g_old, t, deps=deps_default):
        # get the values of Q_{eps+deps} and g_{eps+deps} by using the recursive formula

        perturbed_g = _perturb_g(q0_old, q1_old, g_old, deps=deps)

        perturbed_q = _perturb_q(q0_old, q1_old, perturbed_g, t, deps=deps)
        perturbed_q0 = _perturb_q(q0_old, q1_old, perturbed_g, np.zeros_like(t), deps=deps)
        perturbed_q1 = _perturb_q(q0_old, q1_old, perturbed_g, np.ones_like(t), deps=deps)

        return perturbed_q0, perturbed_q1, perturbed_q, perturbed_g

    def _loss(q, g, y, t):
        # compute the new loss
        q_loss = mse(y, q)
        g_loss = cross_entropy(t, g)
        return q_loss + g_loss

    def _psi(q0, q1, g):
        return np.mean(g*(q1 - q0)) / prob_t

    def tmle(q_t0, q_t1, g, t, y, truncate_level=0.05, deps=deps_default):
        """
        Computes the tmle for the ATT (equivalently: direct effect)

        :param q_t0:
        :param q_t1:
        :param g:
        :param t:
        :param y:
        :param truncate_level:
        :param deps:
        :return:
        """
        q_t0, q_t1, g, t, y = truncate_all_by_g(q_t0, q_t1, g, t, y, truncate_level)

        eps = 0.0

        q0_old = q_t0
        q1_old = q_t1
        g_old = g

        # determine whether epsilon should go up or down
        # translated blindly from line 299 of https://github.com/cran/tmle/blob/master/R/tmle.R
        h1 = t / prob_t - ((1 - t) * g) / (prob_t * (1 - g))
        full_q = (1.0 - t) * q_t0 + t * q_t1
        deriv = np.mean(prob_t*h1*(y-full_q) + t*(q_t1 - q_t0 - _psi(q_t0, q_t1, g)))
        if deriv > 0:
            deps = -deps

        # run until loss starts going up
        # old_loss = np.inf  # this is the thing used by Rose' implementation
        old_loss = _loss(full_q, g, y, t)

        while True:
            # print("Psi: {}".format(_psi(q0_old, q1_old, g_old)))

            perturbed_q0, perturbed_q1, perturbed_q, perturbed_g = _perturb_g_and_q(q0_old, q1_old, g_old, t, deps=deps)

            new_loss = _loss(perturbed_q, perturbed_g, y, t)
            # print("new_loss is: ", new_loss, "old_loss is ", old_loss)

            # # if this is the first step, decide whether to go down or up from eps=0.0
            # if eps == 0.0:
            #     _, _, perturbed_q_neg, perturbed_g_neg = _perturb_g_and_q(q0_old, q1_old, g_old, t, deps=-deps)
            #     neg_loss = _loss(perturbed_q_neg, perturbed_g_neg, y, t)
            #
            #     if neg_loss < new_loss:
            #         return tmle(q_t0, q_t1, g, t, y, deps=-1.0 * deps)

            # check if converged
            if new_loss > old_loss:
                if eps == 0.:
                    print("Warning: no update occurred (is deps too big?)")
                return _psi(q0_old, q1_old, g_old), eps
            else:
                eps += deps

                q0_old = perturbed_q0
                q1_old = perturbed_q1
                g_old = perturbed_g

                old_loss = new_loss

    return tmle


def psi_q_only(q_t0, q_t1, g, t, y, prob_t, truncate_level=0.05):
    q_t0, q_t1, g, t, y = truncate_all_by_g(q_t0, q_t1, g, t, y, truncate_level)

    ite_t = (q_t1 - q_t0)[t == 1]
    estimate = ite_t.mean()
    return estimate


def psi_plugin(q_t0, q_t1, g, t, y, prob_t, truncate_level=0.05):
    q_t0, q_t1, g, t, y = truncate_all_by_g(q_t0, q_t1, g, t, y, truncate_level)

    ite_t = g*(q_t1 - q_t0)/prob_t
    estimate = ite_t.mean()
    return estimate


def psi_aiptw(q_t0, q_t1, g, t, y, prob_t, truncate_level=0.05):
    # the robust ATT estimator described in eqn 3.9 of
    # https://www.econstor.eu/bitstream/10419/149795/1/869216953.pdf

    q_t0, q_t1, g, t, y = truncate_all_by_g(q_t0, q_t1, g, t, y, truncate_level)
    estimate = (t*(y-q_t0) - (1-t)*(g/(1-g))*(y-q_t0)).mean() / prob_t

    return estimate


def psi_very_naive(t, y):
    return y[t == 1].mean() - y[t == 0].mean()


def att_estimates(q_t0, q_t1, g, t, y, prob_t, truncate_level=0.05):

    one_step_tmle = make_one_step_tmle(prob_t, deps_default=0.0001)

    very_naive = psi_very_naive(t,y)
    q_only = psi_q_only(q_t0, q_t1, g, t, y, prob_t, truncate_level)
    plugin = psi_plugin(q_t0, q_t1, g, t, y, prob_t, truncate_level)
    aiptw = psi_aiptw(q_t0, q_t1, g, t, y, prob_t, truncate_level)
    one_step_tmle = one_step_tmle(q_t0, q_t1, g, t, y, truncate_level)  # note different signature

    estimates = {'very_naive': very_naive, 'q_only': q_only, 'plugin': plugin, 'one_step_tmle': one_step_tmle, 'aiptw': aiptw}

    return estimates
