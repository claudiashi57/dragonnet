import copy

from numpy import load

from semi_parametric_estimation.ate import *


def load_truth(replication, knob):
    """
    loading ground truth data
    """

    file_path = '../../result/{}/{}/simulation_outputs.npz'.format(knob, replication)
    data = load(file_path)
    mu_0 = data['mu_0']
    mu_1 = data['mu_1']

    return mu_1, mu_0


def load_data(knob='default', replication=1, model='baseline', train_test='test'):
    """
    loading train test experiment results
    """

    file_path = '../../result/{}/'.format(knob)
    data = load(file_path + '{}/{}/0_replication_{}.npz'.format(replication, model, train_test))

    return data['q_t0'].reshape(-1, 1), data['q_t1'].reshape(-1, 1), data['g'].reshape(-1, 1), \
           data['t'].reshape(-1, 1), data['y'].reshape(-1, 1), data['index'].reshape(-1, 1), data['eps'].reshape(-1, 1)


def get_estimate(q_t0, q_t1, g, t, y_dragon, index, eps, truncate_level=0.01):
    """
    getting the back door adjustment & TMLE estimation
    """

    psi_n = psi_naive(q_t0, q_t1, g, t, y_dragon, truncate_level=truncate_level)
    psi_tmle, psi_tmle_std, eps_hat, initial_loss, final_loss, g_loss = psi_tmle_cont_outcome(q_t0, q_t1, g, t,
                                                                                              y_dragon,
                                                                                              truncate_level=truncate_level)
    return psi_n, psi_tmle, initial_loss, final_loss, g_loss


def make_table(train_test='train', n_replication=50):
    dict = {'tarnet': {'baseline': {'back_door': 0, }, 'targeted_regularization': 0},
            'dragonnet': {'baseline': 0, 'targeted_regularization': 0},
            'nednet': {'baseline': 0, 'targeted_regularization': 0}}
    tmle_dict = copy.deepcopy(dict)

    for knob in ['dragonnet', 'tarnet']:
        for model in ['baseline', 'targeted_regularization']:
            simple_errors, tmle_errors = [], []
            for rep in range(n_replication):
                q_t0, q_t1, g, t, y_dragon, index, eps = load_data(knob, rep, model, train_test)
                a, b = load_truth(rep, knob)
                mu_1, mu_0 = a[index], b[index]

                truth = (mu_1 - mu_0).mean()

                psi_n, psi_tmle, initial_loss, final_loss, g_loss = get_estimate(q_t0, q_t1, g, t, y_dragon, index, eps,
                                                                                 truncate_level=0.01)

                err = abs(truth - psi_n).mean()
                tmle_err = abs(truth - psi_tmle).mean()
                simple_errors.append(err)
                tmle_errors.append(tmle_err)

            dict[knob][model] = np.mean(simple_errors)
            tmle_dict[knob][model] = np.mean(tmle_errors)

    return dict, tmle_dict


def main():
    dict, tmle_dict = make_table()
    print("The back door adjustment result is below")
    print(dict)

    print("the tmle estimator result is this ")
    print(tmle_dict)


if __name__ == "__main__":
    main()
