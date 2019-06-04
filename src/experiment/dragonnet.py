
from experiment.models import *
import os
import glob
import argparse
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import keras.backend as K
from keras.optimizers import rmsprop, SGD, Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau, TerminateOnNaN
from experiment.data import load_treatment_and_outcome, load_and_format_covariates
from experiment.idhp_data import *


def _split_output(yt_hat, t, y, y_scaler, x, index):
    q_t0 = y_scaler.inverse_transform(yt_hat[:, 0].copy())
    q_t1 = y_scaler.inverse_transform(yt_hat[:, 1].copy())
    g = yt_hat[:, 2].copy()

    if yt_hat.shape[1] == 4:
        eps = yt_hat[:, 3][0]
    else:
        eps = np.zeros_like(yt_hat[:, 2])

    y = y_scaler.inverse_transform(y.copy())
    var = "average propensity for treated: {} and untreated: {}".format(g[t.squeeze() == 1.].mean(),
                                                                        g[t.squeeze() == 0.].mean())
    print(var)

    return {'q_t0': q_t0, 'q_t1': q_t1, 'g': g, 't': t, 'y': y, 'x': x, 'index': index, 'eps': eps}


def train_and_predict_dragons(t, y_unscaled, x, targeted_regularization=True, output_dir='',
                              knob_loss=dragonnet_loss_binarycross, ratio=1., dragon=1):
    y_scaler = StandardScaler().fit(y_unscaled)
    y = y_scaler.transform(y_unscaled)
    train_outputs = []
    test_outputs = []
    runs = 25
    for i in range(runs):
        if dragon == 0:
            dragonnet = make_tarnet(x.shape[1], 0.01)
        elif dragon == 1:
            dragonnet = make_dragonnet(x.shape[1], 0.01)

        metrics = [regression_loss, binary_classification_loss, treatment_accuracy, track_epsilon]

        if targeted_regularization:
            loss = make_tarreg_loss(ratio=ratio, dragonnet_loss=knob_loss)
        else:
            loss = knob_loss

        tf.random.set_random_seed(i)
        np.random.seed(i)
        train_index, test_index = train_test_split(np.arange(x.shape[0]), test_size=0.3)
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        t_train, t_test = t[train_index], t[test_index]
        yt_train = np.concatenate([y_train, t_train], 1)

        import time;
        start_time = time.time()

        dragonnet.compile(
            optimizer=Adam(lr=1e-3),
            loss=loss, metrics=metrics)

        adam_callbacks = [
            TerminateOnNaN(),
            EarlyStopping(monitor='val_loss', patience=2, min_delta=0.),
            ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, verbose=1, mode='auto',
                              min_delta=1e-8, cooldown=0, min_lr=0)

        ]

        dragonnet.fit(x_train, yt_train, callbacks=adam_callbacks,
                      validation_split=0.2,
                      epochs=100,
                      batch_size=512, verbose=1)

        sgd_callbacks = [
            TerminateOnNaN(),
            EarlyStopping(monitor='val_loss', patience=40, min_delta=0.),
            ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, verbose=1, mode='auto',
                              min_delta=0., cooldown=0, min_lr=0)]

        # should pick something better!
        sgd_lr = 1e-5
        momentum = 0.9
        dragonnet.compile(optimizer=SGD(lr=sgd_lr, momentum=momentum, nesterov=True), loss=loss,
                          metrics=metrics)
        dragonnet.fit(x_train, yt_train, callbacks=sgd_callbacks,
                      validation_split=0.2,
                      epochs=300,
                      batch_size=512, verbose=1)

        elapsed_time = time.time() - start_time
        print("***************************** elapsed_time is: ", elapsed_time)

        yt_hat_test = dragonnet.predict(x_test)
        yt_hat_train = dragonnet.predict(x_train)

        test_outputs += [_split_output(yt_hat_test, t_test, y_test, y_scaler, x_test, test_index)]
        train_outputs += [_split_output(yt_hat_train, t_train, y_train, y_scaler, x_train, train_index)]
        K.clear_session()

    return test_outputs, train_outputs


def train_and_predict_ned(t, y_unscaled, x, targeted_regularization=True, output_dir='',
                          knob_loss=dragonnet_loss_binarycross, ratio=1., dragon=1):
    y_scaler = StandardScaler().fit(y_unscaled)
    y = y_scaler.transform(y_unscaled)

    train_outputs = []
    test_outputs = []
    runs = 25
    for i in range(runs):
        nednet = make_ned(x.shape[1], 0.01)

        metrics_ned = [ned_loss]
        metrics_cut = [regression_loss]

        tf.random.set_random_seed(i)
        np.random.seed(i)
        train_index, test_index = train_test_split(np.arange(x.shape[0]), test_size=0.3)
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        t_train, t_test = t[train_index], t[test_index]
        yt_train = np.concatenate([y_train, t_train], 1)

        import time;
        start_time = time.time()

        nednet.compile(
            optimizer=Adam(lr=1e-3),
            loss=ned_loss, metrics=metrics_ned)

        adam_callbacks = [
            TerminateOnNaN(),
            EarlyStopping(monitor='val_loss', patience=2, min_delta=0.),
            ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, verbose=1, mode='auto',
                              min_delta=1e-8, cooldown=0, min_lr=0)
        ]

        nednet.fit(x_train, yt_train, callbacks=adam_callbacks,
                   validation_split=0.2,
                   epochs=100,
                   batch_size=512, verbose=1)

        sgd_callbacks = [
            TerminateOnNaN(),
            EarlyStopping(monitor='val_loss', patience=40, min_delta=0.),
            ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, verbose=1, mode='auto',
                              min_delta=0., cooldown=0, min_lr=0)]

        sgd_lr = 1e-5
        momentum = 0.9
        nednet.compile(optimizer=SGD(lr=sgd_lr, momentum=momentum, nesterov=True), loss=ned_loss,
                       metrics=metrics_ned)
        nednet.fit(x_train, yt_train, callbacks=sgd_callbacks,
                   validation_split=0.2,
                   epochs=300,
                   batch_size=512, verbose=1)

        t_hat_test = nednet.predict(x_test)[:, 1]
        t_hat_train = nednet.predict(x_train)[:, 1]

        # cutting the activation layer
        cut_ned = post_cut(nednet, x.shape[1], 0.01)

        cut_ned.compile(
            optimizer=Adam(lr=1e-3),
            loss=dead_loss, metrics=metrics_cut)

        adam_callbacks = [
            TerminateOnNaN(),
            EarlyStopping(monitor='val_loss', patience=2, min_delta=0.),
            ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, verbose=1, mode='auto',
                              min_delta=1e-8, cooldown=0, min_lr=0)
        ]

        cut_ned.fit(x_train, yt_train, callbacks=adam_callbacks,
                    validation_split=0.2,
                    epochs=100,
                    batch_size=512, verbose=1)

        elapsed_time = time.time() - start_time
        print("***************************** elapsed_time is: ", elapsed_time)

        sgd_callbacks = [
            TerminateOnNaN(),
            EarlyStopping(monitor='val_loss', patience=40, min_delta=0.),
            ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, verbose=1, mode='auto',
                              min_delta=0., cooldown=0, min_lr=0)]

        # should pick something better!
        sgd_lr = 1e-5
        momentum = 0.9
        cut_ned.compile(optimizer=SGD(lr=sgd_lr, momentum=momentum, nesterov=True), loss=dead_loss,
                        metrics=metrics_ned)

        cut_ned.fit(x_train, yt_train, callbacks=sgd_callbacks,
                    validation_split=0.2,
                    epochs=300,
                    batch_size=512, verbose=1)

        y_hat_test = cut_ned.predict(x_test)
        y_hat_train = cut_ned.predict(x_train)

        yt_hat_test = np.concatenate([y_hat_test, t_hat_test.reshape(-1, 1)], 1)
        yt_hat_train = np.concatenate([y_hat_train, t_hat_train.reshape(-1, 1)], 1)

        test_outputs += [_split_output(yt_hat_test, t_test, y_test, y_scaler, x_test, test_index)]
        train_outputs += [_split_output(yt_hat_train, t_train, y_train, y_scaler, x_train, train_index)]
        K.clear_session()
    return test_outputs, train_outputs


def run_acic(data_base_dir='../../data/', simulation_output_dir='../../dragonnet/knob/knob_option/',
             knob_loss=dragonnet_loss_binarycross,
             ratio=1., dragon=1, folder='a'):
    covariate_csv = os.path.join(data_base_dir, 'x.csv')
    x_raw = load_and_format_covariates(covariate_csv)
    simulation_dir = os.path.join(data_base_dir, folder)

    simulation_files = sorted(glob.glob("{}/*".format(simulation_dir)))

    for idx, simulation_file in enumerate(simulation_files):
        cf_suffix = "_cf"
        file_extension = ".csv"
        if simulation_file.endswith(cf_suffix + file_extension):
            continue
        ufid = os.path.basename(simulation_file)[:-4]

        t, y, sample_id, x = load_treatment_and_outcome(x_raw, simulation_file)
        ufid_output_dir = os.path.join(simulation_output_dir, str(ufid))

        os.makedirs(ufid_output_dir, exist_ok=True)
        np.savez_compressed(os.path.join(ufid_output_dir, "simulation_outputs.npz"),
                            t=t, y=y, sample_id=sample_id, x=x)

        if dragon == 'nednet':
            train_outputs_dir = os.path.join(ufid_output_dir, 'nednet')
            os.makedirs(train_outputs_dir, exist_ok=True)
            test_outputs, train_outputs = train_and_predict_ned(t, y, x,
                                                                targeted_regularization=False,
                                                                output_dir=train_outputs_dir,
                                                                knob_loss=knob_loss, ratio=ratio, dragon=dragon)
        else:
            for is_targeted_regularization in [True, False]:
                if is_targeted_regularization:
                    train_outputs_dir = os.path.join(ufid_output_dir, 'targeted_regularization')
                else:
                    train_outputs_dir = os.path.join(ufid_output_dir, 'baseline')
            os.makedirs(train_outputs_dir, exist_ok=True)
            test_outputs, train_outputs = train_and_predict_dragons(t, y, x,
                                                                    targeted_regularization=is_targeted_regularization,
                                                                    output_dir=train_outputs_dir,
                                                                    knob_loss=knob_loss, ratio=ratio, dragon=dragon)
        for num, output in enumerate(test_outputs):
            np.savez_compressed(os.path.join(train_outputs_dir, "{}_replication_test.npz".format(num)),
                                **output)

        for num, output in enumerate(train_outputs):
            np.savez_compressed(os.path.join(train_outputs_dir, "{}_replication_train.npz".format(num)),
                                **output)


def run_ihdp(data_base_dir='/Users/claudiashi/data/ihdp_csv', simulation_output_dir='~/result/ihdp/',
             knob_loss=dragonnet_loss_binarycross,
             ratio=1., dragon=1):

    simulation_files = sorted(glob.glob("{}/*.csv".format(data_base_dir)))

    for idx, simulation_file in enumerate(simulation_files):
        simulation_output_dir = os.path.join(simulation_output_dir, str(idx))
        os.makedirs(simulation_output_dir, exist_ok=True)

        x = load_and_format_covariates_ihdp(simulation_file)
        t, y, y_cf, mu_0, mu_1 = load_all_other_crap(simulation_file)
        np.savez_compressed(os.path.join(simulation_output_dir, "simulation_outputs.npz"),
                            t=t, y=y, y_cf=y_cf, mu_0=mu_0, mu_1=mu_1)

        for is_targeted_regularization in [False]:
            print("Is targeted regularization: {}".format(is_targeted_regularization))
            # fit the model, make predictions on test and train sets, and return those to us
            # this defaults to running the procedure on 5 splits
            test_outputs, train_output = train_and_predict_ned(t, y, x,
                                                               targeted_regularization=is_targeted_regularization,
                                                               output_dir=simulation_output_dir,
                                                               knob_loss=knob_loss, ratio=ratio, dragon=dragon)

            if is_targeted_regularization:
                train_output_dir = os.path.join(simulation_output_dir, "targeted_regularization")
            else:
                train_output_dir = os.path.join(simulation_output_dir, "baseline")
            os.makedirs(train_output_dir, exist_ok=True)

            # save the outputs of for each split (1 per npz file)
            for num, output in enumerate(test_outputs):
                np.savez_compressed(os.path.join(train_output_dir, "{}_replication_test.npz".format(num)),
                                    **output)

            for num, output in enumerate(train_output):
                np.savez_compressed(os.path.join(train_output_dir, "{}_replication_train.npz".format(num)),
                                    **output)


def run_exp(data_base_dir='/Users/claudiashi/data/test/',
            simulation_output_dir='/Users/claudiashi/result/experiment/dragonnet', dragon='nednet',
            folder='a', dataset='acic'):
    if dataset == 'acic':
        run_acic(data_base_dir=data_base_dir, simulation_output_dir=simulation_output_dir, folder=folder,
                 knob_loss=dragonnet_loss_binarycross, dragon=dragon)
    elif dataset == 'ihdp':
        run_ihdp(data_base_dir=data_base_dir, simulation_output_dir=simulation_output_dir,
                 knob_loss=dragonnet_loss_binarycross, dragon=dragon)


def turn_knob(data_base_dir='/Users/claudiashi/data/test/', knob='dragonnet', folder='a',
              output_base_dir=' /Users/claudiashi/result/experiment/', dataset='ihdp'):
    simulation_output_dir = os.path.join(output_base_dir, knob)

    if knob == 'dragonnet':
        run_exp(data_base_dir=data_base_dir, simulation_output_dir=simulation_output_dir, folder=folder, dataset=
        dataset)

    if knob == 'tarnet':
        run_exp(data_base_dir=data_base_dir, simulation_output_dir=simulation_output_dir, dragon=0, folder=folder,
                dataset=
                dataset)

    if knob == 'nednet':
        run_exp(data_base_dir=data_base_dir, simulation_output_dir=simulation_output_dir, dragon='nednet',
                folder=folder, dataset=dataset)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_base_dir', type=str, help="path to directory LBIDD")
    parser.add_argument('--knob', type=str, default='early_stopping',
                        help="early_stopping/with or loss/mse or ratio/0")
    parser.add_argument('--folder', type=str, help='which datasub directory')
    parser.add_argument('--output_base_dir', type=str, help="directory to save the output")

    args = parser.parse_args()

    turn_knob(args.data_base_dir, args.knob, args.folder, args.output_base_dir)


if __name__ == '__main__':
    main()
