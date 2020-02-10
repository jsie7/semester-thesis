#!/usr/bin/env python

"""Module classifying the turbidity of images.

This module contains the training, prediction and post-processing functions,
which are used to predict the turbidity of video frames.

The 'predictor' and 'scaler' arguments of predict_turbidity() have to be
functions returning a predictor and scaler instance, respectively.

Todo:
    * In _train_predict() remove the dependency on all HSV histograms. Right
      now the function needs all three histograms despite (maybe) not using
      them for training. This should be changed and can be controlled via the
      train_ind parameter.

"""

# built-in modules
import copy
import time

# third party modules
import numpy as np
import pywt
from scipy.signal import medfilt

# fix random seed
np.random.seed(21)


def _adaptive_clipping(x):
    """Empirical exponential function.

    f(x) = 0.005*np.power(1000,x*4.53909278672530947) - 0.1*x - 0.005

    The function returns the value of an exponential function. The function
    maps range(0,0.1) to an exponential shape and fixes f(0) = 0 and
    f(0.1) = 0.1. The function was determined empirically.

    Args:
        x (float): Point at which 'f(x)' should be evaluated.

    Returns:
        float: 'f' evaluated at x.

    """
    return 0.005*np.power(1000, x*4.53909278672530947) - 0.1*x - 0.005


def _smooth(x, wavelet="db4", sigma=1.0, medfilter=True, kernel_size=7):
    """Smooting function for turbidity prediction.

    This function smooths the output of the turbidity predictor with wavelet
    denoising and (moving) median filtering.
    First the wavelet denoising is applied, followed by the optional median
    filtering. At the end, the signal is clipped to range(0,1). In
    addition, values close to 0 (range(0,0.1)) and 1 (range(0.9,1)) are
    pushed closer to 0 and 1, respecitvely, using the _adaptive_clipping
    function.

    Args:
        x (np.ndarray): Noisy, unsmoothed input.
        wavelet (str): Wavelet type used in the wavelet decomposition.
        sigma (float): Scaling factor for 'np.sqrt(2*np.log(len(x)))', which
            is used as the noise threshold in wavelet space.
        medfilter (bool): If True apply median filter, if False don't.
        kernel_size (int): Moving window size used by the median filter.

    Returns:
        np.ndarray: Denoised and smoothed input.

    """
    coeff = pywt.wavedec(x, wavelet, mode="per")
    uthresh = sigma * np.sqrt(2 * np.log(len(x)))
    coeff[1:] = (pywt.threshold(i, value=uthresh, mode="soft") for i in
        coeff[1:])
    y = pywt.waverec(coeff, wavelet, mode="per")

    if medfilter:
        y = medfilt(y, kernel_size=kernel_size)

    y = np.clip(y, a_min=0, a_max=1)
    y[np.where(y < 0.1)] = _adaptive_clipping(y[np.where(y < 0.1)])
    y[np.where(y > 0.9)] = 1 - _adaptive_clipping(1 - y[np.where(y > 0.9)])

    if x.shape[0] % 2:
        y = y[:-1]

    return y


def _train_predict(X_train, y_train, X_predict, y_predict, train_ind,
        predictor, scaler=None, statistics=False):
    """Train predictor on training set and predict turbidity on test set.

    Internal function, which is called by predict_turbidity(). It invokes the
    scaler passed to it and subsequently the passed predictor. Training occurs
    on X_train and y_train, while the trained predictor is tested on X_predict
    and y_predict.

    Args:
        X_train (list): List of np.ndarrays containing the training data. Each
            list entry represents one video.
        y_train (list): List of np.ndarrays containing the training labels.
        X_predict (list): List of np.ndarrays containing the test data.
        y_predict (list): List of np.ndarrays containing the test labels.
        train_ind (list): Defines which data is used for training.
            If statistics is True, train_ind has to contain all four
            statistics. If statistics is False, train_ind refers to the HSV
            histograms (where: 0 -> h, 1 -> s, 2 -> v).
            ATTENTION: This only specifies which data is used for training.
                The code however needs all HSV histograms to work!
        predictor (instance): Predictor instance used for training and
            prediction. Must have members 'fit()' and 'predict_proba()'.
        scaler (instance): Scaler instance used to scale data before training
            and prediction. If None, no scaler is invoked.
        statistics (bool): If True histogram statistics are computed and used
            as features; If False no statistics are computed and the full
            histograms are used as features.

    Returns:
        tuple: (Predicted probability of X_predict, y_predict)

    """
    t = time.time()

    # TRAINING
    X_train_h = []
    X_train_s = []
    X_train_v = []
    for x in X_train:
        X_train_h.append(x[0])
        X_train_s.append(x[1])
        X_train_v.append(x[2])

    X_train_h = np.concatenate(X_train_h)
    X_train_s = np.concatenate(X_train_s)
    X_train_v = np.concatenate(X_train_v)
    y_train = np.concatenate(y_train)

    ind = np.random.permutation(np.arange(0, y_train.shape[0]))
    X_train_h = X_train_h[ind]
    X_train_s = X_train_s[ind]
    X_train_v = X_train_v[ind]
    y_train = y_train[ind]

    X_train_list = [X_train_h, X_train_s, X_train_v]

    if statistics:
        means_train = []
        stds_train = []
        skews_train = []
        kurts_train = []

        for X_train in X_train_list:
            X = copy.deepcopy(X_train)
            w = copy.copy(X)
            w[np.where(X.sum(axis=1) == 0), 0] = 1.0
            mean_train = np.average(np.repeat(np.arange(X.shape[1])
                .reshape(-1, 1), X.shape[0], axis=1).T, weights=w, axis=1)
            std_train = np.sqrt(np.average(np.square(np.repeat(np.arange(
                X.shape[1]).reshape(-1, 1), X.shape[0], axis=1).T -
                mean_train.reshape(-1, 1)), weights=w, axis=1))
            skew_train = np.divide(np.average(np.power(np.repeat(np.arange(
                X.shape[1]).reshape(-1, 1), X.shape[0], axis=1).T -
                mean_train.reshape(-1, 1), 3), weights=w, axis=1),
                np.power(std_train, 3))
            kurt_train = np.divide(np.average(np.power(np.repeat(np.arange(
                X.shape[1]).reshape(-1, 1), X.shape[0], axis=1).T -
                mean_train.reshape(-1, 1), 4), weights=w, axis=1),
                np.power(std_train, 4)) - 3
            skew_train[np.isnan(skew_train)] = 0
            kurt_train[np.isnan(kurt_train)] = 0

            means_train.append(mean_train)
            stds_train.append(std_train)
            skews_train.append(skew_train)
            kurts_train.append(kurt_train)
        stats_list = [means_train, stds_train, skews_train, kurts_train]

        assert len(train_ind) == 4
        ind = []
        glob_ind = []
        for i, j in enumerate(train_ind):
            if j:
                ind.append(j)
                glob_ind.append(i)

        train_stats = []
        k = 0
        for i, stat in enumerate(stats_list):
            if i in glob_ind:
                train_stats.append([stat[j] for j in ind[k]])
                k = k + 1

        if scaler is None:
            predictor.fit(np.vstack(train_stats).T, y_train)
        else:
            predictor.fit(scaler.fit_transform(np.vstack(train_stats).T),
                y_train)
    else:
        X_train = [X_train_list[i] for i in train_ind]
        if scaler is None:
            predictor.fit(np.concatenate(X_train, axis=1), y_train)
        else:
            predictor.fit(scaler.fit_transform(np.concatenate(X_train,
                axis=1)), y_train)

    # PREDICTION
    X_predict_h = []
    X_predict_s = []
    X_predict_v = []
    lengths = []
    for i, x in enumerate(X_predict):
        lengths.append(y_predict[i].shape[0])
        X_predict_h.append(x[0])
        X_predict_s.append(x[1])
        X_predict_v.append(x[2])
    lengths = np.cumsum(lengths)

    X_predict_h = np.concatenate(X_predict_h)
    X_predict_s = np.concatenate(X_predict_s)
    X_predict_v = np.concatenate(X_predict_v)

    X_predict_list = [X_predict_h, X_predict_s, X_predict_v]

    if statistics:
        means_predict = []
        stds_predict = []
        skews_predict = []
        kurts_predict = []
        for X_predict in X_predict_list:
            w = copy.copy(X_predict)
            w[np.where(X_predict.sum(axis=1) == 0), 0] = 1.0
            mean_predict = np.average(np.repeat(np.arange(X_predict.shape[1])
                .reshape(-1, 1), X_predict.shape[0], axis=1).T, weights=w,
                axis=1)
            std_predict = np.sqrt(np.average(np.square(np.repeat(np.arange(
                X_predict.shape[1]).reshape(-1, 1), X_predict.shape[0],
                axis=1).T - mean_predict.reshape(-1, 1)), weights=w, axis=1))
            skew_predict = np.divide(np.average(np.power(np.repeat(np.arange(
                X_predict.shape[1]).reshape(-1, 1), X_predict.shape[0],
                axis=1).T - mean_predict.reshape(-1, 1), 3), weights=w,
                axis=1), np.power(std_predict, 3))
            kurt_predict = np.divide(np.average(np.power(np.repeat(np.arange(
                X_predict.shape[1]).reshape(-1, 1), X_predict.shape[0],
                axis=1).T - mean_predict.reshape(-1, 1), 4), weights=w,
                axis=1), np.power(std_predict, 4)) - 3
            skew_predict[np.isnan(skew_predict)] = 0
            kurt_predict[np.isnan(kurt_predict)] = 0

            means_predict.append(mean_predict)
            stds_predict.append(std_predict)
            skews_predict.append(skew_predict)
            kurts_predict.append(kurt_predict)
        stats_list = [means_predict, stds_predict, skews_predict,
                      kurts_predict]

        assert len(train_ind) == 4
        ind = []
        glob_ind = []
        for i, j in enumerate(train_ind):
            if j:
                ind.append(j)
                glob_ind.append(i)

        predict_stats = []
        k = 0
        for i, stat in enumerate(stats_list):
            if i in glob_ind:
                predict_stats.append([stat[j] for j in ind[k]])
                k = k + 1

        if scaler is None:
            y_probas = predictor.predict_proba(np.vstack(train_stats).T)
        else:
            y_probas = predictor.predict_proba(scaler.fit_transform(
                np.vstack(train_stats).T))
    else:
        X_predict = [X_predict_list[i] for i in train_ind]
        if scaler is None:
            y_probas = predictor.predict_proba(np.concatenate(X_predict,
                                                              axis=1))
        else:
            y_probas = predictor.predict_proba(scaler.fit_transform(
                np.concatenate(X_predict, axis=1)))

    y_proba = []
    for i in range(len(lengths)):
        if i == 0:
            y_proba.append(y_probas[:lengths[i]])
        else:
            y_proba.append(y_probas[lengths[i-1]:lengths[i]])

    print("Training and prediction took {0} seconds".format(time.time()-t))
    return y_proba, y_predict


def predict_turbidity(data, predictor=None, scaler=None, pred_len=5,
        train_ind=[0, 1, 2], statistics=False):
    """Function predicting the turbidity of images.

    This function predicts the turbidity of frames of videos passed to the
    function. Subsequently, the prediction is smoothed by the internal smooth
    function.

    Args:
        data (dict): Dictionary containing the processed frames for which the
            turbidity should be predicted. Keys are the video file names,
            values are tuples (X, y). The tuple (X, y) is of the format which
            was returned by the VideoProcessor class.
            WARNING: If this function is used within a data pipeline, this
                dictionary will be produced by the pipeline. This could be
                a source of error.
        predictor (function): Function returning a predictor instance. See
            turbidity_args.py as well.
        scaler (function): Function returning a scaler instance. See
            turbidity_args.py as well.
        pred_len (int): Number of videos in test set. The test set will
            contain 'pred_len' videos, the training set will contain the rest.
        train_ind (list): Indices of the data, which should be used for
            training. Passed to the internal _train_predict function.
        statistics (bool): If True, histogram statistics are used for training
            and prediction instead of full histograms. Passed to internal
            _train_predict function.

    Returns:
        dictionary: Keys are the video file names, values are tuples of true
            labels and predicted labels.

    """
    video_list = data.keys()

    if len(video_list) % pred_len != 0:
        iterations = len(video_list)//pred_len + 1
    else:
        iterations = len(video_list)//pred_len

    X = []
    y = []
    for vid in video_list:
        X.append(data[vid][0])
        y.append(data[vid][1])

    y_proba = []
    y_true = []
    for itr in range(iterations):
        if itr < (iterations - 1):
            X_predict = X[itr*pred_len:(itr+1)*pred_len]
            y_predict = y[itr*pred_len:(itr+1)*pred_len]
            X_train = X[:itr*pred_len]
            X_train += X[(itr+1)*pred_len:]
            y_train = y[:itr*pred_len]
            y_train += y[(itr+1)*pred_len:]
            y_proba_temp, y_true_temp = _train_predict(X_train, y_train,
                                                       X_predict, y_predict,
                                                       train_ind=train_ind,
                                                       predictor=predictor(),
                                                       scaler=scaler(),
                                                       statistics=statistics)
            y_proba += y_proba_temp
            y_true += y_true_temp
        else:
            X_predict = X[itr*pred_len:]
            y_predict = y[itr*pred_len:]
            X_train = X[:itr*pred_len]
            y_train = y[:itr*pred_len]
            y_proba_temp, y_true_temp = _train_predict(X_train, y_train,
                                                       X_predict, y_predict,
                                                       train_ind=train_ind,
                                                       predictor=predictor(),
                                                       scaler=scaler(),
                                                       statistics=statistics)
            y_proba += y_proba_temp
            y_true += y_true_temp

    for i, y in enumerate(y_proba):
        y_proba[i] = np.vstack([_smooth(y[:, 0], wavelet='haar', sigma=0.01,
                                        medfilter=True, kernel_size=5),
                                _smooth(y[:, 1], wavelet='haar', sigma=0.01,
                                        medfilter=True, kernel_size=5)]).T

    data_dict = dict(zip(video_list, zip(y_true, y_proba)))
    return data_dict
