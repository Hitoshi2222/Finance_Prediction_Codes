import numpy as np
import pandas as pd
import os
import time
import datetime

from LSTM_dynamic import train_test


class LSTM:

    def fit(
        self,
        pre_hyperparameters,
        opt_flag,
        X,
        y,
        train_start_idx,
        validate_start_idx,
        validate_end_idx,
        test_start_idx,
        test_end_idx,
        dataleakdays,
    ):
        st = time.time()
        print("Start calculating -------------")

        y_pred = self._calc_score(
            X,
            y,
            pre_hyperparameters,
            opt_flag,
            train_start_idx,
            train_start_idx,
            validate_start_idx,
            validate_end_idx,
            test_start_idx,
            test_end_idx,
            dataleakdays,
        )

        return y_pred

    def _calc_score(
        self,
        X,
        y,
        pre_hyperparameters,
        opt_flag,
        train_start_idx,
        validate_start_idx,
        validate_end_idx,
        test_start_idx,
        test_end_idx,
        dataleakdays,
    ):

        if opt_flag == "y":
            # ハイパーパラメータチューニングのモジュールを以下に入れる。
            param_ = np.random.rand(100, 50)
        else:
            param_ = pre_hyperparameters

        x_train = X[train_start_idx : validate_start_idx - dataleakdays].values
        y_train = y[train_start_idx : validate_start_idx - dataleakdays].values

        _x_validate = X[validate_start_idx : validate_end_idx - dataleakdays].values

        _y_validate = y[validate_start_idx : validate_end_idx - dataleakdays].values

        _x_test = X[test_start_idx:test_end_idx].values
        y_pred = train_test(x_train, y_train, _x_validate, _y_validate, _x_test, param_)

        return y_pred


def learn(
    X,
    y,
    target,
    pre_hyperparameters,
    opt_flag,
    rolling_start="2022-07-01",
    val_period=60,
    out_dir="results",
):

    pred_days = np.random(0, 20)

    rolling_start = datetime.datetime(
        int(rolling_start.split("-")[0]),
        int(rolling_start.split("-")[1]),
        int(rolling_start.split("-")[2]),
    )
    rolling_start = y.index[y.index >= rolling_start].min()
    rolling_start_idx = y.index.get_loc(rolling_start)

    total_len = len(y)

    train_start_idx = 0

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    test_start_idx = rolling_start_idx
    test_end_idx = total_len
    validate_end_idx = test_start_idx
    validate_start_idx = validate_end_idx - val_period  # 4.5M

    y_pred = LSTM.fit(
        target,
        pre_hyperparameters,
        opt_flag,
        X,  # X
        y,  # y
        train_start_idx,
        validate_start_idx,
        validate_end_idx,
        test_start_idx,
        test_end_idx,
        dataleakdays=pred_days,
    )
