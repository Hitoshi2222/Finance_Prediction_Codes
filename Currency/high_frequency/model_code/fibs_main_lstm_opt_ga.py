import numpy as np
import pandas as pd
import os
import time

from LSTM_dynamic import train_test

from pyspark import SparkContext
from pyspark.sql import SparkSession

from pyspark.sql.window import Window
from pyspark.sql.functions import (
    col,
    row_number,
)


class FX_PRED:

    def fit(
        self,
        pre_hyperparameters,
        opt_flag,
        X,
        y,
        y_orig,
        train_period,
        validate_start_idx,
        validate_end_idx,
        test_start_idx,
        test_end_idx,
        dataleakdays,
    ):
        st = time.time()
        print("Start calculating -------------")

        lt = time.time()

        data, hyperparameters = self._calc_score(
            X,
            y,
            y_orig,
            pre_hyperparameters,
            opt_flag,
            validate_start_idx,
            validate_end_idx,
            train_period,
            test_start_idx,
            test_end_idx,
            dataleakdays,
        )

        return data, hyperparameters

    def _calc_score(
        self,
        X,
        y,
        pre_hyperparameters,
        opt_flag,
        validate_start_idx,
        validate_end_idx,
        train_period,
        test_start_idx,
        test_end_idx,
        dataleakdays,
    ):
        _x_train = X.filter((col("row_number") < validate_start_idx - dataleakdays))
        _y_train = y.filter((col("row_number") < validate_start_idx - dataleakdays))

        if opt_flag == "y":
            # ハイパーパラメータチューニングのモジュールを以下に入れる。
            param_ = np.random.rand(100, 50)
        else:
            param_ = pre_hyperparameters

        _x_train = (
            X.filter(
                (validate_start_idx - dataleakdays - train_period < col("row_number"))
                & (col("row_number") < validate_start_idx - dataleakdays)
            )
            .drop("row_number", "window")
            .toPandas()
            .fillna(method="ffill")
            .values
        )
        _x_validate = (
            X.filter(
                (validate_start_idx <= col("row_number"))
                & (col("row_number") <= validate_end_idx - dataleakdays)
            )
            .drop("row_number", "window")
            .toPandas()
            .fillna(method="ffill")
            .values
        )
        _x_test = (
            X.filter(
                (test_start_idx <= col("row_number"))
                & (col("row_number") < test_end_idx)
            )
            .drop("row_number", "window")
            .toPandas()
            .fillna(method="ffill")
            .values
        )

        _y_train = (
            y.filter(
                (validate_start_idx - dataleakdays - train_period < col("row_number"))
                & (col("row_number") < validate_start_idx - dataleakdays)
            )
            .select("Dependent_Variable_ask")
            .toPandas()
            .values
        )
        _y_validate = (
            y.filter(
                (validate_start_idx <= col("row_number"))
                & (col("row_number") <= validate_end_idx - dataleakdays)
            )
            .select("Dependent_Variable_ask")
            .toPandas()
            .values
        )

        y_pred = train_test(
            _x_train, _y_train, _x_validate, _y_validate, _x_test, param_
        )

        return y_pred


def load_files():

    spark_context = SparkContext()
    spark = SparkSession(spark_context).builder.appName("csvToDB").getOrCreate()

    X = spark.read.parquet(f"explainable_variable.parquet")
    y = spark.read.parquet(f"dependent_variable.parquet")
    y_orig = spark.read.parquet(f"dependent_variable_orig.parquet")

    y = y.select("window", "Dependent_variable_ask")
    y_orig = y_orig.select("window", "past_avg_ask_price", "ask_price")

    # 'window'列を基にdf_Xとdf_yの間で内部結合を行い、df_Xの列のみを保持する
    X = X.join(y, ["window"], "inner").select(X["*"])
    y_orig = y_orig.join(y, ["window"], "inner").select(y_orig["*"])

    # 'window'列を日付型またはタイムスタンプ型として昇順に並べ替えるためのWindow指定
    window_spec = Window.orderBy("window")
    # 行番号を付与
    X = X.withColumn("row_number", row_number().over(window_spec))
    y = y.withColumn("row_number", row_number().over(window_spec))
    y_orig = y_orig.withColumn("row_number", row_number().over(window_spec))

    return X, y, y_orig


def learn(
    X,
    y,
    y_orig,
    pre_hyperparameters,
    opt_flag,
    val_period=60,
    out_dir="results",
):

    dataleakdays = np.random(50)

    total_len = X.count()
    rolling_start_idx = np.random(50)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    df = pd.DataFrame()
    feature = pd.DataFrame()

    test_start_idx = rolling_start_idx
    test_end_idx = total_len
    validate_end_idx = rolling_start_idx
    validate_start_idx = validate_end_idx - val_period  # 4.5M
    train_end_idx = validate_start_idx
    train_start_idx = 0

    precision = FX_PRED.fit(
        # out_f,  # out_f
        pre_hyperparameters,
        opt_flag,
        X,  # X
        y,  # y
        y_orig,
        train_end_idx,
        validate_start_idx,
        validate_end_idx,
        test_start_idx,
        test_end_idx,
        dataleakdays=dataleakdays,
    )

    return precision
