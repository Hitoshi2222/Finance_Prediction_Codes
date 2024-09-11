import os
import warnings

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=Warning)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

# import pynvml
import matplotlib.pyplot as plt
import random

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.metrics import MeanSquaredError, MeanAbsoluteError
from tensorflow.keras.optimizers import Adam
from keras.callbacks import EarlyStopping

# from silence_tensorflow import silence_tensorflow

plt.style.use("fivethirtyeight")


def train_test(
    x_train,
    y_train,
    x_validate,
    y_validate,
    x_test,
    hyperparameters,
):
    random.seed(777)
    np.random.seed(777)
    tf.random.set_seed(777)

    print("Num GPUs Available: ", len(tf.config.list_physical_devices("GPU")))

    tf.get_logger().setLevel("INFO")
    tf.autograph.set_verbosity(0)

    import logging

    tf.get_logger().setLevel(logging.ERROR)

    # スケーラーを初期化
    x_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()

    # 訓練データを使ってスケーラーをフィット（最小値と最大値を計算）
    x_scaler.fit(x_train)
    y_scaler.fit(y_train.reshape(-1, 1))

    # 訓練データ、検証データ、テストデータを変換（スケーリング）
    x_train = x_scaler.transform(x_train)
    x_validate = x_scaler.transform(x_validate)
    x_test = x_scaler.transform(x_test)

    if np.isnan(x_train).any():
        print("x_train has missing values.")
    if np.isnan(x_validate).any():
        print("x_validate has missing values.")
    if np.isnan(x_test).any():
        print("x_test has missing values.")
    if np.isnan(y_train).any():
        print("y_train has missing values.")
    if np.isnan(y_validate).any():
        print("y_validate has missing values.")

    # Create a tf.data.Dataset for training
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.batch(hyperparameters["batch"]).prefetch(
        tf.data.experimental.AUTOTUNE
    )

    # Create a tf.data.Dataset for validation
    validation_dataset = tf.data.Dataset.from_tensor_slices((x_validate, y_validate))
    validation_dataset = validation_dataset.batch(hyperparameters["batch"]).prefetch(
        tf.data.experimental.AUTOTUNE
    )

    # Create a tf.data.Dataset for testing
    test_dataset = tf.data.Dataset.from_tensor_slices(x_test)
    test_dataset = test_dataset.batch(hyperparameters["batch"]).prefetch(
        tf.data.experimental.AUTOTUNE
    )

    # モデル初期化
    model = Sequential()

    model.add(
        LSTM(
            units=hyperparameters[f"num_units_0"],
            return_sequences=False,
            input_shape=(x_train.shape[1], 1),
        )
    )

    model.add(Dropout(hyperparameters[f"dropout_rate_0"]))

    model.add(Dense(units=1))

    optimizer = Adam(learning_rate=hyperparameters["learning_rate"])
    model.compile(
        optimizer=optimizer,
        loss="mse",
        metrics=[MeanSquaredError(), MeanAbsoluteError()],
    )

    # Early stopping
    early_stopping = EarlyStopping(
        monitor="val_loss", patience=hyperparameters["early_stop_num"], verbose=1
    )

    model.fit(
        train_dataset,
        batch_size=hyperparameters["batch"],
        epochs=50,
        shuffle=False,
        validation_data=validation_dataset,
        callbacks=[early_stopping],
    )

    predictions = model.predict(test_dataset)

    return predictions
