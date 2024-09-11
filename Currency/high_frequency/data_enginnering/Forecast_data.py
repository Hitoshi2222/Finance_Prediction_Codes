from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

# Market Price=moving average
from pyspark.sql.window import Window
from pyspark.sql.functions import (
    avg,
    sum,
    col,
    stddev,
    max,
    min,
    count,
    window,
    row_number,
    to_timestamp,
    date_format,
    lead,
    first,
    last,
    skewness,
    kurtosis,
)


def Volume_vs_Average_Execute_Price(data_frame, frequency):

    data_frame = data_frame.withColumn(
        "ValueRateProduct", col("Execute Volume") * col("Execute Price")
    )
    df_aggregated = (
        data_frame.groupBy(window(col("Execute Time"), frequency))
        .agg(
            sum("ValueRateProduct").alias("Total Execute Volume"),
            avg("Execute Price").alias("Average Execute Price"),
        )
        .orderBy("window")
    )

    df_aggregated = df_aggregated.withColumn("window", col("window.start"))

    return df_aggregated


def Execute_Type_data_for_forecast(data_frame, frequency):

    # 必要な列のみを選択
    df_aggregated_excute_type = (
        data_frame.groupBy(window("Execute Time", frequency), "Execute Type")
        .count()
        .groupBy("window")
        .pivot("Execute Type")
        .sum("count")
        .withColumnRenamed("MARKET", "MARKET_count")
        .withColumnRenamed("INSTANT", "INSTANT_count")
        .withColumnRenamed("LIMIT", "LIMIT_count")
        .withColumnRenamed("STOP", "STOP_count")
        .orderBy("window")
    )

    df_aggregated_excute_type = df_aggregated_excute_type.fillna(0)

    df_aggregated = df_aggregated_excute_type.withColumn("window", col("window.start"))

    result_df = df_aggregated.orderBy("window")

    return result_df


def Side_type_data_for_forecast(data_frame, frequency):

    # 必要な列のみを選択
    df_aggregated_excute_type = (
        data_frame.groupBy(window("Execute Time", frequency), "Side")
        .count()
        .groupBy("window")
        .pivot("Side")
        .sum("count")
        .withColumnRenamed("SELL", "sell_count")
        .withColumnRenamed("BUY", "buy_count")
        .orderBy("window")
    )

    df_aggregated_excute_type = df_aggregated_excute_type.fillna(0)
    df_aggregated = df_aggregated_excute_type.withColumn("window", col("window.start"))

    return df_aggregated


def Trade_Type_data_for_forecast(data_frame, frequency):

    aggregated_df = (
        data_frame.groupBy(window(col("Execute Time"), frequency), col("Trade Type"))
        .agg(count("Trade Type").alias("Count of type"))
        .orderBy("window")
    )

    # Pivot the table to have separate columns for each Trade Type
    pivoted_df = (
        aggregated_df.groupBy("window")
        .pivot("Trade Type")
        .agg(sum("Count of type"))
        .fillna(0)
        .orderBy("window")
    )

    pivoted_df = pivoted_df.withColumn("window", col("window.start"))

    return pivoted_df


def price_enginner(data_frame, frequency):

    data_frame = (
        data_frame.groupBy(window(col("Execute Time"), frequency))
        .agg(
            max("Execute Price").alias("Max Price"),
            min("Execute Price").alias("Min Price"),
            first("Execute Price").alias("First Price"),
            last("Execute Price").alias("Last Price"),
            stddev("Execute Price").alias("Standard Deviation"),
            skewness("Execute Price").alias("Skewness"),
            kurtosis("Execute Price").alias("Kurtosis"),
        )
        .withColumn("Price Range", col("Max Price") - col("Min Price"))
        .orderBy("window")
    )

    data_frame = data_frame.withColumn("window", col("window.start"))

    return data_frame


def read_bk_price(data_frame, frequency):

    window_spec = Window.partitionBy(window(col("quote_time5"), frequency)).orderBy(
        col("quote_time5")
    )
    data_frame = data_frame.withColumn("row_num", row_number().over(window_spec))

    # row_numが1の行のみを選択
    first_prices = data_frame.filter(col("row_num") == 1).select(
        col("quote_time5").alias("time"), "ask_price", "bid_price"
    )
    # print(first_prices.columns)

    first_prices = first_prices.withColumnRenamed("time", "window")

    return first_prices


spark_context = SparkContext()
spark = SparkSession(spark_context).builder.appName("csvToDB").getOrCreate()

frequency = "X minutes"

data_frame_bk = spark.read.parquet("X_bk.parquet")

data_ask_price = read_bk_price(data_frame_bk, frequency)
data_ask_price = data_ask_price.withColumn(
    "window", date_format(col("window"), "yyyy-MM-dd HH:mm:ss")
)
data_ask_price = data_ask_price.withColumn(
    "window", to_timestamp(col("window"), "yyyy-MM-dd HH:mm:ss")
)


data_frame = spark.read.parquet("X.parquet")
Volume_vs_Average_Execute_Price_data = Volume_vs_Average_Execute_Price(
    data_frame, frequency
)

Execute_Type_data = Execute_Type_data_for_forecast(data_frame, frequency)
Side_type_data = Side_type_data_for_forecast(data_frame, frequency)
Trade_Type_data = Trade_Type_data_for_forecast(data_frame, frequency)
price_enginnered_data = price_enginner(data_frame, frequency)

# 一つ目と二つ目のデータフレームを連結
joined_df = Volume_vs_Average_Execute_Price_data.join(
    Execute_Type_data, on="window", how="outer"
)

# 三つ目のデータフレームを連結
joined_df = joined_df.join(Side_type_data, on="window", how="outer")
# 四つ目のデータフレームを連結
joined_df = joined_df.join(Trade_Type_data, on="window", how="outer")
# 五つ目のデータフレームを連結
joined_df = joined_df.join(price_enginnered_data, on="window", how="outer")

# Window specification
windowSpec = Window.orderBy("window")
# `window` 列のみ1行前にシフト
joined_df = joined_df.withColumn("window", lead(col("window")).over(windowSpec))


# 五つ目のデータフレームを連結
joined_df = joined_df.join(data_ask_price, on="window", how="inner")

joined_df.write.mode("overwrite").parquet(
    f"explainable_valiable_{frequency}_data_ver1.parquet"
)
