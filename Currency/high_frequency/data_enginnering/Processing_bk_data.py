from pyspark.sql import SparkSession
from pyspark.sql.functions import col, window, row_number
from pyspark.sql.window import Window

# Sparkセッションの初期化
spark = SparkSession.builder.appName(
    "First ASK_PRICE X Minutes in Chunks"
).getOrCreate()

# データの読み込み
data_frame = spark.read.parquet("X_bk.parquet")

# print(unique_dates)

# ウィンドウを定義してX分間隔で最初のask_priceを取得
window_spec = Window.partitionBy(window(col("quote_time5"), "X minutes")).orderBy(
    col("quote_time5")
)
data_frame = data_frame.withColumn("row_num", row_number().over(window_spec))

# row_numが1の行のみを選択
first_ask_prices = data_frame.filter(col("row_num") == 1).select(
    col("quote_time5").alias("time"), "ask_price", "bid_price"
)
# 結果をParquetファイルとして保存
output_path = f"X_data.parquet"
first_ask_prices.write.parquet(output_path, mode="overwrite")

first_ask_prices.show()

# 処理の進行状況を表示
print(f"Processed processed_ask_data saved to {output_path}")
