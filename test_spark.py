from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .master("local[*]") \
    .appName("test") \
    .getOrCreate()

print("🔥 Spark arrancó correctamente")

spark.stop()