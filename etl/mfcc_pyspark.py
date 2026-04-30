
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf
from pyspark.sql.types import ArrayType, FloatType

import librosa


# 🔹 Función que transforma UN audio → MFCC
def extract_mfcc(path):
    try:
        # cargar audio
        y, sr = librosa.load(path, sr=22050)

        # extraer MFCC
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

        # convertir a lista (Spark no guarda numpy directo)
        return mfcc.flatten().tolist()

    except Exception as e:
        print(f"❌ Error en {path}: {e}")
        return []


# 🔹 Función principal que usa PySpark
def process_mfcc(data):

    # 1. Crear sesión Spark
    spark = SparkSession.builder \
        .appName("Audio MFCC Pipeline") \
        .getOrCreate()

    # 2. Convertir lista Python → DataFrame Spark
    df = spark.createDataFrame(data)

    # 3. Registrar función como UDF (User Defined Function)
    mfcc_udf = udf(extract_mfcc, ArrayType(FloatType()))

    # 4. Aplicar transformación (columna nueva)
    df = df.withColumn("mfcc", mfcc_udf(df["path"]))

    print("✅ MFCC generados con PySpark")

    return df