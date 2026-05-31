
from pathlib import Path

from pyspark.sql import SparkSession

from etl.save_parquet import save_data


def test_save_data(tmp_path, caplog):

    """
    Integration test:  Verify that save_data correctly saves a parquet.
    """

    # -----------------------------
    # ARRANGE
    # -----------------------------

    spark = (
        SparkSession.builder
        .master("local[1]")
        .appName("test-save")
        .getOrCreate()
    )

    sample_data = [
        {
            "path": "audio.wav",
            "label": 0,
            "class_name": "Asma"
        }
    ]

    df = spark.createDataFrame(sample_data)

    output_path = tmp_path / "parquet_output"

    # -----------------------------
    # ACT
    # -----------------------------

    save_data(df, output_path)

    # -----------------------------
    # ASSERT
    # -----------------------------

    # Verificar que exista carpeta
    assert output_path.exists()

    # Verificar que Spark pueda leerlo
    loaded_df = spark.read.parquet(
        str(output_path)
    )

    # Validar cantidad de filas
    assert loaded_df.count() == 1

    # Validar logs
    assert (
        "Starting to save parquet flooring in"
        in caplog.text
    )

    assert (
        "Saved successfully"
        in caplog.text
    )

    spark.stop()

    