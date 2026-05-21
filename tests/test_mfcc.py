from pathlib import Path

from pyspark.sql import SparkSession

from etl.mfcc_pyspark import (
    extract_mfcc,
    process_mfcc
)


# -----------------------------------
# Test extract_mfcc
# -----------------------------------

def test_extract_mfcc_real_audio(caplog):

    """
    Unit test:
    Verifica extracción MFCC
    sobre un audio real.
    """

    # -----------------------------
    # ARRANGE
    # -----------------------------

    audio_path = Path(
        "data/raw/Asma"
    )

    # Buscar primer WAV
    wav_files = list(
        audio_path.glob("*.wav")
    )

    assert wav_files, (
        "No existen archivos WAV "
        "para testing"
    )

    test_audio = wav_files[0]

    # -----------------------------
    # ACT
    # -----------------------------

    mfcc = extract_mfcc(
        str(test_audio)
    )

    # -----------------------------
    # ASSERT
    # -----------------------------

    # Debe devolver lista
    assert isinstance(
        mfcc,
        list
    )

    # No debe venir vacía
    assert mfcc

    # Debe contener floats
    assert isinstance(
        mfcc[0],
        float
    )

    # Validar logs
    assert (
        "Procesando audio"
        in caplog.text
    )

    assert (
        "MFCC extraído correctamente"
        in caplog.text
    )


# -----------------------------------
# Test process_mfcc
# -----------------------------------

def test_process_mfcc_pipeline(caplog):

    """
    Integration test:
    Verifica pipeline Spark MFCC.
    """

    # -----------------------------
    # ARRANGE
    # -----------------------------

    spark = (
        SparkSession.builder
        .master("local[1]")
        .appName("test-mfcc")
        .getOrCreate()
    )

    audio_path = Path(
        "data/raw/Asma"
    )

    wav_files = list(
        audio_path.glob("*.wav")
    )

    assert wav_files, (
        "No existen WAVs "
        "para testing"
    )

    sample_data = [
        {
            "path": str(wav_files[0]),
            "label": 0,
            "class_name": "Asma"
        }
    ]

    # -----------------------------
    # ACT
    # -----------------------------

    df = process_mfcc(
        spark,
        sample_data
    )

    # -----------------------------
    # ASSERT
    # -----------------------------

    # Verificar columnas
    assert "mfcc" in df.columns

    # Verificar cantidad filas
    assert df.count() == 1

    # Obtener primer registro
    row = df.first()

    # Validar MFCC generado
    assert row["mfcc"]

    # Validar tipo
    assert isinstance(
        row["mfcc"],
        list
    )

    # Validar logs
    assert (
        "Iniciando pipeline MFCC"
        in caplog.text
    )

    assert (
        "MFCC generados correctamente"
        in caplog.text
    )

    spark.stop()