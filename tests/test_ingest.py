
from etl.ingest import ingest_data
from pathlib import Path



def test_ingest_data_real_dataset(caplog):
    """
    Integration Test:
    Valida que el proceso de ingesta funcione correctamente
    usando el dataset real del proyecto.
    """

    # 1. ARRANGE
    # Crear objeto Path multiplataforma
    base_path = Path("data/raw")

    # Verificar que exista el dataset
    assert base_path.exists(), ("The data/raw folder does not exist")

    # 2. ACT
    # Ejecutar la función real
    data = ingest_data(base_path)

    # 3. ASSERT

    # Validar tipo de dato
    assert isinstance(data, list)

    # Validar que no venga vacío
    assert data

    # Validar estructura del primer registro
    registro = data[0]

    assert "path" in registro
    assert "label" in registro
    assert "class_name" in registro

    # Verificar que el archivo exista realmente
    assert Path(registro["path"]).exists()

    # -----------------------------
    # Verificar logs (caplog)
    # -----------------------------

    assert (
        "Initiating the ingestion process"
        in caplog.text
    )

    assert (
        "Total audio files uploaded"
        in caplog.text
    )








