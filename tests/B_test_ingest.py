
from pathlib import Path

from etl.ingest import ingest_data
from config.settings import DISEASE


def test_ingest_data_real_dataset(caplog):
    """
    Verify that the ingestion process:

        - correctly reads the actual dataset
        - generates the expected structure
        - processes all WAV files
        - records informative logs
    """

    # --------------------------------------------------
    # ARRANGE
    # --------------------------------------------------

    base_path = Path("data/raw")

    assert base_path.exists(), ("The data/raw folder does not exist")

    # --------------------------------------------------
    # INGEST
    # --------------------------------------------------

    data = ingest_data(base_path)

    # --------------------------------------------------
    # ASSERT
    # --------------------------------------------------

    # It must return a list
    assert isinstance(data, list)

    # It should not come empty.
    assert len(data) > 0

    # --------------------------------------------------
    # Validate the structure of the first record
    # --------------------------------------------------

    record = data[0]

    assert "path" in record
    assert "label" in record
    assert "class_name" in record

    # --------------------------------------------------
    # Validate the existence of files
    # --------------------------------------------------

    for item in data:
        assert Path(item["path"]).exists()

    # --------------------------------------------------
    # Validate found classes
    # --------------------------------------------------

    classes_found = {
        item["class_name"]
        for item in data
    }

    assert classes_found == set(DISEASE)

    # --------------------------------------------------
    # Validate total amount of WAV
    # --------------------------------------------------

    total_wavs = sum(
        len(
            list(
                (base_path / disease).glob("*.wav")
            )
        )
        for disease in DISEASE
    )

    assert len(data) == total_wavs

    # --------------------------------------------------
    # Validate logs
    # --------------------------------------------------

    assert (
        "Initiating the ingestion process"
        in caplog.text
    )

    assert (
        "Total audio files uploaded"
        in caplog.text
    )








