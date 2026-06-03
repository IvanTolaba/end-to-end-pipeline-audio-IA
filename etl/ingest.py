import logging
from pathlib import Path
from typing import Dict, List
from config.settings import DISEASE, EXTENSION_AUDIO

logger = logging.getLogger(__name__)


def ingest_data(base_path: str | Path) -> List[Dict]:
    """
    Performs WAV file ingestion from the respiratory dataset.

    Args:
        base_path: Base path of the dataset.
    Returns:
        List of dictionaries containing metadata {
                    "path": str(file),
                    "label": label,
                    "class_name": disease
                }
    Raises:
        FileNotFoundError:
            If the dataset directory or class folders do not exist.
        ValueError:
            If no WAV files are found.
    """
    logger.info("Initiating the ingestion process from raw storage")
    base_path = Path(base_path)

    # --------------------------------------------------
    # Validate dataset directory
    # --------------------------------------------------
    if not base_path.exists() or not base_path.is_dir():
        logger.error("Dataset directory does not exist: %s", base_path)
        raise FileNotFoundError(f"Dataset directory does not exist: {base_path}")

    # --------------------------------------------------
    # Validate all disease folders
    # --------------------------------------------------
    missing_classes = [
        disease for disease in DISEASE if not (base_path / disease).exists()
    ]

    if missing_classes:
        logger.error("Missing class folders: %s", ", ".join(missing_classes))
        raise FileNotFoundError(
            f"Missing class folders: {', '.join(missing_classes)}"
        )

    # --------------------------------------------------
    # Process audio files
    # --------------------------------------------------
    data = []

    for label, disease in enumerate(DISEASE):
        class_path = base_path / disease
        logger.info("Processing class folder: %s", disease)
        wav_count = 0

        for file in class_path.iterdir():
            if file.suffix.lower() != EXTENSION_AUDIO:
                continue

            wav_count += 1
            data.append(
                {
                    "path": str(file),
                    "label": label,
                    "class_name": disease
                }
            )

        logger.info("Class %s: %s raw WAV files found", disease, wav_count)

    # --------------------------------------------------
    # Validate dataset content
    # --------------------------------------------------
    if not data:
        logger.error("No WAV files were found in the dataset")
        raise ValueError("No WAV files were found in the dataset")

    logger.info("Total raw audio files uploaded to pipeline: %s", len(data))
    return data