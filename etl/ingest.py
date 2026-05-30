from pathlib import Path
from typing import List, Dict
from config.settings import (EXTENSION_AUDIO, DISEASE)
import logging

logger = logging.getLogger(__name__)


def ingest_data(base_path: str | Path) -> List[Dict]:
    """
    It performs WAV file ingestion from the respiratory dataset.

    Args:
        base_path:Base path of the dataset.
    Returns:
        List of dictionaries with metadata.
    Raises:
        FileNotFoundError: If the base path does not exist.
    """

    logger.info("Initiating the ingestion process")

    # Convert to Path
    base_path = Path(base_path)

    # Verify that the base folder exists
    if not base_path.exists() or not base_path.is_dir():        
        raise FileNotFoundError(
            f"The folder {base_path} does not exist"
        )

    data = []

    # Browse classes
    for label, clase in enumerate(DISEASE):

        class_path = base_path / clase

        # Validate class folder
        if not class_path.exists():
            logger.warning( "There is no folder for class: %s", clase)
            continue
        
        logger.info( "Processing class: %s",clase)

        # Browse files
        for file in class_path.iterdir():

            if file.suffix == EXTENSION_AUDIO:

                data.append({
                    "path": str(file),
                    "label": label,
                    "class_name": clase
                })

    # Validate empty dataset
    if not data:
        logger.warning("No WAV files were found")

    logger.info("Total audio files uploaded: %s",len(data))

    return data
