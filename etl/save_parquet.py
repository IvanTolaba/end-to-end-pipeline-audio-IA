
from pathlib import Path
from pyspark.sql import DataFrame

import logging
logger = logging.getLogger(__name__)


def save_data(df: DataFrame,output_path: str | Path) -> None:
    """
    Save a Spark DataFrame in parquet format.

    Args:
        df: Spark DataFrame to save.
        output_path: where the parquet flooring will be stored.

    Raises:
        ValueError: If the DataFrame is empty.

        Exception: If an error occurs during saving.
    """

    logger.info("Starting to save parquet flooring in: %s",output_path)

    output_path = Path(output_path)

    # Validate empty DataFrame
    if df.rdd.isEmpty():
        logger.warning("The DataFrame is empty")
        raise ValueError("You cannot save an empty DataFrame")

    try:
        (
            df.write
            .mode("overwrite")
            .parquet(str(output_path))
        )

        
        logger.info("Parquet stored in %s",output_path)
        logger.info("Saved successfully")

    except Exception as error:
        logger.exception("Error saving parquet: %s",error)
        raise




