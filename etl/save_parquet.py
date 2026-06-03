import logging
from pathlib import Path
from pyspark.sql import DataFrame

logger = logging.getLogger(__name__)


def save_data(df: DataFrame, output_path: str | Path) -> None:
    """
    Save structured features Spark DataFrame in highly optimized Parquet layout.

    Args:
        df: Processed acoustic Spark DataFrame.
        output_path: Target directory of the processed Data Lake layer.
    Raises:
        ValueError: If the DataFrame contains no internal data.
    """
    logger.info("Initiating persistence job to Silver layer: %s", output_path)
    output_path = Path(output_path)

    if df.rdd.isEmpty():
        logger.error("Aborting save operation. Spark DataFrame contains no elements.")
        raise ValueError("Cannot write an unpopulated or empty DataFrame")

    try:
        # Overwrite state ensures pipeline can re-run automatically via Airflow orchestrations
        (
            df.write
            .mode("overwrite")
            .parquet(str(output_path))
        )
        logger.info("Data Lake Silver table written in Parquet format at: %s", output_path)

    except Exception as error:
        logger.exception("Critical error encountered while storing Parquet files: %s", error)
        raise