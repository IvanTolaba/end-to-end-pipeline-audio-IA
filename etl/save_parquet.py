
from pathlib import Path
from pyspark.sql import DataFrame

import logging
logger = logging.getLogger(__name__)


def save_data(df: DataFrame,output_path: str | Path) -> None:
    """
    Guarda un DataFrame Spark en formato parquet.

    Args:
        df:DataFrame de Spark a guardar.
        output_path: dónde se guardará parquet.

    Raises:
        ValueError:Si el DataFrame está vacío.

        Exception:Si ocurre un error durante el guardado.
    """

    logger.info("Iniciando guardado parquet en: %s",output_path)

    output_path = Path(output_path)

    # Validar DataFrame vacío
    if df.rdd.isEmpty():
        logger.warning("El DataFrame está vacío")
        raise ValueError("No se puede guardar un DataFrame vacío")

    try:
        (
            df.write
            .mode("overwrite")
            .parquet(str(output_path))
        )

        
        logger.info("Parquet guardado en %s",output_path)
        logger.info("Guardado completado correctamente")

    except Exception as error:
        logger.exception("Error al guardar parquet: %s",error)
        raise




