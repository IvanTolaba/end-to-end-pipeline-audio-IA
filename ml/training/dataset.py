# Carga el dataset Parquet
import logging
import pandas as pd
from pathlib import Path

logger = logging.getLogger(__name__)

def load_dataset(parquet_path: str | Path) -> pd.DataFrame:
    """
    Carga el dataset optimizado en formato Parquet generado por el ETL.
    """
    logger.info(f"Cargando datos distribuidos desde: {parquet_path}")
    path = Path(parquet_path)
    
    if not path.exists():
        raise FileNotFoundError(f"El directorio Parquet no existe en la ruta: {path}")
        
    # Pandas lee de forma nativa todo un directorio particionado de Parquet
    df = pd.read_parquet(path)
    logger.info(f"Dataset cargado con éxito. Total de registros: {len(df)}")
    return df