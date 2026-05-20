from pathlib import Path
from typing import List, Dict
from config.settings import (EXTENSION_AUDIO, DISEASE)
import logging

logger = logging.getLogger(__name__)


def ingest_data(base_path: str | Path) -> List[Dict]:
    """
    Realiza la ingesta de archivos WAV desde el dataset respiratorio.

    Args:
        base_path:Ruta base del dataset.

    Returns:
        Lista de diccionarios con metadata.

    Raises:
        FileNotFoundError: Si la ruta base no existe.
    """

    logger.info("Iniciando proceso de ingesta")

    # Convertir a Path
    base_path = Path(base_path)

    # Validar que exista la carpeta base
    if not base_path.exists():        
        raise FileNotFoundError(
            f"La carpeta {base_path} no existe"
        )

    data = []

    # Recorrer clases
    for label, clase in enumerate(DISEASE):

        class_path = base_path / clase

        #...
        # Validar carpeta de clase
        if not class_path.exists():
            logger.warning( "No existe carpeta para clase: %s", clase)
            continue
        

        #----

        logger.info( "Procesando clase: %s",clase)

        # Recorrer archivos
        for file in class_path.iterdir():

            if file.suffix == EXTENSION_AUDIO:

                data.append({
                    "path": str(file),
                    "label": label,
                    "class_name": clase
                })

    # Validar dataset vacío
    if not data:
        logger.warning("No se encontraron archivos WAV")

    logger.info("Total audios cargados: %s",len(data))

    return data
