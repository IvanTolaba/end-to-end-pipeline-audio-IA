import sys
import os

# 1. PREPARACIÓN DEL ENTORNO (Prioridad absoluta)
os.environ["HADOOP_HOME"] = "C:\\hadoop"
os.environ["hadoop.home.dir"] = "C:\\hadoop"
os.environ["PATH"] = "C:\\hadoop\\bin;" + os.environ["PATH"]

# 2. IMPORTS DEL PROYECTO
sys.path.append(os.path.abspath("."))

from etl.ingest import ingest_data
from etl.mfcc_pyspark import process_mfcc
from etl.save_parquet import save_data

def run():
    # Ingesta
    data = ingest_data("data/raw")
    
    # Procesamiento (Aquí es donde Spark se despierta)
    df = process_mfcc(data)
    
    # Verificación
    print(f"📊 Filas a procesar: {df.count()}")
    df.show(5)
    
    # Guardado definitivo
    save_data(df, "data/processed_parquet/mfcc_data")

if __name__ == "__main__":
    run()


