from etl.ingest import ingest_data
from etl.mfcc_pyspark import process_mfcc
from etl.save_parquet import save_data

def run_pipeline():

    data = ingest_data("data/raw")

    df = process_mfcc(data)

    save_data(df, "data/processed")

    df.show(5)


if __name__ == "__main__":
    run_pipeline()

