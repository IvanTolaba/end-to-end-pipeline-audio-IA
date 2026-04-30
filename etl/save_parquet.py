

def save_data(df, output_path):
    print(f"💾 Guardando en: {output_path}")

    df.write \
        .mode("overwrite") \
        .parquet(output_path)

    print("✅ Guardado completado")

