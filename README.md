# End-to-End Respiratory Audio AI Pipeline

> 🌐 **Language / Idioma:** English | [Leer en Español (Spanish)](./README.es.md)

---

## Arquitectura

```text
Audio Respiratorio (.wav)
            │
            ▼
      ETL Pipeline
            │
 ┌──────────┼──────────┐
 │          │          │
 ▼          ▼          ▼
Ingesta   MFCC     Parquet
            │
            ▼
   Dataset Procesado
            │
            ▼
      CNN + BLSTM
            │
            ▼
 Modelo Entrenado (.keras)
            │
            ▼
         FastAPI
            │
            ▼
          Docker
            │
            ▼
          Render# End-to-End Pipeline for Respiratory Disease Classification using Deep Learning

## Descripción

Proyecto de ingeniería de datos y Deep Learning para la clasificación automática de enfermedades respiratorias a partir de audios pulmonares.

El sistema implementa un pipeline completo que abarca:

- Ingesta de audios respiratorios.
- Procesamiento distribuido mediante PySpark.
- Extracción de características acústicas (MFCC).
- Almacenamiento de datos procesados en formato Parquet.
- Entrenamiento de una red neuronal CNN-BLSTM.
- Exposición del modelo mediante una API REST desarrollada con FastAPI.
- Contenerización con Docker.
- Despliegue en la nube utilizando Render.
- Orquestación del pipeline mediante Apache Airflow.

---

## Arquitectura

```text
Audio Respiratorio (.wav)
            │
            ▼
      ETL Pipeline
            │
 ┌──────────┼──────────┐
 │          │          │
 ▼          ▼          ▼
Ingesta   MFCC     Parquet
            │
            ▼
   Dataset Procesado
            │
            ▼
      CNN + BLSTM
            │
            ▼
 Modelo Entrenado (.keras)
            │
            ▼
         FastAPI
            │
            ▼
          Docker
            │
            ▼
          Render