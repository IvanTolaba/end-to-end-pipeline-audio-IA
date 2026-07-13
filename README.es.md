# Pipeline End-to-End de IA y MLOps para el Diagnóstico Automático de Patologías Pulmonares mediante Sonidos Respiratorios

> 🌐 **Language / Idioma:** [Read in English (Inglés)](./README.md) | Español

Este proyecto implementa una solución completa de Data Engineering, Machine Learning y MLOps para la diagnóstico automático de patologías pulmonares a partir de sonidos respiratorios. Se desarrolló un pipeline reproducible que automatiza la ingesta, el procesamiento distribuido de señales, la extracción de características MFCC y la generación de datasets para el entrenamiento y evaluación de cuatro arquitecturas de Deep Learning (LSTM, BiLSTM, CNN-LSTM y CNN-BiLSTM). La arquitectura CNN-LSTM obtuvo el mejor rendimiento y fue desplegada en producción mediante FastAPI, Docker y Render para realizar inferencias en tiempo real.

---

![Python](https://img.shields.io/badge/Python-3.11-blue)
![PySpark](https://img.shields.io/badge/PySpark-Preprocessing-skyblue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![FastAPI](https://img.shields.io/badge/FastAPI-Production-green)
![Pytest](https://img.shields.io/badge/Pytest-Passing-brightgreen)
![Docker](https://img.shields.io/badge/Docker-Containerized-blue)
![Airflow](https://img.shields.io/badge/Airflow-Orchestrated-red)
![Render](https://img.shields.io/badge/Deployment-Render-purple)

## 📸 Arquitectura General
![Arquitectura del proyecto](./images/arq4.png)

*La arquitectura resume el pipeline End-to-End implementado, integrando Data Engineering, Machine Learning y MLOps para automatizar todo el flujo, desde la adquisición de sonidos respiratorios hasta el despliegue del modelo CNN-LSTM en producción mediante FastAPI, Docker y Render.*

---
## 🎬 Demo

![Demo API](./images/Animation.gif)

---

## 🎯 Objetivo

Desarrollar una plataforma End-to-End que automatice el procesamiento distribuido de sonidos respiratorios, la extracción de características acústicas y el diagnóstico automático de patologías pulmonares (Asma, Neumonía, EPOC y condición normal) mediante técnicas de Inteligencia Artificial y MLOps.

---

## ✨ Características

- Pipeline End-to-End reproducible
- Procesamiento distribuido con PySpark.
- Extracción automática de características MFCC.
- Entrenamiento de cuatro arquitecturas de Deep Learning: CNN, BiLSTM, CNN-LSTM y CNN-BiLSTM.
- Pipeline completamente orquestado con Airflow.
- API REST con FastAPI.
- Contenerización mediante Docker.
- Despliegue en Render.
- Testing automatizado con Pytest.

---

## ⚙️ Flujo del Pipeline End-to-End

El sistema automatiza el ciclo completo de procesamiento de sonidos respiratorios, mapeando el flujo desde la adquisición de la señal hasta la inferencia en producción:

1. **Adquisición de Datos e Ingesta:** Carga de registros fonomecánicos pulmonares integrando múltiples bases de datos clínicas internacionales de referencia.
2. **Procesamiento Distribuido:** Filtrado de ruido (*Denoise*) y segmentación de señales acústicas optimizadas a gran escala mediante **PySpark**.
3. **Ingeniería de Características Avanzada:** Conversión de señales en representaciones numéricas mediante la extracción de Coeficientes Cepstrales en las Frecuencias de Mel (**MFCC**). Este proceso se ejecuta bajo la siguiente secuencia matemática y de filtrado:
   - **Preénfasis:** Aplicación de un filtro de altas frecuencias para compensar la atenuación natural de los componentes agudos en el tracto respiratorio.
   - **Enventanado (Windowing):** Segmentación mediante ventanas de *Hamming* de 25 ms con un solapamiento (*overlap*) del 25% para garantizar la estacionariedad de la señal de audio de corta duración sin perder continuidad temporal.
   - **Análisis de Frecuencia (FFT):** Cálculo de la Transformada Rápida de Fourier para mapear el espectro de potencia de la señal.
   - **Banco de Filtros Mel:** Mapeo logarítmico mediante funciones de transferencia triangulares espaciadas en la escala Mel para imitar la percepción auditiva no lineal humana.
   - **Transformada de Coseno Discreta (DCT):** Decorrelación de los coeficientes para obtener los MFCCs finales, descartando componentes de ruido de alta variabilidad.
4. **Almacenamiento de Datos:** Persistencia de vectores de características e información estructurada utilizando formatos eficientes de nivel analítico (**Parquet** y **JSON**).
5. **Entrenamiento de Modelos Candidatos:** Entrenamiento reproducible de múltiples arquitecturas avanzadas de aprendizaje profundo (**CNN, BiLSTM, CNN-LSTM y CNN-BiLSTM**) utilizando **Google Colab (GPU)**. El proceso incorporó balanceo y normalización de clases, prevención de **Data Leakage** mediante GroupShuffleSplit y técnicas de regularización como **Early Stopping**, permitiendo detener el entrenamiento cuando el desempeño sobre el conjunto de validación dejaba de mejorar, reduciendo el riesgo de overfitting.
6. **Evaluación y Selección:** Cálculo automático de métricas comparativas (Accuracy, Precision, Recall, F1-Score, Curvas ROC-AUC y Matrices de Confusión) para la selección del modelo de mejor rendimiento global.
7. **Registro y Seguimiento de Modelos:** Versionado del código fuente y artefactos mediante **Git y GitHub**. Almacenamiento optimizado del modelo final seleccionado (`.keras`).
8. **Producción y Despliegue:** Exposición de endpoints optimizados para inferencia en tiempo real mediante **FastAPI**, aislamiento completo del entorno con **Docker** y despliegue continuo en la nube a través de **Render**.
9. **Orquestación y Automatización:** Automatización integral y monitoreo de todo el flujo de tareas y dependencias (DAGs) mediante **Apache Airflow**.


---

## 🛡️ Características de Ingeniería y MLOps

* **Calidad de Datos:** Implementación de un pipeline robusto que incluye prevención de *Data Leakage* mediante `GroupShuffleSplit`, aislamiento completo entre los conjuntos de entrenamiento, validación y prueba, normalización de las características (features) para mejorar la estabilidad del entrenamiento y balanceo automático de clases para mitigar el desbalance presente en las patologías subrepresentadas.

* **Reproducibilidad:** Configuración centralizada mediante archivos de *settings* y variables de entorno. Versionado de código, configuraciones y artefactos del modelo mediante Git y GitHub.

* **Escalabilidad:** Procesamiento distribuido de señales respiratorias mediante PySpark. Arquitectura modular desacoplada para facilitar mantenimiento y extensión. Persistencia eficiente utilizando formatos analíticos Parquet y JSON.

* **Producción:** API REST desarrollada con FastAPI para inferencia en tiempo real. Carga optimizada del modelo mediante ciclo de vida (*lifespan*) para minimizar latencia. Contenerización completa utilizando Docker. Despliegue automatizado en la nube mediante Render.

* **Orquestación y Automatización** Automatización integral del pipeline mediante Apache Airflow. Ejecución reproducible de procesos ETL, entrenamiento y evaluación. Generación automática de métricas, reportes y artefactos del modelo.

---
## 🏗️ Ingeniería aplicada

Durante el desarrollo se implementaron prácticas de Ingeniería de Software y MLOps:

- Arquitectura modular.
- Separación entre entrenamiento e inferencia.
- Configuración mediante variables de entorno.
- Logging.
- Testing.
- Docker.
- Pipeline reproducible.
- Orquestación con Airflow.
- Persistencia en formatos Parquet.

---

## 🛠️ Tecnologías Utilizadas

| Área | Tecnologías / Herramientas |
| :--- | :--- |
| **🧠 Machine Learning e IA** | TensorFlow, Keras, Scikit-Learn, NumPy, Pandas |
| **🎵 Procesamiento de Audio** | Librosa, SoundFile |
| **⚙️ Ingeniería de Datos y Orquestación** | Apache Airflow, Apache Spark (PySpark), Parquet, JSON |
| **⚡ Backend y API** | FastAPI, Uvicorn, Postman |
| **🐳 MLOps y Despliegue** | Docker, Git, GitHub, Render |
| **🧪 Testing y Calidad** | Pytest, Logging, Caplog |
| **📊 Visualización** | Matplotlib, Seaborn |

---
## 📂 Estructura del Proyecto

```text
.end-to-end-pipeline-audio-IA
├── airflow/                     # DAGs y orquestación del pipeline con Apache
├── api/                         # API REST desarrollada con FastAPI
├── config/                      # Configuración centralizada 
├── docker/                      # # Dockerfiles para los distintos servicios|  
|    ├── airflow.Dockerfile      # Servicio de orquestación del pipeline
|    ├── api.Dockerfile          # Servicio de inferencia mediante FastAPI
|    ├── etl.Dockerfile          # Servicio de procesamiento distribuido con PySpark
|    ├── training.Dockerfile     # Servicio de entrenamiento y evaluación de modelos         
│    
├── etl/                         # Pipeline ETL para procesamiento 
|    |                           # distribuido de audio
|    ├── ingest.py               # Ingesta de audios y metadatos
|    ├── mfcc_pyspark.py         # Extracción distribuida de MFCC con PySpark
|    ├── save_parquet.py         # Persistencia del dataset en formato Parquet
|    ├── segment.py              # Segmentación de señales respiratorias
|
├── images/                      # Recursos gráficos utilizados en el README
├── ml/                          # Pipeline de Machine Learning
|    ├── artifacts/              # Modelos entrenados y artefactos generados
|    ├── evaluation/             # Evaluación del rendimiento del modelo
|    ├── examples/               # Audios de muestra para probar modelo desplegado 
|    ├── reports/                # Reportes de métricas
│    ├── training/               # Entrenamiento de los modelos
│         
├── notebooks/                   # Análisis exploratorio y experimentación 
├── requirements/                # Dependencias específicas de cada componente
├── .gitignore
├── docker-compose.yml           # Orquestación de contenedores
├── LICENSE   
├── main.py                      # Punto de entrada 
├── pytest.ini                   # Configuración de Pytest
├── README.es.md                 # Documentación en español
└── README.md                    # Documentación principal en inglés

```

---

### 🏗️ Architecture Overview

| Layer | Directory | Responsibility |
|-------|-----------|----------------|
| Workflow Orchestration | `airflow/` | DAGs that orchestrate ETL, model training and evaluation. |
| Data Engineering | `etl/` | Data ingestion, preprocessing, segmentation and distributed MFCC extraction with PySpark. |
| Machine Learning | `ml/` | Model training, evaluation, reports and artifacts. |
| Model Serving | `api/` | REST API for real-time inference using FastAPI. |
| Infrastructure | `docker/` | Dockerfiles and containerization of all services. |
| Configuration | `config/` | Centralized configuration using YAML files. |
| Quality Assurance | `tests/` | Automated testing with Pytest. |

---
### 🏗️ Arquitectura del Proyecto

El proyecto sigue una arquitectura modular por capas, separando las responsabilidades de orquestación, ingeniería de datos, entrenamiento de modelos, despliegue, infraestructura y aseguramiento de la calidad. Esta organización facilita la mantenibilidad, la escalabilidad y la reproducibilidad de la solución.

| Capa | Directorio | Responsabilidad |
|------|------------|-----------------|
| **Orquestación de Flujos** | `airflow/` | Define los DAGs que automatizan y coordinan el pipeline ETL, el entrenamiento y la evaluación de los modelos. |
| **Data Engineering** | `etl/` | Implementa la ingesta, el preprocesamiento, la segmentación y la extracción distribuida de características MFCC mediante PySpark. |
| **Machine Learning** | `ml/` | Contiene el entrenamiento, la evaluación, los reportes y los artefactos generados por los modelos. |
| **Servicio de Modelos (Serving)** | `api/` | Expone el modelo entrenado mediante una API REST desarrollada con FastAPI para realizar inferencias en tiempo real. |
| **Infraestructura** | `docker/` | Incluye los Dockerfiles y la configuración necesaria para la contenerización de los distintos servicios. |
| **Configuración** | `config/` | Centraliza la configuración del proyecto mediante archivos YAML. |
| **Aseguramiento de la Calidad** | `tests/` | Contiene las pruebas automatizadas implementadas con Pytest para validar el correcto funcionamiento del sistema. |

---

## ☁️ Despliegue

La plataforma se encuentra desplegada en **Render**, utilizando contenedores Docker y una API REST desarrollada con FastAPI para realizar inferencias en tiempo real.

---

## 🧠 Experimentación y Arquitecturas Evaluadas

Como parte de la investigación y desarrollo del Trabajo Final, se diseñaron, entrenaron y compararon exhaustivamente cuatro arquitecturas basadas en Deep Learning, las cuales fueron entrenadas y evaluadas bajo exactamente el mismo protocolo experimental para determinar el enfoque óptimo en la clasificación de señales bioacústicas respiratorias:

1. **CNN (Convolutional Neural Network):** Diseñada para actuar como extractor automático de características de alto nivel a partir de las matrices de coeficientes MFCC espaciales.
2. **BLSTM (Bidirectional Long Short-Term Memory):** Enfocada puramente en el modelado de dependencias secuenciales a largo plazo, analizando el contexto temporal hacia adelante y hacia atrás.
3. **CNN-LSTM (Híbrida Secuencial):** Una combinación donde la CNN extrae mapas de características espaciales y una capa LSTM convencional procesa su evolución temporal de forma unidireccional.
4. **CNN-BiLSTM (Híbrida Bidireccional):** Integra bloques convolucionales robustos (`Conv2D`, `MaxPooling2D`) acoplados a capas recurrentes bidireccionales (`Bidirectional(LSTM)`), capturando tanto la morfología espectral como el contexto secuencial completo (pasado y futuro) del ciclo respiratorio.

> 🚀 **Nota de Despliegue en Producción:** Tras un análisis riguroso de métricas, la arquitectura híbrida **CNN-LSTM** fue seleccionada para el despliegue productivo final en **Render** debido a su consistencia, capacidad superior de generalización frente a ruido acústico y el rendimiento reflejado en las métricas.

---

## 📊 Experimentación y Selección del Modelo 

Para determinar la arquitectura óptima, se evaluaron de forma exhaustiva cuatro variantes de redes neuronales profundas bajo el mismo pipeline de preprocesamiento de PySpark y el mismo protocolo de entrenamiento. 

A fin de garantizar la fiabilidad estadística y mitigar el sobreajuste (*overfitting*), se aplicó una estrategia de división *Hold-Out* (70% train, 15% val, 15% test) combinada y *Early Stopping*.

### Tabla Comparativa de Rendimiento 

| Arquitectura Evaluada | Exactitud (Accuracy) | Precisión (Precision) | Sensibilidad (Recall) | Puntuación F1 (F1-Score) | ROC-AUC Macro | Estado en el Pipeline / Despliegue |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **🧠 CNN-LSTM** | **81 %** | **0.8** | **0.8** | **0.8** | **0.94** | 🟢 **Seleccionado y Desplegado (Render)**  |
| **🧠 CNN-BiLSTM** | *80.23 %* | *0.80* | *0.80* | *0.79* | *0.94* | 🟡 Evaluado en Fase de Tesis |
| **🧠 CNN** | *80.23 %* | *0.8* | *0.79* | *0.79* | *0.93* | 🟡 Evaluado en Fase de Tesis |
| **🧠 BiLSTM** | *68 %* | *0.68* | *0.68* | *0.67* | *0.88* | 🟡 Evaluado en Fase de Tesis |

> 📑 **Nota de Ingeniería:** *Aunque se implementaron y evaluaron cuatro arquitecturas bajo el mismo protocolo experimental, la arquitectura CNN-LSTM logró el mejor rendimiento general en términos de extracción de características espaciales y alineación temporal. En consecuencia, fue seleccionada como el modelo de producción final actualmente implementado en Render.*

### Curvas Analíticas del Modelo Seleccionado

<div align="left">  
  <img src="./images/roc-cnn-lstm.PNG" width="55%" alt="Curvas ROC - Modelo CNN-BiLSTM" />
  <p><i>Curva ROC-AUC Macro del modelo en producción (CNN-LSTM).</i></p>
</div>

<div align="left">  
  <img src="./images/matriz_confusion_profesional.png" width="55%" alt="Matriz de confusión - Modelo CNN-BiLSTM" />
  <p><i>Matriz de Confusión resultante del modelo en producción (CNN-LSTM).</i></p>
</div>

---

## 🗃️ Datasets y Adquisición de Datos

El modelo fue entrenado, validado y evaluado utilizando un conjunto consolidado de tres bases de datos públicas de sonidos respiratorios, ampliamente utilizadas en investigación biomédica. Las grabaciones fueron obtenidas en hospitales y centros de investigación internacionales y cuentan con anotaciones clínicas realizadas por especialistas. En conjunto, estos datos proporcionan una base sólida para el desarrollo de modelos de Inteligencia Artificial, al ofrecer información clínicamente confiable, técnicamente consistente y disponible para su utilización en investigación científica.



### 🌐 1. ICBHI 2017 Respiratory Sound Database
* **Origen:** Recopilado de forma independiente por el laboratorio Lab3R de la [Universidad de Aveiro (Portugal)](https://www.ua.pt/pt/essua) (junto al Hospital Infante D. Pedro), la [Universidad Aristóteles de Tesalónica (Grecia)](https://www.auth.gr/) (Hospital Papanikolaou) y la [Universidad de Coímbra (Portugal)](https://www.uc.pt/en/).
* **Validación:** Etiquetas clínicas validadas por neumólogos expertos en el marco del desafío internacional ICBHI.
* **Enlace Oficial:** [Sitio del Desafío ICBHI 2017](https://bhichallenge.med.auth.gr/)

### 🌐 2. Annotated Lung Sounds Dataset (ALSD-Net)
* **Origen:** Desarrollado por la [Universidad de Ciencia y Tecnología de Jordania](https://www.just.edu.jo/Pages/Default.aspx) en colaboración directa con el Hospital Universitario King Abdullah.
* **Hardware de Captura:** Grabaciones fonomecánicas pulmonares capturadas mediante un **estetoscopio electrónico 3M Littmann modelo 3200** colocado en diversas posiciones anatómicas de la pared torácica.
* Cubren edades de 21 a 90 años.
* **Enlace Oficial:** [ALSD en Mendeley Data](https://data.mendeley.com/datasets/jwyy9np4gv/3)

### 🌐 3. Pulmonary (Lungs) Sound Dataset
* **Origen:** Recopilado y clasificado por médicos profesionales del [Hospital Fortis](https://www.fortishealthcare.com/location/fortis-flt-lt-rajan-dhall-hospital-vasant-kunj) en Nueva Delhi, India.
* **Especificación Técnica:** Las grabaciones se realizaron mediante un estetoscopio electrónico conectado a una computadora portátil a través de un amplificador de señal. El sistema físico se configuró específicamente para **amplificar el rango de frecuencias críticas entre 70 Hz y 2000 Hz**, garantizando la captura precisa de fenómenos acústicos respiratorios.
* **Enlace Oficial:** [Pulmonary Sound en Mendeley Data](https://data.mendeley.com/datasets/fr7zvy8j5s/1)

---

## 📊 Resumen Estadístico del Dataset Consolidado

La integración y homologación de las **tres bases de datos públicas internacionales** permitió construir un conjunto consolidado de **1.247 grabaciones** provenientes de **más de 238 pacientes**, proporcionando una muestra más diversa y representativa para el entrenamiento, validación y evaluación de modelos de clasificación de **Asma, EPOC, Neumonía y Normal**.

| Dataset                     | Grabaciones | Pacientes/Sujetos | Método de adquisición                            |
| :-------------------------- | :---------: | :---------------: | :----------------------------------------------- |
| **ICBHI 2017**              |     866     |        126        | Estetoscopio electrónico                         |
| **ALSD-Net**                |     137     |        112        | Estetoscopio electrónico                         |
| **Pulmonary (Lungs) Sound** |     244     |       *N/D*       | Estetoscopio electrónico y micrófono             |
| **Total consolidado**       |  **1.247**  |      **238+**     | **Clasificación: Asma, EPOC, Neumonía y Normal** |

> 🛡️ Calidad de los datos y prevención de Data Leakage: Las cantidades de grabaciones corresponden al conjunto consolidado de audios seleccionado para este proyecto, no al tamaño original de cada dataset. Se conservaron únicamente las clases **Asma, EPOC, Neumonía y Normal**, descartando las demás patologías. Posteriormente, los audios fueron procesados mediante un pipeline uniforme que incluyó **filtrado de calidad, segmentación con un solapamiento del 25 %, extracción de coeficientes MFCC, normalización y balanceo de clases**. Cada segmento conservó un identificador del audio de origen, lo que permitió realizar el particionado mediante **GroupShuffleSplit**, garantizando que todos los segmentos derivados de una misma grabación permanecieran en el mismo conjunto (entrenamiento, validación o prueba) y evitando así el **Data Leakage**.

---

# 🧪 ¡Probá la API en producción!

**📥 Paso 1: Descargá un Audio de Prueba**

Descargá a tu computadora cualquiera de estas muestras reales para enviarlas a la API:

| Patología Real | Enlace de Descarga de Prueba | Estado Esperado de la API |
| :--- | :--- | :--- |
| **Asma** | [📥 Descargar Audio de Prueba](./ml/examples/Asthma.wav) | `Clasificación: Asma` |
| **EPOC** | [📥 Descargar Audio de Prueba](./ml/examples/Copd.wav) | `Clasificación: EPOC` |
| **Neumonía** | [📥 Descargar Audio de Prueba](./ml/examples/Pneumonia.wav) | `Clasificación: Neumonía` |
| **Normal** | [📥 Descargar Audio de Prueba](./ml/examples/Normal.wav) | `Clasificación: Normal` |

⚠️ **¿Querés probar con tus propias grabaciones?**

Para obtener resultados comparables con los del entrenamiento, las grabaciones deben cumplir las siguientes condiciones:
* **Hardware:** Grabaciones realizadas exclusivamente mediante un estetoscopio electrónico.
* **Frecuencia de muestreo:** Mínimo de 44.1 kHz.
* **Duración:** No menor a 3 segundos (para capturar al menos un ciclo respiratorio completo) y no mayor a 10 segundos (para mantener un procesamiento e inferencia eficientes).

---

**🚀 Paso 2: Probá la API desplegada en Render**

> ⏳ **Nota de Inicio Técnico (Cold Start):** La aplicación se encuentra desplegada en el plan gratuito de Render. Si la página inicial tarda en cargar, esperá entre 30 a 60 segundos sin refrescar para que el contenedor se reactive del estado de reposo.

Accedé a la API desplegada en Render, cuya documentación interactiva está disponible mediante Swagger UI:

🔗 Enlace de la API: [API en Render](https://end-to-end-pipeline-audio-ia.onrender.com/docs).

Una vez que visualices la interfaz interactiva de FastAPI, seguí estos pasos:

1. Buscá el endpoint con etiqueta verde **POST `/predict`** y hacé clic sobre él para desplegar el panel.
2. Hacé clic en el botón **Try it out** (ubicado arriba a la derecha del panel desplegado).
3. En el campo de carga de archivos (`file`), hacé clic en **Examinar ...** y subí el audio `.wav` que descargaste en el Paso 1.
4. Presioná el botón horizontal azul **Execute**.

---

**📄 Ejemplo de Respuesta de la API**

Tras procesar el audio, la API ejecuta automáticamente el pipeline de procesamiento, que comprende el filtrado de calidad, la segmentación, la extracción de coeficientes MFCC y la inferencia mediante la arquitectura neuronal **CNN-LSTM**. Como resultado, devuelve una respuesta HTTP 200 OK con un objeto JSON que contiene el estado de la operación, el nombre del archivo procesado, la clase predicha y el nivel de confianza asociado.

```json
{
  "status": "success",
  "filename": "107_2b4_Pr_mc_AKGC417L.wav",
  "prediction": "EPOC",
  "confidence": "99.93%"
}
```

--- 

## 🎓 Trabajo Final de Ingeniería e Iniciativa MLOps

Este proyecto tiene su origen en el **Trabajo Final de la carrera de Ingeniería en Informática** de la **Facultad de Ingeniería de la Universidad Nacional de Jujuy (FI-UNJu)**. La investigación científica y el desarrollo del núcleo de IA se realizaron bajo la dirección y el respaldo del **Laboratorio de GeoTecnologías y Ciencias de las Imágenes (FI-UNJu)**. El Trabajo Final fue defendido obteniendo la calificación máxima de **10/10**.

El Trabajo Final abordó el diseño, entrenamiento y validación de modelos de Deep Learning para la clasificación automática de patologías pulmonares Asma, Neumonía, EPOC y condición Normal mediante sonidos respiratorios.

Tras la aprobación académica, el proyecto evolucionó como una iniciativa personal con el objetivo de transformar un prototipo de investigación en una plataforma End-to-End orientada a producción, incorporando prácticas de Ingeniería de Datos, Machine Learning y MLOps.

Entre las principales mejoras implementadas se encuentran:

* 🚀 Exposición del modelo mediante una **API REST** desarrollada con **FastAPI** y documentación automática con Swagger/OpenAPI.
* 🔄 Automatización del pipeline de datos,entrenamiento y evaluación mediante **Apache Airflow**.
* ⚡ Procesamiento distribuido de señales biomédicas utilizando **PySpark**.
* 🐳 Contenerización completa de la plataforma mediante **Docker**, garantizando portabilidad y reproducibilidad.
* ☁️ Despliegue continuo en la nube utilizando **Render**, permitiendo inferencias en tiempo real.

---
## 📝 Publicaciones y Contribuciones Académicas

La rigurosidad, metodologías y validaciones aplicadas en este proyecto fueron arbitradas, aprobadas y expuestas en prestigiosos congresos de ciencias de la computación:

1. **Tolaba, N. I., & Sarmiento, G. N. R. (2025).** *“Identificación Inteligente de Enfermedades Pulmonares en Audios Respiratorios”*. XXVII Congreso Argentino de Ciencias de la Computación (**CACIC 2025**). Universidad Nacional de Jujuy.
2. **Tolaba, N. I., et al. (2025).** *“Deep Learning aplicado a la identificación de cantos de anuncio de Boana riojana (Amphibia: Anura)”*. XXVI Workshop de Investigadores en Ciencias de la Computación (**WICC 2025**). *Nota: Validación exitosa de la adaptabilidad y robustez transespecie de las arquitecturas híbridas desarrolladas sobre señales bioacústicas de anfibios de la región.*

---
## ⚠️ Limitaciones

* El sistema constituye una herramienta de apoyo y **no reemplaza el diagnóstico médico profesional**.
* El modelo fue entrenado y evaluado utilizando **datasets públicos para investigación**, por lo que su desempeño en otros entornos clínicos puede diferir.
* El rendimiento puede verse afectado por diferencias en los dispositivos de captura, el ruido ambiental y las condiciones de adquisición del audio.
* El número de grabaciones disponibles para algunas patologías es limitado. La incorporación de un mayor volumen y diversidad de registros clínicos permitiría mejorar la capacidad de generalización del modelo y potencialmente incrementar sus métricas de desempeño.

---

## 🔮 Líneas de Trabajo Futuro

* **Validación Clínica Regional:** Evaluar los modelos entrenados utilizando registros clínicos y condiciones acústicas controladas de pacientes en entornos hospitalarios.

* **Ampliación del Conjunto de Datos:** Incorporar nuevas grabaciones provenientes de diferentes instituciones de salud para incrementar la diversidad de pacientes, dispositivos de captura y condiciones clínicas. Un mayor volumen y variedad de datos permitirá mejorar la capacidad de generalización del modelo y potencialmente incrementar su desempeño.

* **Optimización para Dispositivos Móviles:** Exportar el modelo a formatos ligeros como **TensorFlow Lite** (`.tflite`) u **ONNX** para habilitar la inferencia *offline* en dispositivos Android e iOS.

* **Asistencia en Puntos de Atención (Point-of-Care):** Integrar la plataforma con estetoscopios digitales y soluciones de telemedicina para brindar apoyo al diagnóstico en centros de atención primaria y zonas con acceso limitado a especialistas.

---

## 📄 Licencia

Este proyecto está bajo la Licencia MIT; ya puedes ver el archivo [LICENSE](LICENSE) para más detalles.


