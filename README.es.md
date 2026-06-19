# Pipeline End-to-End de IA para la Detección Automática de Patologías Pulmonares mediante Sonidos Respiratorios

> 🌐 **Language / Idioma:** [Read in English (Inglés)](./README.md) | Español

Este proyecto implementa una solución completa de Machine Learning y MLOps que automatiza el procesamiento distribuido de sonidos respiratorios, la extracción de características acústicas (MFCC) y la inferencia mediante una arquitectura híbrida CNN-BiLSTM para la clasificación automática de Asma, EPOC, Neumonía y condiciones respiratorias normales a partir de grabaciones obtenidas con estetoscopios electrónicos.

---

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Pyspark](https://img.shields.io/badge/Pyspark-Preprocesamiento-sky)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![FastAPI](https://img.shields.io/badge/FastAPI-Producción-green)
![Pytest](https://img.shields.io/badge/Pytest-Implemented-green)
![Tests](https://img.shields.io/badge/Unit%20Tests-Passing-brightgreen)
![Tests](https://img.shields.io/badge/tests-passing-brightgreen)
![Docker](https://img.shields.io/badge/Docker-Listo-blue)
![Airflow](https://img.shields.io/badge/Airflow-Orquestado-red)
![Render](https://img.shields.io/badge/Despliegue-Render-purple)

## 📸 Arquitectura General
![Arquitectura del proyecto](./images/arq4.png)

*El sistema automatiza el ciclo completo (Ciclo de vida de los datos y del modelo) implementando principios de aislamiento de entornos, entrenamiento reproducible en la nube y despliegue continuo.*

---
## 🎬 Demo

![Demo API](./images/Animation.gif)

---

## 🎯 Objetivo

Desarrollar una plataforma End-to-End capaz de automatizar el procesamiento de sonidos respiratorios y asistir en la detección temprana de patologías pulmonares mediante técnicas de Inteligencia Artificial y MLOps.

---

## ✨ Características

- Procesamiento distribuido de audio mediante Apache Spark.
- Extracción automática de características MFCC.
- Entrenamiento reproducible.
- Arquitecturas CNN, BLSTM, CNN-LSTM y CNN-BLSTM.
- Pipeline completamente orquestado con Airflow.
- API REST desarrollada con FastAPI.
- Contenerización mediante Docker.
- Despliegue en la nube.
- Testing automatizado.

---
## ⚙️ Flujo del Pipeline End-to-End

El sistema automatiza el ciclo completo de procesamiento de sonidos respiratorios, mapeando el flujo desde la adquisición de la señal hasta la inferencia en producción:

1. **Adquisición de datos e Ingesta:** Carga de registros fonomecánicos pulmonares integrando múltiples bases de datos clínicas internacionales.
2. **Procesamiento Distribuido:** Remuestreo (16kHz), filtrado de ruido (*Denoise*) y segmentación de señales acústicas optimizadas a gran escala mediante **PySpark**.
3. **Ingeniería de características:** Conversión de señales en representaciones numéricas mediante la extracción de Coeficientes Cepstrales en las Frecuencias de Mel (**MFCC**).
3. **Ingeniería de Características Avanzada:** Conversión de señales en representaciones numéricas mediante la extracción de Coeficientes Cepstrales en las Frecuencias de Mel (**MFCC**). 
Este proceso se ejecuta bajo la siguiente secuencia matemática y de filtrado:
   - **Preénfasis:** Aplicación de un filtro de altas frecuencias para compensar la atenuación natural de los componentes agudos en el tracto respiratorio.
   - **Enventanado (Windowing):** Segmentación mediante ventanas de *Hamming* de 25 ms con un solapamiento (*overlap*) del 25% para garantizar la estacionariedad de la señal de audio de corta duración sin perder continuidad temporal.
   - **Análisis de Frecuencia (FFT):** Cálculo de la Transformada Rápida de Fourier para mapear el espectro de potencia.
   - **Banco de Filtros Mel:** Mapeo logarítmico mediante funciones de transferencia triangulares espaciadas en la escala Mel para imitar la percepción auditiva no lineal humana.
   - **Transformada de Coseno Discreta (DCT):** Decorrelación de los coeficientes para obtener los MFCCs finales, descartando componentes de ruido de alta variabilidad.
4. **Almacenamiento de datos:** Persistencia de vectores de características e información estructurada utilizando formatos eficientes de nivel analítico (**Parquet** y **JSON**).
5. **IA, entrenamiento del modelo:** Entrenamiento reproducible de una arquitectura híbrida avanzada (**CNN + BiLSTM**) utilizando **Google Colab (GPU)**, aplicando balanceo y normalización de clases y técnicas estrictas de prevención de *Data Leakage* (`GroupShuffleSplit`).
6. **Evaluación del modelo:** Cálculo automático de Accuracy, Precision, Recall, F1-Score, Matriz de Confusión, Curvas ROC-AUC.
7. **Registro y seguimiento de modelos:** Versionado del código fuente y artefactos mediante **Git y GitHub**. Almacenamiento del modelo final entrenado (`.keras`).
8. **Producción y Despliegue:** Exposición de endpoints optimizados mediante **FastAPI**, aislamiento del entorno con **Docker** y despliegue productivo en la nube a través de **Render**.
9. **Orquestación:** Automatización y monitoreo de todo el flujo de tareas (DAGs) mediante **Apache Airflow**.
---

## 🛡️ Características de Ingeniería y MLOps

* **Calidad de Datos:** Prevención estricta de *Data Leakage* mediante particionado por archivo de origen utilizando `GroupShuffleSplit`. Aislamiento completo entre conjuntos de entrenamiento, validación y prueba.  Balanceo automático de clases para reducir sesgos en patologías subrepresentadas.

* **Reproducibilidad:** Configuración centralizada mediante archivos de *settings* y variables de entorno. Pipeline determinístico utilizando semillas controladas (*random seeds*). Versionado de código, configuraciones y artefactos del modelo mediante Git y GitHub.

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
├── airflow/           # Orquestación de tareas y definición de DAGs
├── api/               # Código base de la API REST (FastAPI)
├── config/            # Archivos de configuración y entornos (.env)
├── data/              # Data Lake Local (Estructura de almacenamiento)
├── ml/                # Pipeline de Machine Learning
│   ├── training/      # Scripts de entrenamiento y optimización
│   ├── evaluation/    # Reportes de performance
│   ├── artifacts/     # Pesos de los modelos entrenados (.keras)
│   └── reports/       # Gráficos y curvas analíticas (arq.png)
├── notebooks/         # Espacios de experimentación y EDA
├── tests/             # Pruebas unitarias y de integración
├── Dockerfile         # Configuración del contenedor de producción
├── requirements/      # Dependencias desacopladas por módulo
└── README.md

```
---

## 🧠 Experimentación y Arquitecturas Evaluadas

Como parte de la investigación y desarrollo del Trabajo Final, se diseñaron, entrenaron y compararon exhaustivamente cuatro arquitecturas basadas en Deep Learning para determinar el enfoque óptimo en la clasificación de señales bioacústicas respiratorias:

1. **CNN (Convolutional Neural Network):** Diseñada para actuar como extractor automático de características visuales de alto nivel a partir de las matrices de coeficientes MFCC espaciales.
2. **BLSTM (Bidirectional Long Short-Term Memory):** Enfocada puramente en el modelado de dependencias secuenciales a largo plazo, analizando el contexto temporal hacia adelante y hacia atrás.
3. **CNN-LSTM (Híbrida Secuencial):** Una combinación donde la CNN extrae mapas de características espaciales y una capa LSTM convencional procesa su evolución temporal de forma unidireccional.
4. **CNN-BiLSTM (Híbrida Bidireccional):** Integra bloques convolucionales robustos (`Conv2D`, `MaxPooling2D`) acoplados a capas recurrentes bidireccionales (`Bidirectional(LSTM)`), capturando tanto la morfología espectral como el contexto secuencial completo (pasado y futuro) del ciclo respiratorio.

> 🚀 **Nota de Despliegue en Producción:** Tras un análisis riguroso de métricas, la arquitectura híbrida **CNN-BiLSTM** fue seleccionada para el despliegue productivo final en **Render** debido a su consistencia, capacidad superior de generalización frente a ruido acústico y el rendimiento reflejado en las curvas ROC-AUC.

---

## 📊 Experimentación y Selección del Modelo (Model Comparison)

Para determinar la arquitectura óptima, se evaluaron de forma exhaustiva cuatro variantes de redes neuronales profundas bajo el mismo pipeline de preprocesamiento de PySpark y el mismo protocolo de entrenamiento determinístico. 

A fin de garantizar la fiabilidad estadística y mitigar el sobreajuste (*overfitting*), se aplicó una estrategia de división *Hold-Out* (70% train, 15% val, 15% test) combinada y *Early Stopping*.

### Tabla Comparativa de Rendimiento (Performance Matrix)

| Arquitectura Evaluada | Exactitud (Accuracy) | Precisión (Precision) | Sensibilidad (Recall) | Puntuación F1 (F1-Score) | ROC-AUC Macro | Estado en el Pipeline / Despliegue |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **🧠 CNN-BiLSTM** | **80.23 %** | **0.80** | **0.80** | **0.79** | **0.94** | 🟢 **Seleccionado y Desplegado (Render)** |
| **🧠 CNN-LSTM** | *78.45 %* | *0.78* | *0.77* | *0.77* | *0.92* | 🟡 Evaluado en Fase de Tesis |
| **🧠 CNN** | *74.12 %* | *0.74* | *0.75* | *0.73* | *0.89* | 🟡 Evaluado en Fase de Tesis |
| **🧠 BiLSTM** | *70.33 %* | *0.70* | *0.69* | *0.69* | *0.86* | 🟡 Evaluado en Fase de Tesis |

> 📑 **Nota de Ingeniería:** *Aunque se implementaron y evaluaron cuatro arquitecturas bajo el mismo protocolo experimental, la arquitectura CNN-BiLSTM logró el mejor rendimiento general en términos de extracción de características espaciales y alineación temporal. En consecuencia, fue seleccionada como el modelo de producción final actualmente implementado en Render.*

### Curvas Analíticas del Modelo Seleccionado

<div align="left">  
  <img src="./images/r2.PNG" width="55%" alt="Curvas ROC - Modelo CNN-BiLSTM" />
  <p><i>Curva ROC-AUC Macro del modelo en producción (CNN-BiLSTM).</i></p>
</div>

<div align="left">  
  <img src="./images/mc_es.png" width="55%" alt="Matriz de confusión - Modelo CNN-BiLSTM" />
  <p><i>Matriz de Confusión resultante del modelo en producción (CNN-BiLSTM).</i></p>
</div>

---

## 🗃️ Datasets Utilizados

El modelo fue entrenado, validado y evaluado utilizando una base consolidada construida a partir de tres repositorios clínicos internacionales utilizados en investigación biomédica, con anotaciones y validaciones realizadas por especialistas.

### ICBHI 2017 Respiratory Sound Database

* Dataset de referencia recopilado de forma independiente por el laboratorio Lab3R de la Universidad de Aveiro (Portugal), el Hospital Infante D. Pedro (Portugal), la Universidad Aristóteles de Tesalónica (Grecia) y la Universidad de Coímbra (Portugal).
* **920 grabaciones** provenientes de **126 pacientes**.
* Etiquetas clínicas validadas por neumólogos expertos.

### Annotated Lung Sounds Dataset (ALSD)

* Desarrollado por la Universidad de Ciencia y Tecnología de Jordania en colaboración con el Hospital Universitario King Abdullah.
* **340 grabaciones** correspondientes a **112 sujetos**.
* Incluye registros normales y múltiples patologías respiratorias.

### Pulmonary (Lungs) Sound Dataset

* Recopilado por el Hospital Fortis (Nueva Delhi, India).
* **676 grabaciones respiratorias** clasificadas por profesionales de la salud.
* Contiene diversas condiciones pulmonares, incluyendo asma, EPOC y neumonía.

### 📊 Resumen del Dataset Consolidado

| Dataset                 | Grabaciones | Pacientes/Sujetos |
| ----------------------- | ----------: | ----------------: |
| ICBHI 2017              |         920 |               126 |
| ALSD                    |         340 |               112 |
| Pulmonary (Lungs) Sound |         676 |               N/D |
| **Total**               |   **1.936** |          **238+** |

La integración de estas fuentes permitió construir un conjunto de datos más diverso y representativo, mejorando la capacidad de generalización del modelo para la clasificación automática de **Asma, EPOC, Neumonía y sonidos respiratorios normales**.


---
## 🧪 ¡Probalo en Producción! (Audios de Prueba)

Podés interactuar directamente con el modelo desplegado utilizando la interfaz interactiva Swagger UI:

🔗 Enlace de la API: [API Render](https://end-to-end-pipeline-audio-ia.onrender.com/docs). 

**📥 Paso 1: Descargá un Audio de Prueba**

Descargá a tu computadora cualquiera de estas muestras reales para enviarlas a la API:

| Patología Real | Enlace de Descarga de Prueba | Estado Esperado de la API |
| :--- | :--- | :--- |
| **Asma** | [📥 Descargar Audio de Prueba](./ml/examples/Asthma.wav) | `Clasificación: Asma` |
| **EPOC** | [📥 Descargar Audio de Prueba](./ml/examples/Copd.wav) | `Clasificación: EPOC` |
| **Neumonía** | [📥 Descargar Audio de Prueba](./ml/examples/Pneumonia.wav) | `Clasificación: Neumonía` |
| **Normal** | [📥 Descargar Audio de Prueba](./ml/examples/Normal.wav) | `Clasificación: Normal` |


**🚀 Paso 2: Guía de Inferencia en la Interfaz (Swagger)**

    ⏳ Nota de Inicio Técnico (Cold Start): La aplicación se encuentra desplegada en el plan gratuito de Render, si la página inicial se muestra oscura o tarda en cargar, esperá entre 30 a 60 segundos sin refrescar para que el servicio se reactive.

Una vez que visualices la interfaz interactiva de FastAPI, seguí estos pasos:

1. Buscá el endpoint con etiqueta verde POST /predict y hacé clic sobre él para desplegarlo.

2. Hacé clic en el botón Try it out (ubicado arriba a la derecha del panel desplegado).

3. En el campo de carga de archivos (file), hacé clic en Seleccionar archivo y subí el audio .wav que descargaste en el Paso 1.

4. Presioná el botón azul grande Execute.


**📄 Ejemplo de Respuesta de la API**

Tras procesar el audio en tiempo real extrayendo los MFCCs e inyectándolos en la red neuronal CNN-BiLSTM, la API te devolverá un estado 200 con este formato:

```
{
  "status": "success",
  "filename": "107_2b4_Pr_mc_AKGC417L.wav",
  "prediction": "Epoc",
  "confidence": "99.93%"
}
```
---
## 🎓 Trabajo Final de Ingeniería

Este proyecto fue desarrollado como Trabajo Final de la carrera de Ingeniería.

✅ Proyecto aprobado con calificación **10/10**.

El trabajo presenta el diseño e implementación de una plataforma End-to-End basada en Inteligencia Artificial para la clasificación automática de patologías pulmonares mediante sonidos respiratorios.

---
## 📝 Publicaciones y Contribuciones Académicas

La rigurosidad, metodologías y validaciones cruzadas aplicadas en este proyecto fueron arbitradas, aprobadas y expuestas en prestigiosos congresos de ciencias de la computación:

1. **Tolaba, N. I., & Sarmiento, G. N. R. (2025).** *“Identificación Inteligente de Enfermedades Pulmonares en Audios Respiratorios”*. XXVII Congreso Argentino de Ciencias de la Computación (**CACIC 2025**). Universidad Nacional de Jujuy.
2. **Tolaba, N. I., et al. (2025).** *“Deep Learning aplicado a la identificación de cantos de anuncio de Boana riojana (Amphibia: Anura)”*. XXVI Workshop de Investigadores en Ciencias de la Computación (**WICC 2025**). *Nota: Validación exitosa de la adaptabilidad y robustez transespecie de las arquitecturas híbridas desarrolladas sobre señales bioacústicas de anfibios de la región.*

---
## ⚠️ Limitaciones

- No reemplaza el diagnóstico médico.
- El modelo fue entrenado sobre datasets públicos.
- El rendimiento puede variar con dispositivos de captura diferentes.

---
## 🔮 Líneas de Trabajo Futuro

- **Validación Clínica Regional:** Evaluar los modelos entrenados utilizando registros clínicos y condiciones acústicas controladas de pacientes locales en entornos hospitalarios de la provincia.
- **Optimización para Dispositivos Móviles:** Exportar el modelo entrenado a formatos ligeros como *TensorFlow Lite* (`.tflite`) o *ONNX* para habilitar la inferencia *offline* directamente en aplicaciones móviles (Android/iOS).
- **Asistencia en Puntos de Atención (Point-of-Care):** Integración con hardware de estetoscopios digitales accesibles para telemedicina en zonas rurales de bajos recursos.

---




