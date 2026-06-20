# Pipeline End-to-End de IA y MLOps para la Diagnóstico Automático de Patologías Pulmonares mediante Sonidos Respiratorios

> 🌐 **Language / Idioma:** [Read in English (Inglés)](./README.md) | Español

Este proyecto implementa una solución completa de Machine Learning y MLOps para la diagnóstico automático de patologías pulmonares a partir de sonidos respiratorios. Se desarrollaron y evaluaron cuatro arquitecturas de Deep Learning (LSTM, BiLSTM, CNN-LSTM y CNN-BiLSTM) utilizando el mismo pipeline de procesamiento distribuido basado en PySpark y extracción de características MFCC. La arquitectura CNN-LSTM obtuvo el mejor rendimiento y fue desplegada en producción mediante FastAPI, Docker y Render para realizar inferencias en tiempo real.

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

*El sistema automatiza el ciclo de vida completo del Machine Learning, desde la ingesta de datos hasta el despliegue del modelo CCN-LSTM en producción, aplicando procesamiento distribuido, entrenamiento reproducible, versionado y principios de MLOps.*

---
## 🎬 Demo

![Demo API](./images/Animation.gif)

---

## 🎯 Objetivo

Desarrollar una plataforma End-to-End que automatice el procesamiento distribuido de sonidos respiratorios, la extracción de características acústicas y el diagnóstico automático de patologías pulmonares mediante técnicas de Inteligencia Artificial y MLOps.

---

## ✨ Características

- Procesamiento distribuido de audio mediante Apache Spark.
- Extracción automática de características MFCC.
- Entrenamiento reproducible.
- Arquitecturas CNN, BiLSTM, CNN-LSTM y CNN-BiLSTM.
- Pipeline completamente orquestado con Airflow.
- API REST desarrollada con FastAPI.
- Contenerización mediante Docker.
- Despliegue en la nube.
- Testing automatizado.

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
  <img src="./images/r2.PNG" width="55%" alt="Curvas ROC - Modelo CNN-BiLSTM" />
  <p><i>Curva ROC-AUC Macro del modelo en producción (CNN-LSTM).</i></p>
</div>

<div align="left">  
  <img src="./images/mc_es.png" width="55%" alt="Matriz de confusión - Modelo CNN-BiLSTM" />
  <p><i>Matriz de Confusión resultante del modelo en producción (CNN-LSTM).</i></p>
</div>

---

## 🗃️ Datasets y Adquisición de Datos

El modelo fue entrenado, validado y evaluado utilizando un conjunto consolidado de tres bases de datos públicas de sonidos respiratorios ampliamente utilizadas en investigación biomédica. Las grabaciones fueron adquiridas en hospitales y centros de investigación internacionales y cuentan con anotaciones clínicas realizadas por especialistas.

### 🌐 1. ICBHI 2017 Respiratory Sound Database
* **Origen:** Recopilado de forma independiente por el laboratorio Lab3R de la [Universidad de Aveiro (Portugal)](https://www.ua.pt/pt/essua) (junto al Hospital Infante D. Pedro), la [Universidad Aristóteles de Tesalónica (Grecia)](https://www.auth.gr/) (Hospital Papanikolaou) y la [Universidad de Coímbra (Portugal)](https://www.uc.pt/en/).
* **Volumen:** **920 grabaciones** de audio correspondientes a **126 pacientes**.
* **Validación:** Etiquetas clínicas validadas por neumólogos expertos en el marco del desafío internacional ICBHI.
* **Enlace Oficial:** [Sitio del Desafío ICBHI 2017](https://bhichallenge.med.auth.gr/)

### 🌐 2. Annotated Lung Sounds Dataset (ALSD-Net)
* **Origen:** Desarrollado por la [Universidad de Ciencia y Tecnología de Jordania](https://www.just.edu.jo/Pages/Default.aspx) en colaboración directa con el Hospital Universitario King Abdullah.
* **Hardware de Captura:** Grabaciones fonomecánicas pulmonares capturadas mediante un **estetoscopio electrónico 3M Littmann modelo 3200** colocado en diversas posiciones anatómicas de la pared torácica.
* **Volumen:** **340 grabaciones** correspondientes a **112 sujetos** (35 sanos y 77 con afecciones, cubriendo edades de 21 a 90 años).
* **Enlace Oficial:** [ALSD en Mendeley Data](https://data.mendeley.com/datasets/jwyy9np4gv/3)

### 🌐 3. Pulmonary (Lungs) Sound Dataset
* **Origen:** Recopilado y clasificado por médicos profesionales del [Hospital Fortis](https://www.fortishealthcare.com/location/fortis-flt-lt-rajan-dhall-hospital-vasant-kunj) en Nueva Delhi, India.
* **Especificación Técnica:** Las grabaciones se realizaron mediante un estetoscopio electrónico conectado a una computadora portátil a través de un amplificador de señal. El sistema físico se configuró específicamente para **amplificar el rango de frecuencias críticas entre 70 Hz y 2000 Hz**, garantizando la captura precisa de fenómenos acústicos respiratorios.
* **Volumen:** **676 grabaciones respiratorias** que abarcan un espectro clínico diverso.
* **Enlace Oficial:** [Pulmonary Sound en Mendeley Data](https://data.mendeley.com/datasets/fr7zvy8j5s/1)

---

### 📊 Resumen Estadístico del Dataset Consolidado

La integración y homologación de estas tres fuentes internacionales permitieron construir un conjunto de datos robusto de **1.936 grabaciones** y más de **238 pacientes**, permitiendo construir un conjunto de datos más diverso y representativo para el entrenamiento y evaluación de los modelos de clasificación. de **Asma, EPOC, Neumonía y condiciones normales**.

**Proceso de integración:** Todos los datasets fueron sometidos a un proceso uniforme de preprocesamiento, segmentación, normalización y balanceo antes del entrenamiento de los modelos.

| Dataset | Grabaciones | Pacientes/Sujetos | Método de Adquisición |
| :--- | :---: | :---: | :--- |
| **ICBHI 2017** | 920 | 126 | Grabaciones Multi-centro (Portugal/Grecia) |
| **ALSD-Net** | 340 | 112 | Estetoscopio Electrónico 3M Littmann 3200 |
| **Pulmonary (Lungs) Sound** | 676 | *N/D* | Amplificación de Frecuencias (70 Hz - 2000 Hz) |
| **Total Consolidado** | **1.936** | **238+** | **Clasificación: Asma, EPOC, Neumonía, Normal** |

---
### 📊 Distribución y Composición del Dataset Consolidado

La integración y homologación de las tres fuentes internacionales dio como resultado un espectro de datos diverso y representativo. A continuación se detalla la distribución exacta de muestras de audio (segmentos de ciclos respiratorios) clasificados por patología e identificando el desbalanceo biológico inicial del problema:

| Diagnóstico / Clase | Cantidad de Muestras (Audios) | Porcentaje (%) | Estado de Distribución |
| :--- | :---: | :---: | :--- |
| **🫁 EPOC** (Enfermedad Pulmonar Obstructiva Crónica) | 1.100 | 56.82 % | Mayoritaria (Predominante) |
| **🫁 Normal** (Condición Saludable) | 526 | 27.17 % | Moderada |
| **🫁 Neumonía** | 196 | 10.12 % | Minoritaria |
| **🫁 Asma** | 114 | 5.89 % | Críticamente Minoritaria |
| **Total General** | **1.936** | **100.00 %** | **Dataset Homologado** |

> 🛡️ **Nota de MLOps y Calidad de Datos:** Como se observa en la matriz de distribución, el dataset presenta un marcado desbalanceo de clases (típico en entornos médicos reales donde ciertas patologías crónicas tienen mayor tasa de registro). Para mitigar este sesgo y evitar que la red neuronal CNN-BiLSTM optimizara únicamente para la clase mayoritaria (EPOC), el pipeline implementa **técnicas de balanceo automático mediante ponderación de pesos en la función de pérdida (Class Weights)** durante el entrenamiento, garantizando que el modelo penalice con igual rigurosidad los fallos en clases minoritarias como el Asma o la Neumonía.

---
| Dataset                                      |       Pacientes | Población                        | Clases principales                                                       | Sitios de auscultación            |
| -------------------------------------------- | --------------: | -------------------------------- | ------------------------------------------------------------------------ | --------------------------------- |
| **ICBHI 2017 Respiratory Sound Database**    |             126 | Niños, adultos y adultos mayores | Normal, Asma, EPOC, Neumonía, Bronquiolitis, Bronquiectasias, URTI, etc. | 8 puntos torácicos estandarizados |
| **ALSD-Net (Annotated Lung Sounds Dataset)** |             112 | 21–90 años                       | Normal, Asma, EPOC, Bronquitis, Fibrosis pulmonar, Derrame pleural, etc. | 8 regiones anatómicas del tórax   |
| **Pulmonary (Lungs) Sound Dataset**          | No especificado | Niños, adultos y adultos mayores | Normal, Asma, EPOC, Neumonía, Derrame pleural, Roncus, Sibilancias, etc. | Región pulmonar general           |

> 🛡️ **Nota de MLOps y Calidad de Datos:** Los tres conjuntos de datos contienen grabaciones de sonidos respiratorios obtenidas mediante auscultación clínica. Aunque presentan diferencias en el número de pacientes, patologías y puntos de adquisición, todos fueron unificados mediante un proceso de preprocesamiento, normalización y mapeo de etiquetas para entrenar y evaluar los modelos de clasificación de forma consistente.
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




