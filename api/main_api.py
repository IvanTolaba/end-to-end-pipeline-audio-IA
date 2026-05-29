# Inicializa FastAPI y levanta Rutas.

import logging
from fastapi import FastAPI
from config.settings import API_TITLE, API_DESCRIPTION
from api.routes import inference
from api.services.predictor import predictor_service

# Inicializar Logs globales antes que nada
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

app = FastAPI(title=API_TITLE,description=API_DESCRIPTION)

# Cargar el modelo de IA al iniciar la aplicación (Uso del ciclo de vida de FastAPI)
@app.on_event("startup")
def startup_event():
    predictor_service.load_model()

# Incluir las rutas modulares
app.include_router(inference.router)


