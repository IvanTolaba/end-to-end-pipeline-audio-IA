# Initialize FastAPI and start routes.

import logging
from fastapi import FastAPI
from config.settings import API_TITLE, API_DESCRIPTION
from api.routes import inference
from api.services.predictor import predictor_service

# Initialize global logs 
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

app = FastAPI(title=API_TITLE,description=API_DESCRIPTION)

# Load the AI ​​model when the application starts
@app.on_event("startup")
def startup_event():
    predictor_service.load_model()

# Include modular routes
app.include_router(inference.router)


