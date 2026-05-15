
from fastapi import FastAPI, UploadFile, File
import shutil
import os

from model.predict import predict

app = FastAPI()

UPLOAD_PATH = "temp_audio.wav"


@app.get("/")
def home():
    return {"message": "Audio Classification API"}


@app.post("/predict")
async def predict_audio(file: UploadFile = File(...)):

    with open(UPLOAD_PATH, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    result = predict(UPLOAD_PATH)

    os.remove(UPLOAD_PATH)

    return {"prediction": result}