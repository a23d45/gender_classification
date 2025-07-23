from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict
import io
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from gender_classifier import GenderClassifier

app = FastAPI(
    title="Gender Classification API",
    description="API для определения пола по фотографии",
    version="1.0.0"
)

model = GenderClassifier.load_model('gender_svm.pkl')

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictionResponse(BaseModel):
    gender: str
    probability: Dict[str, float]
    model_accuracy: float

class ModelInfoResponse(BaseModel):
    classes: list
    metrics: Dict[str, float]

@app.get("/model-info", 
         response_model=ModelInfoResponse, 
         tags=["Model Information"],
         summary="Получить информацию о модели",
         description="Возвращает список классов и метрики качества")
async def get_model_info():
    """Получение информации о модели"""
    if model is None:
        raise HTTPException(status_code=500, detail="Модель не доступна")

    return {
        "classes": model.encoder_classes.tolist(),
        "metrics": {
            "test_accuracy": model.metrics['test_accuracy'],
            "train_accuracy": model.metrics['train_accuracy'],
            "precision_male": model.metrics['classification_report']['male']['precision'],
            "recall_male": model.metrics['classification_report']['male']['recall'],
            "f1_male": model.metrics['classification_report']['male']['f1-score'],
            "precision_female": model.metrics['classification_report']['female']['precision'],
            "recall_female": model.metrics['classification_report']['female']['recall'],
            "f1_female": model.metrics['classification_report']['female']['f1-score'],
        }
    }

@app.post("/predict", 
          response_model=PredictionResponse, 
          tags=["Prediction"],
          summary="Определить пол по фотографии",
          description="Принимает изображение и возвращает предсказание пола с вероятностями")
async def predict(file: UploadFile = File(..., description="Изображение лица в формате JPG/PNG")):
    if model is None:
        raise HTTPException(status_code=500, detail="Модель не доступна")
    
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="Файл должен быть изображением")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        
        temp_path = "temp_prediction_image.jpg"
        image.save(temp_path)
        result = model.predict(temp_path)
        if os.path.exists(temp_path):
            os.remove(temp_path)

        return {
            "gender": result["gender"],
            "probability": result["probability"],
            "model_accuracy": result["metrics"]["test_accuracy"]
        }
    
    except Exception as e:
        print(f"ОШИБКА {e}")
        raise HTTPException(
            status_code=400, 
            detail="Не удалось обнаружить лицо на изображении. Убедитесь, что лицо хорошо видно."
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)