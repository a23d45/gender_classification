# Gender Classification Project

Проект для определения пола по фотографии с использованием FaceNet и SVM.

## Структура проекта

* dataset/ — Папка с датасетом изображений  
* test_img/ — Тестовые изображения для проверки  
* app.py — FastAPI веб-сервис для классификации   
* gender_classifier.py — Основной класс классификатора
* faceloading.py — класс для загрузки и обработки изображений  
* gender_classifier.ipynb — Jupyter notebook с исследованиями
* embeddings_data.pkl — Предварительно вычисленные эмбеддинги
* gender_svm.pkl — Обученная SVM модель
* requirements.txt — Зависимости
* facenet_keras.h5 — предобученная модель FaceNet

## Установка
### 1. Настройка виртуального окружения

#### Для Linux/macOS:
```bash
python -m venv venv
source venv/bin/activate
```

#### Для Windows:
```
python -m venv venv
.\venv\Scripts\activate
```
### 2. Установка зависимостей
```
pip install --upgrade pip
pip install --no-cache-dir -r requirements.txt
```
**Используется Python 3.6.2**

