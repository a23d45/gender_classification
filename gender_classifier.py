import os
import pickle
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from keras.models import load_model

from faceloading import FaceLoading

class GenderClassifier:
    def __init__(self, dataset_path=None, facenet_model_path='facenet_keras.h5', embeddings_path='embeddings_data.pkl'):
        self.__face_loader = FaceLoading(None, ['male', 'female'])
        self.__embedder = load_model(facenet_model_path)
        self.__svm = SVC(kernel='rbf', probability=True, random_state=42, class_weight='balanced')
        self.__encoder = LabelEncoder()
        self.__metrics = {}
        self.__X_test = None
        self.__y_test = None
        self.__embeddings = None
        self.__y = None
        self.__y_encoded = None
        self.__X = None
        
        if os.path.exists(embeddings_path):
            print(f'Найден файл с эмбеддингами: {embeddings_path}')
            self.__embeddings, self.__y = self.__load_embeddings(embeddings_path)
            self.__y_encoded = self.__encoder.fit_transform(self.__y)
            print(f'Загружено {len(self.__embeddings)} эмбеддингов')
        elif dataset_path:
            print('Файл с эмбеддингами не найден, загружаем из датасета')
            self.__load_data(dataset_path, embeddings_path)
        else:
            raise ValueError('Необходимо указать dataset_path или предоставить файл с эмбеддингами')

    @property
    def metrics(self):
        '''Геттер для получения метрик модели'''
        return self.__metrics.copy()

    @property
    def encoder_classes(self):
        '''Геттер для получения классов энкодера'''
        return self.__encoder.classes_

    @property
    def test_accuracy(self):
        '''Геттер для получения точности на тестовой выборке'''
        return self.__metrics.get('test_accuracy', None)

    @property
    def train_accuracy(self):
        '''Геттер для получения точности на тренировочной выборке'''
        return self.__metrics.get('train_accuracy', None)

    def __load_embeddings(self, embeddings_path):
        '''Загружает эмбеддинги из файла'''
        with open(embeddings_path, 'rb') as f:
            data = pickle.load(f)
        return data['embeddings'], data['labels']

    def __load_data(self, dataset_path, embeddings_path):
        '''Загружает и обрабатывает данные, сохраняет эмбеддинги'''
        loader = FaceLoading(dataset_path, ['male', 'female'])
        self.__X, self.__y = loader.load_classes()
        self.__y_encoded = self.__encoder.fit_transform(self.__y)
        self.__embeddings = np.array([self.__get_embedding(face) for face in self.__X])
        self.__save_embeddings(embeddings_path)
        
    def __save_embeddings(self, embeddings_path):
        '''Сохраняет эмбеддинги в файл'''
        data = {
            'embeddings': self.__embeddings,
            'labels': self.__y
        }
        with open(embeddings_path, 'wb') as f:
            pickle.dump(data, f)
        print(f'Эмбеддинги сохранены в {embeddings_path}')

    def __get_embedding(self, face_img):
        '''Извлекает эмбеддинг лица'''
        face_img = face_img.astype('float32')
        mean, std = face_img.mean(), face_img.std()
        face_img = (face_img - mean) / std
        face_img = np.expand_dims(face_img, axis=0)
        return self.__embedder.predict(face_img)[0]

    def train(self, test_size=0.2, random_state=17):
        '''Обучает модель на загруженных данных'''
        X_train, self.__X_test, y_train, self.__y_test = train_test_split(
            self.__embeddings, self.__y_encoded, 
            test_size=test_size, random_state=random_state,
            stratify=self.__y_encoded
        )
        
        self.__svm.fit(X_train, y_train)
        
        y_pred = self.__svm.predict(self.__X_test)
        
        train_acc = self.__svm.score(X_train, y_train)
        test_acc = accuracy_score(self.__y_test, y_pred)
        
        self.__metrics = {
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'classification_report': classification_report(
                self.__y_test, y_pred, 
                target_names=self.__encoder.classes_,
                output_dict=True
            ),
            'overfitting_warning': (train_acc - test_acc) > 0.15
        }
        
        print('Метрики обучения:')
        print(f'Train accuracy: {train_acc:.4f}')
        print(f'Test accuracy: {test_acc:.4f}')
        if self.__metrics['overfitting_warning']:
            print('Предупреждение: возможен переобучение модели!')
        
        self.__save_model('gender_svm.pkl')

    def predict(self, image_path):
        '''Предсказывает пол по изображению'''
        face = self.__face_loader.extract_face(image_path)
        if face is None:
            return {'error': 'Не удалось обнаружить лицо'}
        
        embedding = self.__get_embedding(face)
        proba = self.__svm.predict_proba([embedding])[0]
        
        # Получаем индексы классов
        male_idx = self.__encoder.transform(['male'])[0]
        female_idx = self.__encoder.transform(['female'])[0]
        
        # Создаем словарь вероятностей
        proba_dict = {
            'male': float(proba[np.where(self.__svm.classes_ == male_idx)[0][0]]),
            'female': float(proba[np.where(self.__svm.classes_ == female_idx)[0][0]])
        }
        
        predicted_gender = max(proba_dict, key=proba_dict.get)
        
        return {
            'gender': predicted_gender,
            'probability': proba_dict,
            'metrics': self.metrics 
        }

    def __save_model(self, svm_path):
        '''Сохраняет модель SVM и метрики'''
        model_data = {
            'svm': self.__svm,
            'encoder': self.__encoder,
            'metrics': self.__metrics
        }
        with open(svm_path, 'wb') as f:
            pickle.dump(model_data, f)

    @classmethod
    def load_model(cls, svm_path, facenet_model_path='facenet_keras.h5'):
        '''Загружает сохраненную модель'''
        with open(svm_path, 'rb') as f:
            model_data = pickle.load(f)
        
        model = cls(facenet_model_path=facenet_model_path)
        model.__svm = model_data['svm']
        model.__encoder = model_data['encoder']
        model.__metrics = model_data['metrics']
        return model



if __name__ == "__main__":
    try:
        model = GenderClassifier.load_model("gender_svm.pkl")
    except FileNotFoundError:
        model = GenderClassifier(dataset_path="./dataset", embeddings_path='embeddings.pkl')
        model.train()
    print(model.metrics)