
import cv2 as cv
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from mtcnn import MTCNN


class FaceLoading:
    def __init__(self, directory, labels):
        self.directory = directory
        self.target_size = (160,160)
        self.X = []
        self.Y = []
        self.labels = labels
        self.face_detector = MTCNN()

    def extract_face(self, filename):
        img = cv.imread(filename)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        result_detect = self.face_detector.detect_faces(img)
        try:
            x,y,w,h = result_detect[0]['box']
            x,y = abs(x), abs(y)
        except Exception as e:
            return None
        try:
            face = img[y:y+h, x:x+w]
        except Exception as e:
            print(f'extract face 2{e}')
        face_arr = cv.resize(face, self.target_size)
        return face_arr
    

    def load_faces(self, dir):
        FACES = []
        face_count = 0 
        print(f'dir = {os.listdir(dir)}')
        for image_name in os.listdir(dir):
            try:
                path = dir + image_name
                single_face = self.extract_face(path)
                if single_face is not None:
                    FACES.append(single_face)
                    face_count += 1
                if face_count % 200 == 0:
                    print(f'===dir: {dir}, img_num: {face_count}===')
            except Exception as e:
                print(f'load_faces {e}')  
        return FACES, face_count

    def load_classes(self):
        for label in self.labels:
            path = self.directory +'/'+ label +'/'
            FACES, face_count = self.load_faces(path)
            labels = [label for _ in range(face_count)]
            print(f"Loaded successfully: {len(labels)}")
            self.X.extend(FACES)
            self.Y.extend(labels)
        
        return np.asarray(self.X), np.asarray(self.Y)