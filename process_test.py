# Импорт стандартных библиотек
import os
import time
from pathlib import Path, PurePath

# Импорт сторонних библиотек
import cv2  # Библиотека для работы с изображениями и видео
import numpy as np  # Библиотека для работы с массивами
import pandas as pd  # Библиотека для работы с данными в формате таблиц
from tqdm import tqdm  # Библиотека для отображения прогресс-баров
import librosa  # Библиотека для анализа аудиоданных
import tensorflow as tf  # Библиотека для машинного обучения
from tensorflow.keras.applications import ResNet50  # Импорт предобученной модели ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input  # Функция для предобработки данных для ResNet50
from tensorflow.io import gfile  # Для работы с файловой системой
from scipy.fftpack import dct  # Дискретное косинусное преобразование
from scipy.spatial.distance import cosine, euclidean  # Метрики для вычисления расстояний
import moviepy.video.io.VideoFileClip as mp  # Для работы с видеофайлами
from sklearn.metrics.pairwise import cosine_similarity  # Для вычисления косинусного сходства

# Импорт библиотек PyTorch
import torch
import torch.nn.functional as F  # Функции для работы с нейросетями
import torchvision.transforms as transforms  # Трансформации для изображений

# Импорт моделей сегментации на PyTorch
import segmentation_models_pytorch as smp  # Библиотека для работы с сегментационными моделями

# Определяем пути к тестовым видео
test_video = r'C:\Users\Vladimir\PycharmProjects\ML\Kaggle\yappy\test_dataset_test_data_yappy\test_data_yappy\test_dataset\49577a11-51b9-490a-b1f0-df17335219de.mp4'
test_video2 = r'C:\Users\Vladimir\PycharmProjects\ML\Kaggle\yappy\test_dataset_test_data_yappy\test_data_yappy\test_dataset\da9783ba-ceac-47ed-9d8f-30b614e938dd.mp4'
print(test_video)
print(test_video2)

# Определяем класс для кодирования видео с использованием ResNet
class ResNetEncoder():
    def __init__(self):
        # Инициализация модели ResNet50 с предобученными весами
        self.model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

    # 1. Извлечение кадров из видео
    def extract_frames(self, video_path, max_frames=10):
        cap = cv2.VideoCapture(video_path)  # Открываем видео
        frames = []
        success, frame = cap.read()  # Считываем первый кадр
        count = 0
        while success and count < max_frames:  # Пока есть кадры и не превышено максимальное количество
            frames.append(frame)  # Добавляем кадр в список
            success, frame = cap.read()  # Считываем следующий кадр
            count += 1
        cap.release()  # Закрываем видеофайл
        return frames

    # 2. Извлечение гистограммы цветов
    def extract_color_histogram(self, frame):
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # Преобразуем BGR в HSV
        # Вычисляем гистограмму по цветам
        hist = cv2.calcHist([hsv_frame], [0, 1, 2], None, [50, 60, 70], [0, 180, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()  # Нормализуем и преобразуем в одномерный массив
        return hist

    # 3. Извлечение признаков с помощью предобученной модели ResNet50
    def extract_deep_features(self, frame):
        frame_resized = cv2.resize(frame, (224, 224))  # Изменяем размер кадра
        frame_preprocessed = preprocess_input(np.expand_dims(frame_resized, axis=0))  # Предобработка
        features = self.model.predict(frame_preprocessed, verbose=0)  # Извлечение признаков
        return features.flatten()  # Преобразуем в одномерный массив

    # 4. Комбинирование всех признаков (гистограмма цветов, ResNet)
    def extract_combined_features(self, frame):
        color_hist = self.extract_color_histogram(frame)  # Извлекаем гистограмму цветов
        deep_features = self.extract_deep_features(frame)  # Извлекаем глубокие признаки

        # Объединение всех признаков в один вектор
        combined_features = np.hstack((color_hist, deep_features))
        return combined_features

    # 5. Извлечение кадров из видео и получение средних признаков
    def encode(self, video_path, max_frames=10):
        frames = self.extract_frames(video_path, max_frames)  # Извлекаем кадры
        combined_features_list = [self.extract_combined_features(frame) for frame in frames]  # Извлекаем признаки для каждого кадра

        # Усредняем признаки по кадрам
        average_features = np.mean(combined_features_list, axis=0)
        return average_features

    # Функция для вычисления расстояния между векторами признаков
    def get_distance(self, emb1, emb2):
        similarity = 1 - cosine(emb1, emb2)  # Расстояние по косинусной метрике
        return similarity

# Создаем экземпляр класса ResNetEncoder
ResNetEnc = ResNetEncoder()
emb1 = ResNetEnc.encode(test_video)  # Кодируем первое видео
emb2 = ResNetEnc.encode(test_video2)  # Кодируем второе видео
ResNetEnc.get_distance(emb1, emb2)  # Вычисляем расстояние между векторами признаков

class InceptionEncoder():
    def __init__(self):
        self.model = tf.keras.applications.InceptionV3(weights='imagenet', include_top=False, pooling='avg')
    def frames_extract(self, video_file, start_off=0, sampling_fps=1, max_frame_cnt=60):
        """Извлекает кадры из входного видео."""
        if not os.path.exists(video_file):
            return None
          
        vcap = cv2.VideoCapture(video_file)
        fps = vcap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(fps / sampling_fps) if sampling_fps > 0 and sampling_fps < fps else 1
        frame_list, cnt = [], 0
        success, im = vcap.read()
      
        while success and cnt < max_frame_cnt:
            if cnt % frame_interval == 0:
                frame_list.append(im)
            cnt += 1
            success, im = vcap.read()

        return np.array(frame_list) if frame_list else None
    def feature_from_single_image_file(self, image_file):
        """Извлекает вектор признаков из одного изображения."""
      
        if not gfile.exists(image_file):
            print(f"Файл не существует: {image_file}")
            return None
          
        image_data = gfile.GFile(image_file, 'rb').read()
        image_tensor = tf.image.decode_jpeg(image_data, channels=3)
        image_tensor = tf.image.resize(image_tensor, (299, 299))  # Размер входа Inception-v3
        image_tensor = tf.expand_dims(image_tensor, axis=0)
        image_tensor = image_tensor / 255.0

        feature_tensor = self.model.signatures['serving_default'](image_tensor)['pool_3']
        return np.squeeze(feature_tensor.numpy())
    def feature_from_single_video_file(self, video_file, start_off=0, sampling_fps=1, max_frame_cnt=60, padding=True):
        """Извлекает векторы признаков из видеофайла."""
      
        if not gfile.exists(video_file):
            print(f"Файл не существует: {video_file}")
            return None
        frames = self.frames_extract(video_file, start_off, sampling_fps, max_frame_cnt)
        if frames is None:
            return None
          
        features = []
        for frame in frames:
            frame_tensor = tf.convert_to_tensor(frame, dtype=tf.uint8)
            frame_tensor = tf.image.resize(frame_tensor, (299, 299))
            frame_tensor = tf.expand_dims(frame_tensor, axis=0)
            frame_tensor = frame_tensor / 255.0
            feature_tensor = self.model(frame_tensor)
            features.append(np.squeeze(feature_tensor.numpy()))

        if padding and max_frame_cnt > len(features):
            zero_feat = np.zeros([2048], dtype=np.float32)
            features.extend([zero_feat] * (max_frame_cnt - len(features)))

        return np.array(features)
    def encode(self, video_path):
        video_features = self.feature_from_single_video_file(video_path, sampling_fps=1, max_frame_cnt=60, padding=True)
        mean_features = np.mean(video_features, axis=0)
        return mean_features
    def get_distance(self, emb1, emb2):
        similarity = cosine_similarity([emb1], [emb2])[0][0]
        return similarity

    def get_evc_distance(self, emb1, emb2):
        similarity = euclidean(emb1, emb2)
        return similarity


class AudioEncoder():
    def encode(self, video_path):
        video = mp.VideoFileClip(video_path)
        if video.audio is None:
            print(video_path)
            return np.zeros((128,))
        video.audio.write_audiofile("audio1.wav")
        y1, sr1 = librosa.load("audio1.wav", sr=None)
        S1 = librosa.feature.melspectrogram(y=y1, sr=sr1)
        mean_spectrum1 = np.mean(S1, axis=1)
        return mean_spectrum1
      
    def get_distance(self, spect1, spect2):
        similarity = 1 - cosine(spect1, spect2)
        return similarity
class UnetEncoder():
    def __init__(self):
        self.transform = transforms.Compose([transforms.Resize((256, 256))])
        unet_model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=3,
            classes=1
        )
        self.encoder = unet_model.encoder
    def frames_extract(self, video_file, sampling_fps=1, max_frame_cnt=60):
        """Извлекает кадры из видео с использованием OpenCV."""
        vcap = cv2.VideoCapture(video_file)
        fps = vcap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(fps / sampling_fps)
        frames = []
        while success and count < max_frame_cnt:
          if count % frame_interval == 0:
              frames.append(frame)
          success, frame = vcap.read()
          count += 1

        vcap.release()
        return frames
      
    def encode(self, video_path):
      frames = self.frames_extract(video_path)
      features = []

      for frame in frames:
          frame_tensor = self.transform(torch.Tensor(frame).permute(2, 0, 1)).unsqueeze(0)
          with torch.no_grad():
              feature = self.encoder(frame_tensor)[-1]  # Берем последний слой энкодера
          features.append(feature.squeeze(0))
          return torch.mean(torch.stack(features), dim=0)
        
    def get_distance(self, feature1, feature2):
        cosine_sim = F.cosine_similarity(feature1.view(1, -1), feature2.view(1, -1))
        return 1 - cosine_sim.item()

class LengthEncoder:
    def encode(self, video_file):
        cap = cv2.VideoCapture(video_file)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps
        cap.release()
        return np.array([duration])

    def get_distance(self, length1, length2):
        return abs(length1 - length2)







