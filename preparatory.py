import torch
import segmentation_models_pytorch as smp
import cv2
import numpy as np
from scipy.fftpack import dct
import torch.nn.functional as F
import torchvision.transforms as transforms
import pandas as pd
from pathlib import Path, PurePath
import os
import time
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.io import gfile
from scipy.spatial.distance import cosine
import tensorflow as tf
from official.projects.movinet.modeling import movinet
import librosa
import moviepy.video.io.VideoFileClip as mp

class UnetEncoder():
    def __init__(self, video_1: str, video_2: str):
        # Инициализация преобразования и извлечение кадров из видео
        self.transform = transforms.Compose([
            transforms.Resize((256, 256))
        ])
        self.frames1 = self.frames_extract(video_1)
        self.frames2 = self.frames_extract(video_2)

        # Создание модели Unet с ResNet34 в качестве энкодера
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
        success, frame = vcap.read()
        count = 0

        while success and count < max_frame_cnt:
            if count % frame_interval == 0:
                frames.append(frame)
            success, frame = vcap.read()
            count += 1

        vcap.release()
        return frames

    def hamming_distance(self, tensor1, tensor2):
        # Преобразование тензоров в одномерный вид и вычисление косинусного расстояния
        tensor1_flat = tensor1.view(-1, 512)
        tensor2_flat = tensor2.view(-1, 512)
        cosine_sim = F.cosine_similarity(tensor1_flat, tensor2_flat, dim=1)
        cosine_distance = 1 - cosine_sim
        return cosine_distance

    def encode(self):
        frames1 = self.frames1
        frames2 = self.frames2
        similar_frames = 0
        total_frames = 0
        sum = 0
        k = 1

        while True:
            if len(frames1) == k-1 or len(frames2) == k-1:
                break
            frame1 = self.transform(torch.Tensor(frames1[k-1]).permute(2, 0, 1)).unsqueeze(0)
            frame2 = self.transform(torch.Tensor(frames2[k-1]).permute(2, 0, 1)).unsqueeze(0)

            vec1 = self.encoder(frame1)
            vec2 = self.encoder(frame2)
            sdistance = self.hamming_distance(vec1[5], vec2[5])
            k += 1
            sum += sdistance.mean()
        
        return 1 - float(sum/k)


class HashEncoder():
    def __init__(self, v1_path, v2_path):
        # Инициализация путей к видеофайлам
        self.v1_path = v1_path
        self.v2_path = v2_path

    def preprocess_frame(self, frame, size=32):
        # Преобразование кадра в градации серого и изменение его размера
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (size, size))
        return resized

    def compute_dct(self, image):
        # Вычисление DCT (дискретное косинусное преобразование) изображения
        return dct(dct(image.T, norm='ortho').T, norm='ortho')

    def create_hash(self, dct_block, hash_size=8):
        # Создание хеша из DCT-блока
        dct_low = dct_block[:hash_size, :hash_size]
        med = np.median(dct_low)
        return (dct_low > med).flatten()

    def compute_frame_hash(self, frame):
        # Вычисление хеша для отдельного кадра
        preprocessed = self.preprocess_frame(frame)
        dct_result = self.compute_dct(preprocessed)
        return self.create_hash(dct_result)

    def hamming_distance(self, hash1, hash2):
        # Вычисление расстояния Хэмминга между двумя хешами
        return np.sum(hash1 != hash2)

    def extract_frames(self, video_path, num_frames_per_sec=1):
        # Извлечение кадров из видео с заданной частотой
        cap = cv2.VideoCapture(video_path)
        frames = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration_seconds = int(total_frames // fps)
        indices = np.linspace(0, total_frames - 1, num_frames_per_sec * duration_seconds, dtype=int)

        for i in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                frames.append(frame)

        cap.release()
        return frames

    def compute_similarity_matrix(self, frames1, frames2, threshold=0.8):
        # Вычисление матрицы схожести между двумя наборами кадров
        if len(frames1) > len(frames2):
            frames1, frames2 = frames2, frames1

        hashes1 = [self.compute_frame_hash(frame) for frame in frames1]
        hashes2 = [self.compute_frame_hash(frame) for frame in frames2]

        similarity_matrix = np.zeros((len(hashes1), len(hashes2)))
        similar_frames = 0
        
        for i, hash1 in enumerate(hashes1):
            count_in_row = 0
            for j, hash2 in enumerate(hashes2):
                distance = self.hamming_distance(hash1, hash2)
                similarity = 1 - (distance / len(hash1))  # Нормализация (64 бита в хеше)
                if similarity > threshold:
                    count_in_row += 1
                if count_in_row > 0:
                    similar_frames += 1
                    break

                similarity_matrix[i, j] = similarity

        similarity_score = similar_frames / similarity_matrix.shape[0]
        return similarity_matrix, similarity_score

    def compare_videos(self, video1_path, video2_path, num_frames=3):
        # Сравнение двух видео на основе извлеченных кадров
        frames1 = self.extract_frames(video1_path, num_frames)
        frames2 = self.extract_frames(video2_path, num_frames)

        similarity_matrix, similarity_score = self.compute_similarity_matrix(frames1, frames2)
        return similarity_matrix, similarity_score

    def encode(self):
        # Основной метод для кодирования, который возвращает оценку схожести
        if not os.path.exists(self.v1_path) or not os.path.exists(self.v2_path):
            return 0

        t1 = time.time()
        similarity_matrix, similarity_score = self.compare_videos(self.v1_path, self.v2_path)
        return similarity_score


class ResNetEncoder():
    def __init__(self, video1_path, video2_path):
        # Инициализация модели ResNet50 и путей к видеофайлам
        self.model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
        self.video1_path = video1_path
        self.video2_path = video2_path

    def extract_frames(self, video_path, max_frames=10):
        # Извлечение кадров из видео
        cap = cv2.VideoCapture(video_path)
        frames = []
        success, frame = cap.read()
        count = 0
        while success and count < max_frames:
            frames.append(frame)
            success, frame = cap.read()
            count += 1
        cap.release()
        return frames

    def extract_color_histogram(self, frame):
        # Извлечение цветовой гистограммы из кадра
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv_frame], [0, 1, 2], None, [50, 60, 70], [0, 180, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        return hist

    def extract_deep_features(self, frame):
        # Извлечение глубоких признаков с помощью ResNet50
        frame_resized = cv2.resize(frame, (224, 224))
        frame_preprocessed = preprocess_input(np.expand_dims(frame_resized, axis=0))
        features = self.model.predict(frame_preprocessed, verbose=0)
        return features.flatten()

    def extract_combined_features(self, frame):
        # Объединение признаков цветовой гистограммы и глубоких признаков
        color_hist = self.extract_color_histogram(frame)
        deep_features = self.extract_deep_features(frame)
        combined_features = np.concatenate((color_hist, deep_features))
        return combined_features

    def compare_videos(self):
        # Сравнение видео
        frames1 = self.extract_frames(self.video1_path)
        frames2 = self.extract_frames(self.video2_path)
        total_distance = 0

        for frame1, frame2 in zip(frames1, frames2):
            combined_features1 = self.extract_combined_features(frame1)
            combined_features2 = self.extract_combined_features(frame2)
            distance = cosine(combined_features1, combined_features2)
            total_distance += distance

        return total_distance / len(frames1)  # Возврат среднего расстояния

    def encode(self):
        # Основной метод для кодирования, который возвращает оценку схожести
        return self.compare_videos()

