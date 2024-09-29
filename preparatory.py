
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

class MovieNetEncoder():
    def __init__(self, video_file_1 ,video_file_2):
        self.video_file_1 = video_file_1
        self.video_file_2 = video_file_2
        self.model_id = 'a0'
        self.resolution = 224
        self.movinet_model = movinet.Movinet(model_id=self.model_id)
        self.movinet_model.trainable = False

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
                frame_resized = cv2.resize(frame, (224, 224))  # Изменение размера кадра под вход модели
                frames.append(frame_resized)
            success, frame = vcap.read()
            count += 1

        vcap.release()
        return np.array(frames)

    def prepare_video(self, frames):
        """Подготовка кадров видео для подачи в MoViNet."""
        frames = np.array(frames, dtype=np.float32)
        frames = frames / 255.0  # Нормализация пикселей
        frames = np.expand_dims(frames, axis=0)  # Добавляем размерность для пакета
        return frames

    def extract_movinet_features(self, video_tensor, model):
        """Извлекает эмбеддинги из видео с помощью модели MoViNet (извлекатель признаков)."""
        output = model(tf.convert_to_tensor(video_tensor))  # Применение модели
        features = output[0]['head']  # Доступ к признакам
        return np.squeeze(features.numpy())  # Преобразование в numpy

    def embed(self, video_file, sampling_fps=1, max_frame_cnt=60):
        frames = self.frames_extract(video_file)
        video_tensor = self.prepare_video(frames)
        video_features = self.extract_movinet_features(video_tensor, self.movinet_model)
        return video_features

    def compare_videos(self, video_path1, video_path2, max_frames=10):
        features_video1 = self.embed(video_path1)
        features_video2 = self.embed(video_path2)
        cos_similarity = 1 - cosine(features_video1, features_video2)
        euclidean_distance = euclidean(features_video1, features_video2)

        return cos_similarity, euclidean_distance

    def encode(self, sampling_fps=1, max_frame_cnt=60):
        cos_sim, euclid_dist = self.compare_videos(self.video_file_1, self.video_file_2)

        return euclid_dist
     

class AudioEncoder():
    def __init__(self, video_file_1 ,video_file_2):
        self.video_file_1 = video_file_1
        self.video_file_2 = video_file_2

    def make_specrtum(self, video):
        video1 = mp.VideoFileClip(video)
        video1.audio.write_audiofile("audio1.wav")

        y1, sr1 = librosa.load("audio1.wav", sr=None)
        S1 = librosa.feature.melspectrogram(y=y1, sr=sr1)
        mean_spectrum1 = np.mean(S1, axis=1)
        return mean_spectrum1

    def encode(self):
        similarity = 1 - cosine(self.make_specrtum(self.video_file_1), self.make_specrtum(self.video_file_2))
        return similarity
     

df = pd.read_csv('test.csv')
df.head()
     
# Импортируем необходимые библиотеки
import requests

def download_video(video_url):
    # Получаем ответ от URL видео
    response = requests.get(video_url, stream=True)

    # Извлекаем имя файла из URL
    filename = video_url.split("/")[-1]

    # Открываем локальный файл для записи загруженного контента
    with open(filename, 'wb') as file:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)

    print(f"{filename} загружен!")


# download_video(video_link)
     
from tqdm import tqdm
tqdm.pandas()
     
# Загружаем видео по уникальным ссылкам
for video_link in tqdm(df['link'].unique()[250:]):
    download_video(video_link)

# Загружаем предобученную модель Inception-v3
model = tf.keras.applications.InceptionV3(weights='imagenet', include_top=False, pooling='avg')

# Сохраняем модель в формате Keras
model.save("inception_v3_model.keras")

# Загружаем модель в формате Keras
inception_v3_model = tf.keras.models.load_model("inception_v3_model.keras")
     
import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.io import gfile

def frames_extract(video_file, start_off=0, sampling_fps=1, max_frame_cnt=60):
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

def create_graph(saved_model_dir):
    """Загружает модель Inception-v3 из директории SavedModel."""
    model = tf.saved_model.load(saved_model_dir)
    return model

def feature_from_single_image_file(image_file, model):
    """Извлекает вектор признаков из одного изображения."""
    if not gfile.exists(image_file):
        print(f"Файл не существует: {image_file}")
        return None

    image_data = gfile.GFile(image_file, 'rb').read()
    image_tensor = tf.image.decode_jpeg(image_data, channels=3)
    image_tensor = tf.image.resize(image_tensor, (299, 299))  # Размер входа Inception-v3

    image_tensor = tf.expand_dims(image_tensor, axis=0)
    image_tensor = image_tensor / 255.0

    feature_tensor = model.signatures['serving_default'](image_tensor)['pool_3']
    return np.squeeze(feature_tensor.numpy())

def feature_from_single_video_file(video_file, model=inception_v3_model, start_off=0, sampling_fps=1, max_frame_cnt=60, padding=True):
    """Извлекает вектор признаков из видеофайла."""
    if not gfile.exists(video_file):
        print(f"Файл не существует: {video_file}")
        return None

    sampling_fps = sampling_fps
    max_frame_cnt = max_frame_cnt
    start_off = start_off
    frames = frames_extract(video_file, start_off, sampling_fps, max_frame_cnt)
    if frames is None:
        return None

    features = []
    for frame in frames:
        frame_tensor = tf.convert_to_tensor(frame, dtype=tf.uint8)
        frame_tensor = tf.image.resize(frame_tensor, (299, 299))  # Размер входа Inception-v3
        frame_tensor = tf.expand_dims(frame_tensor, axis=0)
        frame_tensor = frame_tensor / 255.0

        # Получаем эмбеддинги напрямую через вызов модели
        feature_tensor = model(frame_tensor)
        features.append(np.squeeze(feature_tensor.numpy()))

    # Дополнение, если это необходимо
    if padding and max_frame_cnt > len(features):
        zero_feat = np.zeros([2048], dtype=np.float32)
        features.extend([zero_feat] * (max_frame_cnt - len(features)))

    # Усреднение эмбеддингов по кадрам
    averaged_features = np.mean(features, axis=0)

    return averaged_features
     
df['link_vid'] = df['uuid'].apply(lambda x: f'/content/{x}.mp4')
df['embeddings'] = df['link_vid'].progress_apply(feature_from_single_video_file)
     
# 100%|██████████| 1494/1494 [42:43<00:00,  1.72s/it]

df.to_csv('sample_data/embeds1500.csv')
     
train = pd.read_csv('train.csv')
train
     
# Создаем полный датафрейм, объединяя с обучающим набором
full_df = df.merge(train[['uuid', 'is_duplicate', 'duplicate_for']], on='uuid', how='left')
     
full_df.rename(columns={'uuid': 'video_1', 'duplicate_for': 'video_2'}, inplace=True)
     
full_df['is_duplicate'] = full_df['is_duplicate'].astype(int)
     
videos = list(full_df['video_1'])
     
# Находим индексы, где video_2 отсутствует
full_df.loc[full_df['video_2'].isna(), 'video_2'].index
     
# Случайным образом создаем дубликаты, чтобы минимизировать несоответствия
import random

for idx in full_df.loc[full_df['video_2'].isna(), 'video_1'].index:
    full_df.loc[idx, 'video_2'] = random.choice(videos)

# Проводим сравнение видео с помощью инкапсуляции в классы
# Пример использования класса для обработки видео
video_encoder_1 = MovieNetEncoder(full_df.loc[0, 'video_1'], full_df.loc[0, 'video_2'])
video_encoder_2 = AudioEncoder(full_df.loc[0, 'video_1'], full_df.loc[0, 'video_2'])

# Получаем результаты сравнения
full_df['Inception_cosin'] = full_df.progress_apply(lambda row: video_encoder_1.compare_videos(row['video_1'], row['video_2'])[0], axis=1)
full_df['Inception_eucld'] = full_df.progress_apply(lambda row: video_encoder_1.compare_videos(row['video_1'], row['video_2'])[1], axis=1)

# Финализируем датафрейм для сравнения и вывода
full_df['is_duplicate_pred'] = (full_df['Inception_cosin'] > 0.8).astype(int)  # Настройте порог, если необходимо

# Сохраняем датафрейм в CSV файл
full_df.to_csv('sample_data/final_video_comparison.csv', index=False)

# Отображаем обновленный датафрейм
full_df[['video_1', 'video_2', 'is_duplicate', 'is_duplicate_pred', 'Inception_cosin', 'Inception_eucld']].head()

# Вы можете также визуализировать результаты
import matplotlib.pyplot as plt
import seaborn as sns

# Создаем график для сравнения фактических и предсказанных дубликатов
plt.figure(figsize=(10, 6))
sns.countplot(data=full_df, x='is_duplicate', hue='is_duplicate_pred')
plt.title('Фактические vs Предсказанные Дубликаты')
plt.xlabel('Фактический Дубликат (0 = Нет, 1 = Да)')
plt.ylabel('Количество')
plt.legend(title='Предсказанный Дубликат', loc='upper right', labels=['Нет', 'Да'])
plt.show()

# При желании, оцените метрики производительности, если у вас есть тестовый набор
from sklearn.metrics import classification_report, confusion_matrix

# Предполагая, что у вас есть истинные метки в full_df['is_duplicate']
print(confusion_matrix(full_df['is_duplicate'], full_df['is_duplicate_pred']))
print(classification_report(full_df['is_duplicate'], full_df['is_duplicate_pred']))

# Функция для получения длины видео в секундах
def get_video_length(video_file):
    cap = cv2.VideoCapture(video_file)  # Открываем видеофайл
    fps = cap.get(cv2.CAP_PROP_FPS)  # Получаем количество кадров в секунду
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Получаем общее количество кадров
    cap.release()  # Закрываем видеофайл
    if fps > 0:  # Если fps корректный, считаем длину
        video_length = frame_count / fps  # Длина видео в секундах
    else:
        video_length = 0  # Если fps не найден, длина видео = 0
    return video_length  # Возвращаем длину видео

# Функция для расчета длины видео и их соотношения
def calculate_length_ratio(row):
    length_1 = get_video_length(row['link_vid'])  # Получаем длину первого видео
    length_2 = get_video_length(row['link_vid_2'])  # Получаем длину второго видео

    length_diff = abs(length_1 - length_2)  # Вычисляем разницу в длине
    max_length = max(length_1, length_2)  # Находим максимальную длину

    # Избегаем деления на 0
    if max_length > 0:
        length_ratio = length_diff / max_length  # Вычисляем соотношение длины
    else:
        length_ratio = 0  # Если максимальная длина = 0, соотношение длины = 0

    return length_ratio  # Возвращаем соотношение длины

full_df['length_ratio'] = df.apply(calculate_length_ratio, axis=1)  # Применяем функцию ко всем строкам

full_df.to_csv('data.csv')  # Сохраняем результаты в CSV файл


