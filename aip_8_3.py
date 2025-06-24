import os
import cv2
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 과적합이 계속 일어남

# 이미지 파일 위치
train_dir = r'C:\Users\USER\.cache\kagglehub\datasets\volkandl\car-brand-logos\versions\1\Car_Brand_Logos\Train'
test_dir = r'C:\Users\USER\.cache\kagglehub\datasets\volkandl\car-brand-logos\versions\1\Car_Brand_Logos\Test'

# 파일을 찾아오는 함수
def load_images(folder, class_map):
    images = []
    labels = []
    for brand, index in class_map.items():
        brand_path = os.path.join(folder, brand)
        if os.path.isdir(brand_path):
            for img_file in os.listdir(brand_path):
                img_path = os.path.join(brand_path, img_file)
                try:
                    img = cv2.imread(img_path)
                    if img is None:
                        raise ValueError(f"Invalid image: {img_path}")
                    # img = cv2.resize(img, (224, 224))
                    # img = cv2.resize(img, (128, 128))  # 이미지 크키가 커지면 학습률도 올라감 64보단 낳아짐
                    img = cv2.resize(img, (64, 64))
                    img = img / 255.0  # 사전 정규화
                    images.append(img)
                    labels.append(index)
                except Exception as e:
                    print(f"⚠️ 손상된 파일 건너뛰기: {img_path}")
    return np.array(images), np.array(labels)


# 데이터 증강 (MLP용)
def augment_data(x_train, y_train, augment_factor=2):
    """간단한 데이터 증강"""
    augmented_x = []
    augmented_y = []

    for i in range(len(x_train)):
        # 원본 데이터
        augmented_x.append(x_train[i])
        augmented_y.append(y_train[i])

        # 증강된 데이터 생성
        for _ in range(augment_factor):
            img = x_train[i].reshape(64, 64, 3)

            # 노이즈 추가
            noise = np.random.normal(0, 0.02, img.shape)
            img_noisy = np.clip(img + noise, 0, 1)

            # 밝기 조정
            brightness = np.random.uniform(0.8, 1.2)
            img_bright = np.clip(img * brightness, 0, 1)

            augmented_x.append(img_noisy.flatten())
            augmented_y.append(y_train[i])

            augmented_x.append(img_bright.flatten())
            augmented_y.append(y_train[i])

    return np.array(augmented_x), np.array(augmented_y)


class_map = {brand: idx for idx, brand in enumerate(os.listdir(train_dir))}
num_classes = len(class_map)  # 클레스 길이 확인

# 각 학습, 테스트 데이터를 변수에 입력, 동시에 정규화 수행
x_train, y_train = load_images(train_dir, class_map)
x_test, y_test = load_images(test_dir, class_map)

# shape[0] 이미지의 개수 / 244, 244은 세로, 가로 픽셀 수 / 3은 rgb 컬러 수  // 64 64로 변경
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], x_train.shape[3])
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], x_test.shape[3])

sample_num = x_train.shape[1] * x_train.shape[2] * x_train.shape[3]

# # 이미지 평탄화 3D -> 1D
# x_train = x_train.reshape(x_train.shape[0], -1)  # (샘플수, 224*224*3)
# x_test = x_test.reshape(x_test.shape[0], -1)
x_train_flat = x_train.reshape(-1, sample_num)  # (샘플수, 224*224*3)
x_test_flat = x_test.reshape(-1, sample_num)

# 이미지 증강
x_train_aug, y_train_aug = augment_data(x_train_flat, y_train, augment_factor=1)

# 원핫 인코딩 각 레이블을 0와 1로만 이루어진 벡터로 변환 / 각 픽셀을 0~8까지의 값으로 변환(클레스가 8개이기 때문)
y_train_cat = to_categorical(y_train, num_classes)
y_test_cat = to_categorical(y_test, num_classes)

## 여기까지는 문제 없음

# 훈련/검증 분할
x_train_final, x_val, y_train_final, y_val = train_test_split(
    x_train_aug, y_train,
    test_size=0.2,
    random_state=42,
    stratify=y_train
)

# 모델 만들기
model = Sequential([
    Dense(256, input_shape=(sample_num,)),  # 활성화 함수 분리
    BatchNormalization(),
    tf.keras.layers.Activation('relu'),  # 명시적 활성화 층
    Dropout(0.3),

    # Dense(128, input_shape=(sample_num,)),  # 활성화 함수 분리
    Dense(128),
    BatchNormalization(),
    tf.keras.layers.Activation('relu'),
    Dropout(0.25),

    Dense(64),
    BatchNormalization(),
    tf.keras.layers.Activation('relu'),
    Dropout(0.2),

    # Dense(32),
    # BatchNormalization(),
    # tf.keras.layers.Activation('relu'),
    # Dropout(0.15),

    # Dense(num_classes, activation='sigmoid')  # 테스트 정확도가 softmax에 비해 조금 떨어짐
    Dense(num_classes, activation='softmax')
])

# gkrtmq tmzpwnffld 학습 스케줄링
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.002,
    # decay_steps=1000,
    # decay_steps = len(x_train) // x_train.shape[1] * 5,
    # decay_steps = len(x_train) // 224 * 5,  # 5 에포크 마다 감소
    # decay_steps = len(x_train) // 128 * 5,  # 5 에포크 마다 감소
    decay_steps = len(x_train_final) // 64 * 5,  # 5 에포크 마다 감소
    decay_rate=0.9
)

# 콜백 설정
early_stopping = EarlyStopping(
    monitor='val_accuracy',
    patience=20,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.7,
    patience=10,
    min_lr=1e-6,
    verbose=1
)

model.compile(
    loss='categorical_crossentropy',
    optimizer=Adam(learning_rate=lr_schedule),
    metrics=['accuracy']
)

hist = model.fit(
    x_train_final, y_train_final,
    epochs=150,
    batch_size=32,
    validation_data=(x_val, y_val),
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

# 평가
test_loss, test_acc = model.evaluate(x_test_flat, y_test_cat, verbose=0)
print(f"\n📊 최종 테스트 정확도: {test_acc:.4f} ({test_acc*100:.2f}%)")
