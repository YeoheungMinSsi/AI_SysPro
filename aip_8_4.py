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


train_dir = r'C:\Users\USER\.cache\kagglehub\datasets\volkandl\car-brand-logos\versions\1\Car_Brand_Logos\Train'
test_dir = r'C:\Users\USER\.cache\kagglehub\datasets\volkandl\car-brand-logos\versions\1\Car_Brand_Logos\Test'


# 데이터 로딩 함수 (개선됨)
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
                        continue
                    img = cv2.resize(img, (64, 64))
                    # 히스토그램 평활화로 대비 개선
                    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
                    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
                    img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

                    img = img / 255.0  # 정규화
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


# 데이터 로딩
class_map = {brand: idx for idx, brand in enumerate(os.listdir(train_dir))}
num_classes = len(class_map)
print(f"클래스 수: {num_classes}")
print(f"클래스: {list(class_map.keys())}")

x_train, y_train = load_images(train_dir, class_map)
x_test, y_test = load_images(test_dir, class_map)

print(f"원본 훈련 데이터: {x_train.shape}")
print(f"원본 테스트 데이터: {x_test.shape}")

# 이미지 평탄화
sample_num = x_train.shape[1] * x_train.shape[2] * x_train.shape[3]
x_train_flat = x_train.reshape(-1, sample_num)
x_test_flat = x_test.reshape(-1, sample_num)

# 데이터 증강 적용
print("데이터 증강 중...")
x_train_aug, y_train_aug = augment_data(x_train_flat, y_train, augment_factor=1)
print(f"증강 후 훈련 데이터: {x_train_aug.shape}")

# 원핫 인코딩
y_train_cat = to_categorical(y_train_aug, num_classes)
y_test_cat = to_categorical(y_test, num_classes)

# 훈련/검증 분할
x_train_final, x_val, y_train_final, y_val = train_test_split(
    x_train_aug, y_train_cat, test_size=0.2, random_state=42, stratify=y_train_cat
)

# 개선된 MLP 모델
model = Sequential([
    # 첫 번째 블록 - 더 큰 용량
    # Dense(512, input_shape=(sample_num,)),
    # BatchNormalization(),
    # tf.keras.layers.Activation('relu'),
    # Dropout(0.3),  # 드롭아웃 감소

    # 두 번째 블록
    # Dense(256, input_shape=(sample_num,)),
    # BatchNormalization(),
    # tf.keras.layers.Activation('relu'),
    # Dropout(0.25),

    # 세 번째 블록
    Dense(128, input_shape=(sample_num,)),
    BatchNormalization(),
    tf.keras.layers.Activation('relu'),
    Dropout(0.2),

    # 네 번째 블록
    Dense(64),
    BatchNormalization(),
    tf.keras.layers.Activation('relu'),
    Dropout(0.15),

    # 출력층
    Dense(num_classes, activation='softmax')
])

# 개선된 학습률 스케줄링
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.002,  # 초기 학습률 증가
    decay_steps=len(x_train_final) // 32 * 15,  # 15 에포크마다 감소
    decay_rate=0.95  # 더 천천히 감소
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

# 모델 컴파일
model.compile(
    loss='categorical_crossentropy',
    optimizer=Adam(learning_rate=lr_schedule),
    metrics=['accuracy']
)

# print(model.summary())

# 모델 훈련
print("모델 훈련 시작...")
hist = model.fit(
    x_train_final, y_train_final,
    epochs=50,
    batch_size=32,
    validation_data=(x_val, y_val),
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

# 평가
test_loss, test_acc = model.evaluate(x_test_flat, y_test_cat, verbose=0)
print(f"\n📊 최종 테스트 정확도: {test_acc:.4f} ({test_acc * 100:.2f}%)")

# 학습 과정 시각화
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(hist.history['accuracy'], label='Training Accuracy')
plt.plot(hist.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(hist.history['loss'], label='Training Loss')
plt.plot(hist.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# 클래스별 성능 분석
from sklearn.metrics import classification_report

y_pred = model.predict(x_test_flat)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test_cat, axis=1)

print("\n📈 클래스별 성능:")
class_names = list(class_map.keys())
print(classification_report(y_true_classes, y_pred_classes, target_names=class_names))
