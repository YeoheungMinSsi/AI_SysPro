import os
import cv2
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2

train_dir = r'C:\Users\USER\.cache\kagglehub\datasets\volkandl\car-brand-logos\versions\1\Car_Brand_Logos\Train'
test_dir = r'C:\Users\USER\.cache\kagglehub\datasets\volkandl\car-brand-logos\versions\1\Car_Brand_Logos\Test'


# 데이터 로딩 함수 (개선됨)
def load_images(folder, class_map):
    images = []
    labels = []
    class_counts = {brand: 0 for brand in class_map.keys()}

    for brand, index in class_map.items():
        brand_path = os.path.join(folder, brand)
        if os.path.isdir(brand_path):
            for img_file in os.listdir(brand_path):
                img_path = os.path.join(brand_path, img_file)
                try:
                    img = cv2.imread(img_path)
                    if img is None:
                        continue

                    # 이미지 전처리 강화
                    img = cv2.resize(img, (64, 64))

                    # 가우시안 블러로 노이즈 제거
                    img = cv2.GaussianBlur(img, (3, 3), 0)

                    # CLAHE (Contrast Limited Adaptive Histogram Equalization)
                    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
                    img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

                    # 정규화
                    img = img.astype(np.float32) / 255.0

                    # 표준화 추가
                    img = (img - 0.5) / 0.5

                    images.append(img)
                    labels.append(index)
                    class_counts[brand] += 1

                except Exception as e:
                    print(f"⚠️ 손상된 파일 건너뛰기: {img_path}")

    print("클래스별 데이터 수:")
    for brand, count in class_counts.items():
        print(f"{brand}: {count}")

    return np.array(images), np.array(labels)


# 강화된 데이터 증강
def augment_data_enhanced(x_train, y_train, augment_factor=3):
    """강화된 데이터 증강"""
    augmented_x = []
    augmented_y = []

    for i in range(len(x_train)):
        # 원본 데이터
        augmented_x.append(x_train[i])
        augmented_y.append(y_train[i])

        # 증강된 데이터 생성
        for _ in range(augment_factor):
            img = x_train[i].reshape(64, 64, 3)

            # 1. 회전 (작은 각도)
            angle = np.random.uniform(-15, 15)
            center = (32, 32)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            img_rotated = cv2.warpAffine(img, M, (64, 64))

            # 2. 이동
            tx = np.random.randint(-5, 6)
            ty = np.random.randint(-5, 6)
            M_translate = np.float32([[1, 0, tx], [0, 1, ty]])
            img_translated = cv2.warpAffine(img_rotated, M_translate, (64, 64))

            # 3. 노이즈 추가
            noise = np.random.normal(0, 0.01, img_translated.shape)
            img_noisy = np.clip(img_translated + noise, -1, 1)

            # 4. 밝기/대비 조정
            alpha = np.random.uniform(0.8, 1.2)  # 대비
            beta = np.random.uniform(-0.1, 0.1)  # 밝기
            img_bright = np.clip(alpha * img_noisy + beta, -1, 1)

            augmented_x.append(img_bright.flatten())
            augmented_y.append(y_train[i])

    return np.array(augmented_x), np.array(augmented_y)


# 데이터 로딩
class_map = {brand: idx for idx, brand in enumerate(os.listdir(train_dir))}
num_classes = len(class_map)
print(f"클래스 수: {num_classes}")

x_train, y_train = load_images(train_dir, class_map)
x_test, y_test = load_images(test_dir, class_map)

print(f"원본 훈련 데이터: {x_train.shape}")
print(f"원본 테스트 데이터: {x_test.shape}")

# 이미지 평탄화
sample_num = x_train.shape[1] * x_train.shape[2] * x_train.shape[3]
x_train_flat = x_train.reshape(-1, sample_num)
x_test_flat = x_test.reshape(-1, sample_num)

# 강화된 데이터 증강 적용
print("강화된 데이터 증강 중...")
x_train_aug, y_train_aug = augment_data_enhanced(x_train_flat, y_train, augment_factor=2)
print(f"증강 후 훈련 데이터: {x_train_aug.shape}")

# 클래스 가중치 계산
class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(y_train_aug),
    y=y_train_aug
)
class_weight_dict = dict(enumerate(class_weights))
print(f"클래스 가중치: {class_weight_dict}")

# 원핫 인코딩
y_train_cat = to_categorical(y_train_aug, num_classes)
y_test_cat = to_categorical(y_test, num_classes)

# 훈련/검증 분할
x_train_final, x_val, y_train_final, y_val = train_test_split(
    x_train_aug, y_train_cat, test_size=0.15, random_state=42, stratify=y_train_cat
)

# 더 강력한 MLP 모델
model = Sequential([
    # 첫 번째 블록
    Dense(1024, input_shape=(sample_num,), kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    tf.keras.layers.Activation('relu'),
    Dropout(0.4),

    # 두 번째 블록
    Dense(512, kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    tf.keras.layers.Activation('relu'),
    Dropout(0.35),

    # 세 번째 블록
    Dense(256, kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    tf.keras.layers.Activation('relu'),
    Dropout(0.3),

    # 네 번째 블록
    Dense(128, kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    tf.keras.layers.Activation('relu'),
    Dropout(0.25),

    # 다섯 번째 블록
    Dense(64, kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    tf.keras.layers.Activation('relu'),
    Dropout(0.2),

    # 출력층
    Dense(num_classes, activation='softmax')
])

# 개선된 학습률 스케줄링
initial_lr = 0.003
lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=initial_lr,
    decay_steps=len(x_train_final) // 16 * 100,  # 전체 에포크 고려
    alpha=0.1
)

# 콜백 설정
early_stopping = EarlyStopping(
    monitor='val_accuracy',
    patience=25,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=12,
    min_lr=1e-7,
    verbose=1
)

# 모델 컴파일
model.compile(
    loss='categorical_crossentropy',
    optimizer=Adam(learning_rate=lr_schedule),
    metrics=['accuracy']
)

print(f"모델 파라미터 수: {model.count_params():,}")

# 모델 훈련
print("모델 훈련 시작...")
hist = model.fit(
    x_train_final, y_train_final,
    epochs=100,
    batch_size=16,  # 배치 크기 감소
    validation_data=(x_val, y_val),
    class_weight=class_weight_dict,  # 클래스 가중치 적용
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

# 평가
test_loss, test_acc = model.evaluate(x_test_flat, y_test_cat, verbose=0)
print(f"\n📊 최종 테스트 정확도: {test_acc:.4f} ({test_acc * 100:.2f}%)")

# 학습 과정 시각화
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(hist.history['accuracy'], label='Training Accuracy')
plt.plot(hist.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(hist.history['loss'], label='Training Loss')
plt.plot(hist.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 3, 3)
plt.plot(hist.history['lr'], label='Learning Rate')
plt.title('Learning Rate Schedule')
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
plt.yscale('log')
plt.legend()

plt.tight_layout()
plt.show()

# 클래스별 성능 분석
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

y_pred = model.predict(x_test_flat)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test_cat, axis=1)

print("\n📈 클래스별 성능:")
class_names = list(class_map.keys())
print(classification_report(y_true_classes, y_pred_classes, target_names=class_names))

# 혼동 행렬 시각화
plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_true_classes, y_pred_classes)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()
