import os
import cv2
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import seaborn as sns

train_dir = r'C:\Users\USER\.cache\kagglehub\datasets\volkandl\car-brand-logos\versions\1\Car_Brand_Logos\Train'
test_dir = r'C:\Users\USER\.cache\kagglehub\datasets\volkandl\car-brand-logos\versions\1\Car_Brand_Logos\Test'

# 데이터 로딩 함수 (CNN 전용)
def load_images_cnn(folder, class_map):
    images = []
    labels = []
    for brand, idx in class_map.items():
        brand_path = os.path.join(folder, brand)
        if os.path.isdir(brand_path):
            for img_file in os.listdir(brand_path):
                img_path = os.path.join(brand_path, img_file)
                try:
                    img = cv2.imread(img_path)
                    if img is None: continue

                    # 이미지 전처리 강화
                    img = cv2.resize(img, (64, 64))
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = img / 255.0

                    # 가우시안 블러 적용
                    img = cv2.GaussianBlur(img, (3, 3), 0)

                    images.append(img)
                    labels.append(idx)
                except Exception as e:
                    print(f"⚠️ 손상된 파일 건너뛰기: {img_path}")
    return np.array(images), np.array(labels)


# 데이터 증강 설정
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.2,
    fill_mode='nearest'
)

# 클래스 매핑
class_map = {brand: idx for idx, brand in enumerate(os.listdir(train_dir))}
num_classes = len(class_map)

# 데이터 로딩 (4D 텐서 유지)
x_train, y_train = load_images_cnn(train_dir, class_map)
x_test, y_test = load_images_cnn(test_dir, class_map)

# 원핫 인코딩
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)

# CNN 모델 아키텍처
model = tf.keras.Sequential([
    # 특징 추출부
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Dropout(0.3),

    # 분류기
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(num_classes, activation='softmax')
])

# 최적화 설정
optimizer = tf.keras.optimizers.Adam(
    learning_rate=tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.001,
        decay_steps=len(x_train) // 32 * 10,
        decay_rate=0.9
    )
)

# 콜백 설정
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=8)
]

# 모델 컴파일
model.compile(
    loss='categorical_crossentropy',
    optimizer=optimizer,
    metrics=['accuracy']
)

# 데이터 증강 파이프라인
train_generator = datagen.flow(x_train, y_train, batch_size=32)

# 모델 훈련
history = model.fit(
    train_generator,
    steps_per_epoch=len(x_train) // 32,
    epochs=100,
    validation_data=(x_test, y_test),
    callbacks=callbacks,
    verbose=1
)

# 성능 평가
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"\n✅ 최종 테스트 정확도: {test_acc * 100:.2f}%")

# 학습 과정 시각화
plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Test')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Test')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()

# 혼동 행렬
y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

cm = confusion_matrix(y_true_classes, y_pred_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_map.keys(),
            yticklabels=class_map.keys())
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
