import os
import cv2
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

# 데이터 경로
train_dir = r'C:\Users\USER\.cache\kagglehub\datasets\volkandl\car-brand-logos\versions\1\Car_Brand_Logos\Train'
test_dir = r'C:\Users\USER\.cache\kagglehub\datasets\volkandl\car-brand-logos\versions\1\Car_Brand_Logos\Test'


# 간소화된 데이터 로딩 함수
def load_simple(folder):
    images = []
    labels = []
    class_names = sorted(os.listdir(folder))
    for class_idx, class_name in enumerate(class_names):
        class_path = os.path.join(folder, class_name)
        for img_file in os.listdir(class_path):
            img = cv2.imread(os.path.join(class_path, img_file))
            if img is not None:
                img = cv2.resize(img, (64, 64))  # 크기 축소
                images.append(img / 255.0)  # 정규화
                labels.append(class_idx)
    return np.array(images), np.array(labels)


# 데이터 로드
x_train, y_train = load_simple(train_dir)
x_test, y_test = load_simple(test_dir)

# CNN 모델 (간소화 버전)
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(len(np.unique(y_train)), activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',  # 원-핫 인코딩 제거
              metrics=['accuracy'])

# 훈련 (검증 분리)
history = model.fit(x_train, y_train,
                    epochs=30,
                    validation_split=0.2,
                    batch_size=32)

# 성능 평가
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"\n✅ 테스트 정확도: {test_acc * 100:.2f}%")

# 정확도 그래프
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.show()
