import os
import cv2
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam

# 이미지 받아오기 -> 이미지 전처리 -> 전처리 된 이미지 변수에 넣기(x_train, y_train), (x_test, y_test)
# 이미지 평탄화( 1개의 배열로 ) -> 모델 만들기 -> 모델 컴파일

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
                    img = cv2.resize(img, (128, 128))  # 이미지 크키가 커지면 학습률도 올라감 64보단 낳아짐
                    # img = cv2.resize(img, (64, 64))
                    img = img / 255.0  # 사전 정규화
                    images.append(img)
                    labels.append(index)
                except Exception as e:
                    print(f"⚠️ 손상된 파일 건너뛰기: {img_path}")
    return np.array(images), np.array(labels)

class_map = {brand: idx for idx, brand in enumerate(os.listdir(train_dir))}

num_classes = len(class_map)  # 클레스 길이 확인

# 각 학습, 테스트 데이터를 변수에 입력, 동시에 정규화 수행
x_train, y_train = load_images(train_dir, class_map)
x_test, y_test = load_images(test_dir, class_map)

# shape[0] 이미지의 개수 / 244, 244은 세로, 가로 픽셀 수 / 3은 rgb 컬러 수  // 64 64로 변경
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], x_train.shape[3])
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], x_test.shape[3])

sample_num = x_train.shape[1] * x_test.shape[2] * x_test.shape[3]

# 정규화
# x_train = x_train/255.0
# x_test = x_test/255.0

# # 이미지 평탄화 3D -> 1D
# x_train = x_train.reshape(x_train.shape[0], -1)  # (샘플수, 224*224*3)
# x_test = x_test.reshape(x_test.shape[0], -1)
x_train = x_train.reshape(-1, sample_num)  # (샘플수, 224*224*3)
x_test = x_test.reshape(-1, sample_num)

# 원핫 인코딩 각 레이블을 0와 1로만 이루어진 벡터로 변환 / 각 픽셀을 0~8까지의 값으로 변환(클레스가 8개이기 때문)
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

# y_train = to_categorical(y_train, num_classes)
# y_test = to_categorical(y_test, num_classes)

## 여기까지는 문제 없음

# 모델 만들기
model = Sequential([
    # Dense(512, input_shape=(224 * 224 * 3,)),  # 활성화 함수 분리
    Dense(512, input_shape=(sample_num,)),  # 활성화 함수 분리
    # Dense(512, input_shape=(64 * 64 * 3,)),  # 활성화 함수 분리
    # tf.keras.layers.Activation('relu'),  # 명시적 활성화 층
    # BatchNormalization(),
    # Dense(512, input_shape=(64 * 64 * 3,), activation='relu'),  # 활성화 함수 분리
    BatchNormalization(),
    tf.keras.layers.Activation('relu'),  # 명시적 활성화 층
    Dropout(0.3),

    Dense(256),
    # Dense(256, input_shape=(128 * 128 * 3,)),
    # tf.keras.layers.Activation('relu'),
    BatchNormalization(),
    tf.keras.layers.Activation('relu'),
    Dropout(0.25),

    Dense(128, activation='relu'),
    Dropout(0.2),

    # Dense(num_classes, activation='sigmoid')  # 테스트 정확도가 softmax에 비해 조금 떨어짐
    Dense(num_classes, activation='softmax')
])

# gkrtmq tmzpwnffld 학습 스케줄링
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.001,
    # decay_steps=1000,
    # decay_steps=len(x_train) // 224 * 5,  # 5 에포크 마다 감소
    decay_steps=len(x_train) // 128 * 5,  # 5 에포크 마다 감소
    # decay_steps = len(x_train) // 64 * 5, # 5 에포크 마다 감소
    decay_rate=0.9)

model.compile(
    # loss = 'mse',
    loss='categorical_crossentropy', # mse보다 성능이 좋아짐
    optimizer= Adam(lr_schedule),
    metrics=['accuracy']
)

hist = model.fit(
    x_train, y_train,
    epochs=50,
    batch_size=64,
    validation_split=0.2,
    verbose=2
)

# test_loss, test_acc = model.evaluate(x_test, y_test)
# print(f"\n📊 테스트 정확도: {test_acc:.4f}")
res = model.evaluate(x_test, y_test)
print(f"Accuracy is", res[1]*100)


# plt.plot(hist.history['accuracy'])
# plt.plot(hist.history['val_accuracy'])
# plt.title('Model Accuracy')
# plt.ylabel('Accuracy')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Validation'], loc = 'Lower right')
# plt.grid()
# plt.show()
#
# plt.plot(history.history['Loss'])
# plt.plot(history.history['val_Loss'])
# plt.title('Model Loss')
# plt.ylabel('Loss')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Validation'], loc = 'Upper right')
# plt.grid()
# plt.show()