import os
import cv2
import numpy as np
# import tensorflow as tf
from tensorflow.keras import layers, models, utils, optimizers
import matplotlib.pyplot as plt


train_dir = r'C:\Users\USER\.cache\kagglehub\datasets\volkandl\car-brand-logos\versions\1\Car_Brand_Logos\Train'
test_dir = r'C:\Users\USER\.cache\kagglehub\datasets\volkandl\car-brand-logos\versions\1\Car_Brand_Logos\Test'

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
                    img = cv2.resize(img, (224, 224))
                    img = img / 255.0
                    images.append(img)
                    labels.append(index)
                except Exception as e:
                    print(f"⚠️ 손상된 파일 건너뛰기: {img_path}")
    return np.array(images), np.array(labels)

class_map = {brand: idx for idx, brand in enumerate(os.listdir(train_dir))}
num_classes = len(class_map)

x_train, y_train = load_images(train_dir, class_map)
x_test, y_test = load_images(test_dir, class_map)

x_train = x_train.reshape(x_train.shape[0], 224, 224, 3)
x_test = x_test.reshape(x_test.shape[0], 224, 224, 3)

x_train = x_train / 255.0
x_test = x_test / 255.0

y_train = utils.to_categorical(y_train, num_classes)
y_test = utils.to_categorical(y_test, num_classes)

n_input = 224 * 224 * 3  # 150,528
n_output = num_classes

# 모델 생성
mlp = models.Sequential()
mlp.add(layers.Dense(1024, activation='relu', input_shape=(n_input,)))
mlp.add(layers.Dense(512, activation='relu'))
mlp.add(layers.Dense(256, activation='relu'))
mlp.add(layers.Dense(128, activation='relu'))
mlp.add(layers.Dense(64, activation='relu'))
mlp.add(layers.Dense(n_output, activation='softmax'))

mlp.compile(loss='categorical_crossentropy',
            optimizer=optimizers.Adam(0.001),
            metrics=['accuracy'])

# 데이터를 1D로 펼쳐서 입력
x_train_flat = x_train.reshape(-1, n_input)
x_test_flat = x_test.reshape(-1, n_input)

hist = mlp.fit(x_train_flat, y_train,
               batch_size=128,
               epochs=30,
               validation_data=(x_test_flat, y_test),
               verbose=2)

res = mlp.evaluate(x_test_flat, y_test, verbose=0)
print(f"Accuracy: {res[1]*100:.2f}%")






# plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_accuracy'])
# plt.title('Model Accuracy')
# plt.ylabel('Accuracy')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Validation'], loc = 'Lower right')
# plt.grid()
# plt.show()

# plt.plot(history.history['Loss'])
# plt.plot(history.history['val_Loss'])
# plt.title('Model Loss')
# plt.ylabel('Loss')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Validation'], loc = 'Upper right')
# plt.grid()
# plt.show()