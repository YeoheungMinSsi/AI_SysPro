import os
import cv2
import numpy as np
import tensorflow as tf

mlp = tf.keras.models.Sequential()
Dense = tf.keras.layers.Dense
Adam = tf.keras.optimizers.Adam

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
                    img = cv2.resize(img, (224, 224))
                    img = img / 255.0
                    images.append(img)
                    labels.append(index)
                except Exception as e:
                    print(f"⚠️ 손상된 파일 건너뛰기: {img_path}")
    return np.array(images), np.array(labels)


class_map = {brand: idx for idx, brand in enumerate(os.listdir(train_dir))}

# 각 학습, 테스트 데이터를 변수에 입력
x_train, y_train = load_images(train_dir, class_map)
x_test, y_test = load_images(test_dir, class_map)

# 60000은 이미지의 개수 / 244, 244은 세로, 가로 픽셀 수 / 3은 rgb 컬러 수
x_train = x_train.reshape(x_train.shape[0], 224, 224, 3)
x_test = x_test.reshape(x_test.shape[0], 224, 224, 3)

# 정규화
x_train = x_train/255.0
x_test = x_test/255.0

# 원핫 인코딩 각 레이블을 0와 1로만 이루어진 벡터로 변환 / 각 픽셀을 0~10까지의 값으로 변환
y_train = tf.keras.utils.to_categorical(y_train, 8)
y_test = tf.keras.utils.to_categorical(y_test, 8)

n_input = 255 * 255 * 3
n_output = 8


mlp.add(Dense(units = 1024, activation = "relu",
              input_shape = (n_input,), kernel_initializer = 'random_uniform', bias_initializer = 'random_uniform'))
mlp.add(Dense(units = 512, activation = "relu", kernel_initializer='random_uniform', bias_initializer='zeros'))
mlp.add(Dense(units = 256, activation = "relu", kernel_initializer='random_uniform', bias_initializer='zeros'))
mlp.add(Dense(units = 128, activation = "relu", kernel_initializer='random_uniform', bias_initializer='zeros'))
mlp.add(Dense(units = 65, activation = "relu", kernel_initializer='random_uniform', bias_initializer='zeros'))
mlp.add(Dense(units = n_output, activation = "relu", kernel_initializer='random_uniform', bias_initializer='zeros'))

mlp.compile(loss = 'mse' , optimizer = Adam(learning_rate = 0.001), metrics = ['accuracy'])
hist = mlp.fit(x_train, y_train, batch_size = 128, epochs = 30, validation_data = (x_test, y_test), verbose = 2)

res = mlp.evaluate(x_test, y_test, verbose = 0)
print("Accuracy : ", res[1]*100)
