import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications import ResNet50, VGG19, VGG16, ResNet101V2
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, accuracy_score
from tensorflow.keras.layers import Conv2D, Dropout, Dense, Flatten, MaxPooling2D, GlobalAveragePooling2D, Activation, BatchNormalization, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2



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
                img = cv2.imread(img_path)
                img = cv2.resize(img, (224, 224))
                img = img / 255.0
                images.append(img)
                labels.append(index)
    return np.array(images), np.array(labels)

class_map = {brand: idx for idx, brand in enumerate(os.listdir(train_dir))}

x_train, y_train = load_images(train_dir, class_map)
x_test, y_test = load_images(test_dir, class_map)

y_train = to_categorical(y_train, 8)
y_test = to_categorical(y_test, 8)

train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.95,
    zoom_range=0.95,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset="training")

validation_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset="validation")


base_model = ResNet101V2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

for layer in base_model.layers[:-50]:
    layer.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu', kernel_regularizer=l2(0.01))(x)
x = BatchNormalization()(x)
x = Dropout(0.6)(x)
out = Dense(8, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=out)

model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_accuracy', min_delta=0.001, patience=15, restore_best_weights=True)
checkpoint = ModelCheckpoint("best_model_cnn.keras", monitor='val_accuracy', save_best_only=True, verbose=1)
history = model.fit(
        train_generator,
        epochs=5,
        validation_data=validation_generator,
        batch_size = 64,
        callbacks = [early_stopping, checkpoint])


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.ylim(0, 1)
plt.legend(['train acc', 'valid acc', 'train loss', 'valid loss'], loc = 'lower right')
plt.show()