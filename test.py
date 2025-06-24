# import os
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras.preprocessing.image import load_img, img_to_array
# from tensorflow.keras.applications import ResNet50, VGG19, VGG16, ResNet101V2
# from tensorflow.keras.applications.resnet50 import preprocess_input
# from tensorflow.keras.models import Model
# from tensorflow.keras.utils import to_categorical
# from sklearn.model_selection import train_test_split
# from tensorflow.keras.models import Sequential, load_model
# from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, accuracy_score
# from tensorflow.keras.layers import Conv2D, Dropout, Dense, Flatten, MaxPooling2D, GlobalAveragePooling2D, Activation, BatchNormalization, MaxPooling2D
# from tensorflow.keras.optimizers import Adam
# from keras.callbacks import EarlyStopping, ModelCheckpoint
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.regularizers import l2
#
# def load_images(folder, class_map):
#     images = []
#     labels = []
#     for brand, index in class_map.items():
#         brand_path = os.path.join(folder, brand)
#         if os.path.isdir(brand_path):
#             for img_file in os.listdir(brand_path):
#                 img_path = os.path.join(brand_path, img_file)
#                 img = cv2.imread(img_path)
#                 img = cv2.resize(img, (224, 224))
#                 img = img / 255.0
#                 images.append(img)
#                 labels.append(index)
#     return np.array(images), np.array(labels)
#
#
# (x_train, y_train), (x_test, y_test) =