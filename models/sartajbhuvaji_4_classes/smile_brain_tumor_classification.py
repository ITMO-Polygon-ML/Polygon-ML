import glob
import os.path
from typing import Tuple

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import compute_class_weight

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, GlobalAveragePooling2D, Flatten, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNet
from sklearn.metrics import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix
import keras


class NNExample:
    def __init__(self, image_size, batch_size,
                 epoch=10, weight_path=None, **kwargs):
        self.epoch: int = epoch
        self.batch_size: int = batch_size
        self.image_size: Tuple[int] = image_size

        self.weights_path: str = weight_path

        self.color_mode = kwargs.get('color_mode', 'grayscale')
        self.history = None

        self.train_data_gen = ImageDataGenerator(
            rescale=1. / 255,
            rotation_range=10,
            zoom_range=0.1,
            shear_range=0.1,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True)
        self.val_data_gen = ImageDataGenerator(rescale=1. / 255)

    def construct_model(self):
        model = Sequential([
            Conv2D(filters=32, kernel_size=(5, 5), padding='same',
                   activation='relu', input_shape=(self.image_size[0], self.image_size[1], 1)),
            MaxPool2D(pool_size=(2, 2)),

            Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'),
            MaxPool2D(pool_size=(2, 2)),
            Dropout(0.2),

            Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu'),
            MaxPool2D(pool_size=(2, 2)),
            Dropout(0.2),

            Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu'),
            MaxPool2D(pool_size=(2, 2)),
            Dropout(0.2),

            Flatten(),
            Dense(1024, activation='relu'),
            Dropout(0.4),
            Dense(4, activation='softmax')
        ])

        model.compile(optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        return model

    def create_data_iterator(self, generator, folder_path):
        sub_folders = os.listdir(folder_path)
        iterator = generator.flow_from_directory(
            folder_path,
            target_size=self.image_size,
            color_mode=self.color_mode,
            class_mode='binary' if len(sub_folders) == 2 else 'categorical',
            batch_size=self.batch_size,
            shuffle=True
        )
        return iterator

    def get_callbacks(self) -> list:
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

        early_stop = EarlyStopping(monitor='val_loss', mode='min', patience=3)
        learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss',
                                                    patience=2, verbose=1, factor=0.3, min_lr=0.00001)

        return [early_stop, learning_rate_reduction]

    def fit(self, train_folder, val_folder=None):
        # Data preparation and increase the amount of data set
        gen_train = self.create_data_iterator(self.train_data_gen, train_folder)
        gen_val = None
        if val_folder:
            gen_val = self.create_data_iterator(self.val_data_gen, val_folder)

        model = self.construct_model()

        weights = compute_class_weight(class_weight='balanced', classes=np.unique(gen_train.classes),
                                       y=gen_train.classes)

        class_weight = dict(zip(np.unique(gen_train.classes), weights))

        callbacks_list = self.get_callbacks()
        try:
            steps_per_epoch = len(gen_train.filepaths) // self.batch_size
            validation_steps = len(gen_val.filepaths) // self.batch_size
            self.history = model.fit(
                gen_train,
                validation_data=gen_val,
                epochs=self.epoch,
                class_weight=class_weight,
                steps_per_epoch=steps_per_epoch,
                validation_steps=validation_steps,
                callbacks=callbacks_list
            )
        except Exception as e:
            raise Exception(f'Fitting error: {e}')

        if self.weights_path is None:
            self.weights_path = f'weights_{hash(gen_train)}_{self.epoch}.h5'

        model.save(self.weights_path)

        return self

    def evaluate(self, test_folder, predict_column: str = 'predict', proba_column='proba'):
        if self.weights_path is None or not os.path.exists(self.weights_path):
            raise Exception(f'Weights were not found by path: {self.weights_path}')

        model = keras.models.load_model(self.weights_path)

        iterator = self.val_data_gen.flow_from_directory(
            test_folder,
            target_size=self.image_size,
            color_mode=self.color_mode,
            class_mode=None,
            batch_size=self.batch_size,
            shuffle=False
        )

        N = len(iterator.classes)
        proba = model.predict(iterator, batch_size=self.batch_size, steps=N // self.batch_size + 1)

        if proba.shape[0] > N:
            # in case of odd number N / self.batch_size
            proba = proba[:N]

        if proba.shape[1] == 1:
            proba_ext = np.concatenate((1 - proba, proba), axis=1)
        else:
            proba_ext = proba

        class_map = dict(map(lambda v: (v[1], v[0]), iterator.class_indices.items()))
        predict = list(map(lambda v: class_map[v], np.argmax(proba_ext, axis=1)))

        # Считается обратная вероятность (необходима для некоторых методов оценки)
        proba = np.vstack((1 - proba, proba)).T

        return {predict_column: predict, proba_column: proba}
