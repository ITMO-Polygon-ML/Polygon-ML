import os.path
from typing import Tuple

import numpy as np
from sklearn.utils.class_weight import compute_class_weight

import keras
from keras import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense


class NNExample:
    def __init__(self, image_size, batch_size,
                 epoch=2, weight_path=None, **kwargs):

        self.epoch: int = epoch
        self.batch_size: int = batch_size
        self.image_size: Tuple[int] = image_size

        self.weights_path: str = weight_path

        self.color_mode = kwargs.get('color_mode', 'grayscale')
        self.history = None

        self.train_data_gen = ImageDataGenerator(
            rescale=1. / 255,
            shear_range=0.2,
            width_shift_range=0.1,
            height_shift_range=0.1,
            brightness_range=(0.8, 1.2),
            zoom_range=0.2,
            horizontal_flip=True
        )
        self.val_data_gen = ImageDataGenerator(rescale=1. / 255)

    def construct_model(self):
        model = Sequential()

        model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(self.image_size[0], self.image_size[1], 1)))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(32, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))

        model.add(Conv2D(32, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))

        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))

        model.add(Conv2D(128, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.5))

        model.add(Flatten())

        model.add(Dense(128, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))

        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

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
            proba = np.vstack((1 - proba, proba))

        class_map = dict(map(lambda v: (v[1], v[0]), iterator.class_indices.items()))
        predict = np.apply_along_axis(lambda v: class_map[v[0]], 1, np.argmax(proba, axis=1))

        # Считается обратная вероятность (необходима для некоторых методов оценки)
        proba = np.vstack((1 - predict, predict)).T

        return {predict_column: predict, proba_column: proba}
