import os.path
from typing import Tuple

import numpy as np
from sklearn.utils.class_weight import compute_class_weight

import keras
from keras import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg19 import VGG19
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense


class NNExample:
    def __init__(self, image_size, batch_size,
                 epoch=2, weight_path=None, **kwargs):

        self.epoch: int = epoch
        self.batch_size: int = batch_size
        self.image_size: Tuple[int] = image_size

        self.weights_path: str = weight_path

        self.color_mode = kwargs.get('color_mode', 'rgb')
        self.history = None

        self.train_data_gen = ImageDataGenerator(
            rescale=1. / 255,
            horizontal_flip=2, 
            vertical_flip=2)
        self.val_data_gen = ImageDataGenerator(rescale=1. / 255)

    def construct_model(self):
        model = VGG19(
          input_shape = (self.image_size[0], self.image_size[1], 3),
          include_top = False,
          weights = 'imagenet'
        )

        for layers in model.layers:
          layers.trainable = False

        x = Flatten()(model.output)
        x = Dropout(0.4)(x)
        x = Dense(1, activation = "sigmoid")(x)

        model = keras.Model(model.input, x)
        model.compile(loss = "binary_crossentropy", optimizer = "adam", metrics = "accuracy")
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
