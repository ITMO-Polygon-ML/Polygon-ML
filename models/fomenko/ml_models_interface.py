import os.path
from typing import Tuple

import numpy as np
from sklearn.utils.class_weight import compute_class_weight

import keras
from keras import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras import layers, models
from keras import backend as K
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
        
        """
        В найденной модели генератор задавался таким образом, будет как альтернатива
        self.train_data_gen = ImageDataGenerator(rotation_range=0.2,
                            width_shift_range=0.05,
                            height_shift_range=0.05,
                            shear_range=0.05,
                            zoom_range=0.05,
                            horizontal_flip=True,
                            fill_mode='nearest',
                            rescale=1./255)
        """


    def unet(input_size=(256,256,3)):
        inputs = layers.Input(input_size)
        
        conv1 = layers.Conv2D(64, (3, 3), padding='same')(inputs)
        bn1 = layers.Activation('relu')(conv1)
        conv1 = layers.Conv2D(64, (3, 3), padding='same')(bn1)
        bn1 = layers.BatchNormalization(axis=3)(conv1)
        bn1 = layers.Activation('relu')(bn1)
        pool1 = layers.MaxPooling2D(pool_size=(2, 2))(bn1)

        conv2 = layers.Conv2D(128, (3, 3), padding='same')(pool1)
        bn2 = layers.Activation('relu')(conv2)
        conv2 = layers.Conv2D(128, (3, 3), padding='same')(bn2)
        bn2 = layers.BatchNormalization(axis=3)(conv2)
        bn2 = layers.Activation('relu')(bn2)
        pool2 = layers.MaxPooling2D(pool_size=(2, 2))(bn2)

        conv3 = layers.Conv2D(256, (3, 3), padding='same')(pool2)
        bn3 = layers.Activation('relu')(conv3)
        conv3 = layers.Conv2D(256, (3, 3), padding='same')(bn3)
        bn3 = layers.BatchNormalization(axis=3)(conv3)
        bn3 = layers.Activation('relu')(bn3)
        pool3 = layers.MaxPooling2D(pool_size=(2, 2))(bn3)

        conv4 = layers.Conv2D(512, (3, 3), padding='same')(pool3)
        bn4 = layers.Activation('relu')(conv4)
        conv4 = layers.Conv2D(512, (3, 3), padding='same')(bn4)
        bn4 = layers.BatchNormalization(axis=3)(conv4)
        bn4 = layers.Activation('relu')(bn4)
        pool4 = layers.MaxPooling2D(pool_size=(2, 2))(bn4)

        conv5 = layers.Conv2D(1024, (3, 3), padding='same')(pool4)
        bn5 = layers.Activation('relu')(conv5)
        conv5 = layers.Conv2D(1024, (3, 3), padding='same')(bn5)
        bn5 = layers.BatchNormalization(axis=3)(conv5)
        bn5 = layers.Activation('relu')(bn5)

        up6 = layers.concatenate([layers.Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(bn5), conv4], axis=3)
        conv6 = layers.Conv2D(512, (3, 3), padding='same')(up6)
        bn6 = layers.Activation('relu')(conv6)
        conv6 = layers.Conv2D(512, (3, 3), padding='same')(bn6)
        bn6 = layers.BatchNormalization(axis=3)(conv6)
        bn6 = layers.Activation('relu')(bn6)

        up7 = layers.concatenate([layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(bn6), conv3], axis=3)
        conv7 = layers.Conv2D(256, (3, 3), padding='same')(up7)
        bn7 = layers.Activation('relu')(conv7)
        conv7 = layers.Conv2D(256, (3, 3), padding='same')(bn7)
        bn7 = layers.BatchNormalization(axis=3)(conv7)
        bn7 = layers.Activation('relu')(bn7)

        up8 = layers.concatenate([layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(bn7), conv2], axis=3)
        conv8 = layers.Conv2D(128, (3, 3), padding='same')(up8)
        bn8 = layers.Activation('relu')(conv8)
        conv8 = layers.Conv2D(128, (3, 3), padding='same')(bn8)
        bn8 = layers.BatchNormalization(axis=3)(conv8)
        bn8 = layers.Activation('relu')(bn8)

        up9 = layers.concatenate([layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(bn8), conv1], axis=3)
        conv9 = layers.Conv2D(64, (3, 3), padding='same')(up9)
        bn9 = layers.Activation('relu')(conv9)
        conv9 = layers.Conv2D(64, (3, 3), padding='same')(bn9)
        bn9 = layers.BatchNormalization(axis=3)(conv9)
        bn9 = layers.Activation('relu')(bn9)

        conv10 = layers.Conv2D(1, (1, 1), activation='sigmoid')(bn9)

        return models.Model(inputs=[inputs], outputs=[conv10])

    def construct_model(self):

        smooth=1

        def dice_coef(y_true, y_pred):
            y_true = K.flatten(y_true)
            y_pred = K.flatten(y_pred)
            intersection = K.sum(y_true * y_pred)
            union = K.sum(y_true) + K.sum(y_pred)
            return (2.0 * intersection + smooth) / (union + smooth)

        def dice_coef_loss(y_true, y_pred):
            return 1 - dice_coef(y_true, y_pred)

        def bce_dice_loss(y_true, y_pred):
            bce = keras.losses.BinaryCrossentropy(from_logits=True)
            return dice_coef_loss(y_true, y_pred) + bce(y_true, y_pred)

        model = unet(input_size=(256, 256, 3))

        model.compile(
            optimizer='adam',
            loss=bce_dice_loss,
            metrics=[dice_coef,'accuracy'])

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
