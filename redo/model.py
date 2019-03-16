"""
The main image colorization model.
"""

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import layers
tf.logging.set_verbosity(tf.logging.ERROR)

from constants import *

class ImageColorization():

    def __init__(self):
        inputs = layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 1))

        ll1 = layers.Conv2D(DEPTH, CONV_KERNEL_SIZE, strides=(2, 2), padding='same', activation='relu')(inputs)
        ll2 = layers.Conv2D(2 * DEPTH, CONV_KERNEL_SIZE, strides=(1, 1), padding='same', activation='relu')(ll1)
        ll3 = layers.Conv2D(2 * DEPTH, CONV_KERNEL_SIZE, strides=(2, 2), padding='same', activation='relu')(ll2)
        ll4 = layers.Conv2D(4 * DEPTH, CONV_KERNEL_SIZE, strides=(1, 1), padding='same', activation='relu')(ll3)

        gl1 = layers.Conv2D(4 * DEPTH, CONV_KERNEL_SIZE, strides=(2, 2), padding='same', activation='relu')(ll4)
        gl2 = layers.Conv2D(4 * DEPTH, CONV_KERNEL_SIZE, strides=(1, 1), padding='same', activation='relu')(gl1)
        gl3 = layers.Flatten()(gl2)
        gl4 = layers.Dense(512, activation='relu')(gl3)
        gl4 = layers.Dropout(0.5)(gl4)
        gl5 = layers.Dense(256, activation='relu')(gl4)
        gl5 = layers.Dropout(0.5)(gl5)
        gl6 = layers.RepeatVector(8*8)(gl5)
        gl7 = layers.Reshape([IMAGE_SIZE/4, IMAGE_SIZE/4 ,256])(gl6)

        ml1 = layers.Conv2D(4 * DEPTH, CONV_KERNEL_SIZE, strides=(1, 1), padding='same', activation='relu')(ll4)

        fusionl = layers.Concatenate(axis=3)([gl7, ml1])

        cl1 = layers.Conv2D(2 * DEPTH, CONV_KERNEL_SIZE, strides=(1, 1), padding='same', activation='relu')(fusionl)
        cl2 = layers.UpSampling2D(size=(2, 2), interpolation='nearest')(cl1)
        cl3 = layers.Conv2D(DEPTH/2, CONV_KERNEL_SIZE, strides=(1, 1), padding='same', activation='relu')(cl2)
        cl4 = layers.Conv2D(2, CONV_KERNEL_SIZE, strides=(1, 1), padding='same', activation='sigmoid')(cl3)
        cl5 = layers.UpSampling2D(size=(2, 2), interpolation='nearest', name='colorization_output')(cl4)

        classificationl = layers.Dense(NUM_CLASSES, name='classification_output', activation='softmax')(gl5)

        self.model = Model(inputs = inputs, outputs = [cl5, classificationl])
