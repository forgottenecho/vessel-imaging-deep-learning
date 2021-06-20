# my implementation of Ronneberger's U-Net
# for expansion half of network, used resizing instead of cropping before concatenation
# loss function is pure binary_crossentropy, have not included pixel weight map yet
# used adam optimizer instead of pure sgd

# number of classes

import tensorflow as tf
from tensorflow.keras import Model, Input, layers, initializers
import numpy as np

class Unet(Model):
    def __init__(self, input_shape, K, number_of_filters=64, levels=4, activation='relu', kernel_initializer=initializers.HeNormal):
        self.K = K
        input = Input(shape = input_shape)
        
        
        counterparts = []
        previous_down = input
        
        for i in range(levels):
            down, counterpart = contracting_block(previous_down, number_of_filters)
            counterparts.insert(0, counterpart)

            number_of_filters *= 2
            previous_down = down

        previous_smashed_together = previous_down

        for i in range(levels):
            smashed_together = expanding_block(previous_smashed_together, counterparts[i], number_of_filters)

            number_of_filters /= 2
            previous_smashed_together = smashed_together

        final_concat = previous_smashed_together
        conv = layers.Conv2D(number_of_filters, 3, activation='relu', kernel_initializer=initializers.HeNormal)(final_concat)
        conv = layers.Conv2D(number_of_filters, 3, activation='relu', kernel_initializer=initializers.HeNormal)(conv)
        output = layers.Conv2D(K, 1, activation='softmax', kernel_initializer=initializers.HeNormal, name='maps')(conv)

        super(input, output)

    def contracting_block(previous_down, number_of_filters, kernel_size=3, activation='relu', kernel_initializer=initializers.HeNormal):
        right_convolution = layers.Conv2D(number_of_filters, kernel_size, activation=activation, kernel_initializer=kernel_initializer)(previous_down)
        right_convolution = layers.Conv2D(number_of_filters, kernel_size, activation=activation, kernel_initializer=kernel_initializer)(right_convolution)
        down = layers.MaxPool2D()(right_convolution)

        return down, right_convolution

    def _expanding_block(previous_smashed_together, contracting_counterpart, number_of_filters, kernel_size=3, activation='relu', kernel_initializer=initializers.HeNormal):
        right_convolution = layers.Conv2D(number_of_filters, kernel_size, activation=activation, kernel_initializer=kernel_initializer)(previous_smashed_together)
        right_convolution = layers.Conv2D(number_of_filters, kernel_size, activation=activation, kernel_initializer=kernel_initializer)(right_convolution)
        up = layers.Conv2DTranspose(number_of_filters/2, 2, strides=2, activation=activation, kernel_initializer=kernel_initializer)(right_convolution)
        smashed_together = layers.Concatenate()([tf.image.resize(contracting_counterpart, up.shape[1:3]), up])

        return smashed_together



model.summary()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy','val_accuracy'])


#model.predict(np.empty((3,572,572,1)))

shape=(572, 572, 1)