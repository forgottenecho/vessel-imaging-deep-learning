# my implementation of Ronneberger's U-Net
# for expansion half of network, used resizing instead of cropping before concatenation
# loss function is pure binary_crossentropy, have not included pixel weight map yet
# used adam optimizer instead of pure sgd

# number of classes

import tensorflow as tf
from tensorflow.keras import Model, Input, layers, initializers
import numpy as np

def contracting_block(previous_down, number_of_filters, kernel_size=3, activation='relu', kernel_initializer=initializers.HeNormal):
    right_convolution = layers.Conv2D(number_of_filters, kernel_size, activation=activation, kernel_initializer=kernel_initializer)(previous_down)
    right_convolution = layers.Conv2D(number_of_filters, kernel_size, activation=activation, kernel_initializer=kernel_initializer)(right_convolution)
    down = layers.MaxPool2D()(right_convolution)

    return down, right_convolution

def expanding_block(previous_smashed_together, contracting_counterpart, number_of_filters, kernel_size=3, activation='relu', kernel_initializer=initializers.HeNormal):
    right_convolution = layers.Conv2D(number_of_filters, kernel_size, activation=activation, kernel_initializer=kernel_initializer)(previous_smashed_together)
    right_convolution = layers.Conv2D(number_of_filters, kernel_size, activation=activation, kernel_initializer=kernel_initializer)(right_convolution)
    up = layers.Conv2DTranspose(number_of_filters/2, 2, strides=2, activation=activation, kernel_initializer=kernel_initializer)(right_convolution)
    smashed_together = layers.Concatenate()([tf.image.resize(contracting_counterpart, up.shape[1:3]), up])

    return smashed_together

K = 3


input = Input(shape=(572, 572, 1))

number_of_filters = 64
counterparts = []
previous_down = input
i = 0
while i < 4:
    down, counterpart = contracting_block(previous_down, number_of_filters)
    counterparts.insert(0, counterpart)

    number_of_filters *= 2
    previous_down = down
    i += 1

previous_smashed_together = previous_down
i = 0
while i < 4:
    smashed_together = expanding_block(previous_smashed_together, counterparts[i], number_of_filters)

    number_of_filters /= 2
    previous_smashed_together = smashed_together
    i += 1



# right1 = layers.Conv2D(64, 3, activation='relu', kernel_initializer=initializers.HeNormal)(right1)
# right1 = layers.Conv2D(64, 3, activation='relu', kernel_initializer=initializers.HeNormal)(right1)

# output = layers.Conv2D(K, 1, activation='softmax', kernel_initializer=initializers.HeNormal, name='maps')(right1)



model = Model(input, previous_smashed_together)
model.summary()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy','val_accuracy'])


#model.predict(np.empty((3,572,572,1)))