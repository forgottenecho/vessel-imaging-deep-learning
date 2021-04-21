# my implementation of Ronneberger's U-Net
# for expansion half of network, used resizing instead of cropping before concatenation
# loss function is pure binary_crossentropy, have not included pixel weight map yet
# used adam optimizer instead of pure sgd

# number of classes

import tensorflow as tf
from tensorflow.keras import Model, Input, layers, initializers
import numpy as np

K = 3


input = Input(shape=(572, 572, 1))
left1 = layers.Conv2D(64, 3, activation='relu', kernel_initializer=initializers.HeNormal)(input)
left1 = layers.Conv2D(64, 3, activation='relu', kernel_initializer=initializers.HeNormal)(left1)
down1 = layers.MaxPool2D()(left1)

left2 = layers.Conv2D(128, 3, activation='relu', kernel_initializer=initializers.HeNormal)(down1)
left2 = layers.Conv2D(128, 3, activation='relu', kernel_initializer=initializers.HeNormal)(left2)
down2 = layers.MaxPool2D()(left2)

left3 = layers.Conv2D(256, 3, activation='relu', kernel_initializer=initializers.HeNormal)(down2)
left3 = layers.Conv2D(256, 3, activation='relu', kernel_initializer=initializers.HeNormal)(left3)
down3 = layers.MaxPool2D()(left3)

left4 = layers.Conv2D(512, 3, activation='relu', kernel_initializer=initializers.HeNormal)(down3)
left4 = layers.Conv2D(512, 3, activation='relu', kernel_initializer=initializers.HeNormal)(left4)
down4 = layers.MaxPool2D()(left4)

left5 = layers.Conv2D(1024, 3, activation='relu', kernel_initializer=initializers.HeNormal)(down4)
left5 = layers.Conv2D(1024, 3, activation='relu', kernel_initializer=initializers.HeNormal)(left5)
up5 = layers.Conv2DTranspose(512, 2, strides=2, activation='relu', kernel_initializer=initializers.HeNormal)(left5)

right4 = layers.Concatenate()([tf.image.resize(left4, up5.shape[1:3]), up5])
right4 = layers.Conv2D(512, 3, activation='relu', kernel_initializer=initializers.HeNormal)(right4)
right4 = layers.Conv2D(512, 3, activation='relu', kernel_initializer=initializers.HeNormal)(right4)
up4 = layers.Conv2DTranspose(256, 2, strides=2, activation='relu', kernel_initializer=initializers.HeNormal)(right4)

right3 = layers.Concatenate()([tf.image.resize(left3, up4.shape[1:3]), up4])
right3 = layers.Conv2D(256, 3, activation='relu', kernel_initializer=initializers.HeNormal)(right3)
right3 = layers.Conv2D(256, 3, activation='relu', kernel_initializer=initializers.HeNormal)(right3)
up3 = layers.Conv2DTranspose(128, 2, strides=2, activation='relu', kernel_initializer=initializers.HeNormal)(right3)

right2 = layers.Concatenate()([tf.image.resize(left2, up3.shape[1:3]), up3])
right2 = layers.Conv2D(128, 3, activation='relu', kernel_initializer=initializers.HeNormal)(right2)
right2 = layers.Conv2D(128, 3, activation='relu', kernel_initializer=initializers.HeNormal)(right2)
up2 = layers.Conv2DTranspose(64, 2, strides=2, activation='relu', kernel_initializer=initializers.HeNormal)(right2)

right1 = layers.Concatenate()([tf.image.resize(left1, up2.shape[1:3]), up2])
right1 = layers.Conv2D(64, 3, activation='relu', kernel_initializer=initializers.HeNormal)(right1)
right1 = layers.Conv2D(64, 3, activation='relu', kernel_initializer=initializers.HeNormal)(right1)

output = layers.Conv2D(K, 1, activation='softmax', kernel_initializer=initializers.HeNormal, name='maps')(right1)



model = Model(input, output)
model.summary()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy','val_accuracy'])


#model.predict(np.empty((3,572,572,1)))