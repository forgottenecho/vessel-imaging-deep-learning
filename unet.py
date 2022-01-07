# my implementation of Ronneberger's U-Net
# for expansion half of network, used resizing instead of cropping before concatenation
# loss function is pure binary_crossentropy, have not included pixel weight map yet
# used adam optimizer instead of pure sgd

# number of classes

import tensorflow as tf
from tensorflow.keras import Model, Input, layers, initializers
import numpy as np
import tensorflow.keras as keras

class ContractingBlock(keras.layers.Layer):
    def __init__(self, number_of_filters, kernel_size=3, activation='relu', kernel_initializer=keras.initializers.HeNormal, padding='same'):
        super().__init__()

        self.conv1 = keras.layers.Conv2D(number_of_filters, kernel_size, activation=activation, kernel_initializer=kernel_initializer, padding=padding)
        self.conv2 = keras.layers.Conv2D(number_of_filters, kernel_size, activation=activation, kernel_initializer=kernel_initializer, padding=padding)
        self.pool = keras.layers.MaxPool2D()
    
    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        outputs = self.pool(x) # downsample

        return outputs

class ExpandingBlock(keras.layers.Layer):
    def __init__(self, number_of_filters, kernel_size=3, activation='relu', kernel_initializer=initializers.HeNormal, padding='same'):
        super().__init__()

        self.conv1 = keras.layers.Conv2D(number_of_filters, kernel_size, activation=activation, kernel_initializer=kernel_initializer, padding=padding)
        self.conv2 = keras.layers.Conv2D(number_of_filters, kernel_size, activation=activation, kernel_initializer=kernel_initializer, padding=padding)
        self.up = keras.layers.Conv2DTranspose(number_of_filters/2, 2, strides=2, activation=activation, kernel_initializer=kernel_initializer)
        self.concat = keras.layers.Concatenate()

    def call(self, inputs, contracting_counterpart):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.up(x) # upsampling operation

        # this allows for the cross connections from the contracting half of the unet
        outputs = self.concat([tf.image.resize(contracting_counterpart, x.shape[1:3]), x])

        return outputs

class Unet(Model):
    def __init__(self, input_shape, K, starting_number_of_filters=64, levels=4, activation='relu', kernel_initializer=initializers.HeNormal, padding='same'):
        super().__init__()
        self.K = K
        input = Input(shape = input_shape)
        
        number_of_filters = starting_number_of_filters
        counterparts = []
        previous_down = input
        
        for i in range(levels):
            down, counterpart = self._contracting_block(previous_down, number_of_filters, activation=activation, kernel_initializer=kernel_initializer, padding=padding)
            counterparts.insert(0, counterpart)

            number_of_filters *= 2
            previous_down = down

        previous_smashed_together = previous_down

        for i in range(levels):
            smashed_together = self._expanding_block(previous_smashed_together, counterparts[i], number_of_filters, activation=activation, kernel_initializer=kernel_initializer, padding=padding)

            number_of_filters /= 2
            previous_smashed_together = smashed_together

        final_concat = previous_smashed_together
        conv = layers.Conv2D(number_of_filters, 3, activation=activation, kernel_initializer=kernel_initializer, padding=padding)(final_concat)
        conv = layers.Conv2D(number_of_filters, 3, activation=activation, kernel_initializer=kernel_initializer, padding=padding)(conv)
        output = layers.Conv2D(K, 1, activation='softmax', kernel_initializer=kernel_initializer, name='maps')(conv)

        super().__init__(input, output)
    
    # allows model to make predictions on arbitraily sized inputs
    def any_size_predict():
        pass

    # allows model to train on arbitraily sized input
    def any_size_fit():
        pass

    def _contracting_block(self, previous_down, number_of_filters, kernel_size=3, activation='relu', kernel_initializer=initializers.HeNormal, padding='same'):
        right_convolution = layers.Conv2D(number_of_filters, kernel_size, activation=activation, kernel_initializer=kernel_initializer, padding=padding)(previous_down)
        right_convolution = layers.Conv2D(number_of_filters, kernel_size, activation=activation, kernel_initializer=kernel_initializer, padding=padding)(right_convolution)
        down = layers.MaxPool2D()(right_convolution)

        return down, right_convolution

    def _expanding_block(self, previous_smashed_together, contracting_counterpart, number_of_filters, kernel_size=3, activation='relu', kernel_initializer=initializers.HeNormal, padding='same'):
        right_convolution = layers.Conv2D(number_of_filters, kernel_size, activation=activation, kernel_initializer=kernel_initializer, padding=padding)(previous_smashed_together)
        right_convolution = layers.Conv2D(number_of_filters, kernel_size, activation=activation, kernel_initializer=kernel_initializer, padding=padding)(right_convolution)
        up = layers.Conv2DTranspose(number_of_filters/2, 2, strides=2, activation=activation, kernel_initializer=kernel_initializer)(right_convolution)
        smashed_together = layers.Concatenate()([tf.image.resize(contracting_counterpart, up.shape[1:3]), up])

        return smashed_together



if __name__ == "__main__":
    # this example is the bottom part of the original unet from the whitepaper
    inputs = np.random.random((5,68,68,256))

    # test contracting block
    con = ContractingBlock(512, padding='valid')
    first_out = con(inputs)
    print(first_out.shape) # should be (5, 32, 32, 512)

    # test expanding block
    counterpart = first_out
    exp = ExpandingBlock(1024, padding='valid')
    second_out = exp(first_out, counterpart)
    print(second_out.shape) # should be (5, 56, 56, 1024)

    # # convolutions are valid, so output image is smaller than input image
    # model = Unet(input_shape=(572, 572, 1), K=3, padding='valid')
    # model.summary()
    # model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy','val_accuracy'])
    


    # # convolutions are padded to produce same size output image
    # model2 = Unet(input_shape=(1024, 1024, 1), K=3)
    # model2.summary()
    # model2.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy','val_accuracy'])
