# my implementation of Ronneberger's U-Net
# for expansion half of network, used resizing instead of cropping before concatenation
# loss function does not included pixel weight map yet

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
        cross_connection = self.conv2(x)
        outputs = self.pool(cross_connection) # downsample

        return outputs, cross_connection

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

class Unet(keras.Model):
    def __init__(self, K, starting_number_of_filters=64, levels=4, activation='relu', kernel_initializer=initializers.HeNormal, padding='same'):
        super().__init__()

        self.K = K # number of output classes
        self.con_blocks = [ContractingBlock(starting_number_of_filters*2**i, activation=activation, kernel_initializer=kernel_initializer, padding=padding) for i in range(levels)]
        self.exp_blocks = [ExpandingBlock(starting_number_of_filters*2**i, activation=activation, kernel_initializer=kernel_initializer, padding=padding) for i in range(levels, 0, -1)]
        self.conv1 = keras.layers.Conv2D(starting_number_of_filters, 3, activation=activation, kernel_initializer=kernel_initializer, padding=padding)
        self.conv2 = keras.layers.Conv2D(starting_number_of_filters, 3, activation=activation, kernel_initializer=kernel_initializer, padding=padding)
        self.maps = keras.layers.Conv2D(K, 1, activation='softmax', kernel_initializer=kernel_initializer)
    
    def call(self, inputs):
        # run the contracting half
        cross_inputs = []
        x = inputs
        for con in self.con_blocks:
            x, cross_connection = con(x)
            # print('Contract output shape: ', x.shape)

            # save the pre-pooled output to pipe into the expanidng half
            cross_inputs.insert(0, cross_connection)
        
        # run the expanding half
        for i, exp in enumerate(self.exp_blocks):
            x = exp(x, cross_inputs[i])
            # print('Connection shape: ', cross_inputs[i].shape)
            # print('Exp output shape: ', x.shape)
        
        # do the final convolutions to get the segmentations
        x = self.conv1(x)
        x = self.conv2(x)
        outputs = self.maps(x)
        # print('Final output shape: ', outputs.shape)

        return outputs

if __name__ == "__main__":
    # # this example is the bottom part of the original unet from the whitepaper
    # inputs = np.random.random((5,68,68,256))

    # # test contracting block
    # con = ContractingBlock(512, padding='valid')
    # first_out = con(inputs)
    # print(first_out.shape) # should be (5, 32, 32, 512)

    # # test expanding block
    # counterpart = first_out
    # exp = ExpandingBlock(1024, padding='valid')
    # second_out = exp(first_out, counterpart)
    # print(second_out[0].shape) # should be (5, 56, 56, 1024)

    # this is the paper's unet, matches perfectly (uncomment print statements in unet implementation to verify)
    x = np.random.random((5, 572, 572, 1))
    model = Unet(K=2, padding='valid')
    y = model.predict(x)

    # convolutions are valid, so output image is smaller than input image
    print(y.shape)

    # a unet with same padding, segmentation map does not shrink if height and width
    # are both divisible by 2**num_of_layers
    x = np.random.random((5, 512, 512, 4))
    model = Unet(K=3)
    y = model.predict(x)

    # convolutions are padded to produce same size output image
    print(y.shape)
    model.summary()
    
    # you can use compile and fit just like any tensorflow Model obj
    model.compile(optimizer='adam', loss='mse', metrics=['accuracy','val_accuracy'])
