from unet import Unet


# convolutions are padded to produce same size output image
# K is the number of output classes
# Unet inherits from Model, so you can call fit() and predict(), etc.

model = Unet(input_shape=(1024, 1024, 1), K=3)
model.summary()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy','val_accuracy'])
