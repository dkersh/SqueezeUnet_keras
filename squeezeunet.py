from tensorflow.keras.layers import Activation, Add, GlobalAveragePooling2D, BatchNormalization, Concatenate, Conv2D, Conv2DTranspose, DepthwiseConv2D, Input, Lambda, MaxPooling2D, ReLU, UpSampling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import numpy as np

# SqueezeUnet adapted from Squeeze U-Net: A Memory and Energy Efficient Image Segmentation Network , Beheshti et al, 2020

class squeezeUnet():
    def __init__(self, N_channels = 1, input_width = 256, input_height = 256, N_classes=1, dataset_name = None):
        self.N_channels = N_channels
        self.input_width = input_width
        self.input_height = input_height
        self.image_shape = (self.input_width, self.input_height, self.N_channels)
        self.N_classes = N_classes

        self.model = self.model()

    def fire_block(self, input_layer, squeeze=16, expand=64, ds=False):
        if ds:
            x = Conv2D(squeeze, (1, 1), padding='valid', strides=(2, 2), activation='relu')(input_layer)
        else:
            x = Conv2D(squeeze, (1, 1), padding='same', activation='relu')(input_layer)
        left = Conv2D(expand, (1, 1), padding='same', activation='relu')(x)
        right = Conv2D(expand, (3, 3), padding='same', activation='relu')(x)

        return Concatenate()([left, right])

    def fire_block_transpose(self, input_layer, squeeze=16, expand=64):
        x = Conv2DTranspose(squeeze, (1, 1), padding='valid', strides=(2, 2), activation='relu')(input_layer)
        left = Conv2DTranspose(expand, (1, 1), padding='same', activation='relu')(x)
        right = Conv2DTranspose(expand, (2, 2), padding='same', activation='relu')(x)
        
        return Concatenate()([left, right])

    def downsample_block(self, input_layer, squeeze, expand):
        x = self.fire_block(input_layer, squeeze, expand, ds=True)
        x = self.fire_block(x, squeeze, expand, ds=False)

        return x
    
    def upsample_block(self, input_layer1, input_layer2, squeeze_t, expand_t, squeeze):
        x = self.fire_block_transpose(input_layer1, squeeze_t, expand_t)
        x = Concatenate()([x, input_layer2])
        x = self.fire_block(x, squeeze, expand_t)
        x = self.fire_block(x, squeeze, expand_t)

        return x

    def model(self):
        main_input = Input(shape=self.image_shape)

        x = Conv2D(64, (3, 3), padding='same', activation='relu')(main_input)
        x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)

        ds1 = self.downsample_block(x, 32, 64)
        ds2 = self.downsample_block(ds1, 48, 128)
        ds3 = self.downsample_block(ds2, 64, 256)
        ds4 = self.downsample_block(ds3, 80, 512)

        us1 = self.upsample_block(ds4, ds3, 80, 256, 64)
        us2 = self.upsample_block(us1, ds2, 64, 128, 48)
        us3 = self.upsample_block(us2, ds1, 48, 64, 32)

        output = Conv2DTranspose(64, (2, 2), padding='same', strides=(2, 2), activation='relu')(us3)
        output = Concatenate()([output, x])
        output = Conv2D(64, (3, 3), padding='same', activation='relu')(output)
        output = Conv2D(64, (3, 3), padding='same', activation='relu')(output)
        output = Conv2D(self.N_classes, (1, 1), padding='same', activation='sigmoid')(output)

        model = Model(inputs=[main_input], outputs=[output])
        opt = Adam(lr=0.0002, beta_1=0.5) # Not mentioned in the paper
        model.compile(loss=['binary_crossentropy'], optimizer=opt)

        return model
