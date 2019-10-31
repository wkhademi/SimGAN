import tensorflow as tf
from keras import Input, Model
from keras.layers import Conv2D, ReLU, MaxPool2D, Upsampling2D, Reshape, Add

class SimGAN:
    """
    Implementation of SimGAN model.

    Details can be found here: https://arxiv.org/pdf/1612.07828.pdf
    """
    def __init__(self, refiner_input_shape, discriminator_input_shape):
        refiner_inputs = Input(shape=refiner_input_shape)
        discriminator_inputs = Input(shape=discriminator_input_shape)

        self.refiner = self.refiner_network(refiner_inputs)
        self.discriminator = self.discriminator_network(discriminator_inputs)

    def refiner_network(self, inputs):
        """
        Refiner network of SimGAN meant for images of input size 224x224.
        """
        def resnet_block(self, inputs):
            x = Conv2D(64, 3, strides=1, padding='same', activation='relu')(inputs)
            x = Conv2D(64, 3, strides=1, padding='same')(x)
            skip = Add()([x, inputs])
            out = ReLU()(out)

            return out

        net = Conv2D(64, 7, strides=1, padding='same', activation='relu')(inputs)

        for _ in range(10):
            net = resnet_block(net)

    def discriminator_network(self, inputs):
        """
        Discriminator network of SimGAN meant for images of input size 224x224.
        """
        net = Conv2D(96, 7, strides=4, padding='same', activation='relu')(inputs)
        net = Conv2D(64, 5, strides=2, padding='same', activation='relu')(net)
        net = MaxPool2D(pool_size=3, strides=2, padding='same')(net)
        net = Conv2D(32, 3, strides=2, padding='same', activation='relu')(net)
        net = Conv2D(32, 1, strides=1, activation='relu')(net)
        net = Conv2D(2, 1, strides=1)(net)
        output_map = Reshape((-1, 2))(net) # [b,w,h,c] --> [b, wh, c]

        return output_map

    def refiner_loss(self):
        pass

    def discriminator_loss(self):
        pass

    def self_regularization_loss(self):
        pass
