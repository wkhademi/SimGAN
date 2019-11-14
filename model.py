import keras.backend as K
import tensorflow as tf
from keras import Input, Model
from keras.optimizers import SGD
from keras.layers import Conv2D, ReLU, MaxPool2D, Reshape, Add


class SimGAN:
    """
    Implementation of SimGAN model.

    Details can be found here: https://arxiv.org/pdf/1612.07828.pdf
    """
    def __init__(self, input_shape, optimizer=SGD(2e-4, momentum=0.9), lambda_reg=10.0):
        self.lambda_reg = lambda_reg

        # build discriminator network
        self.discriminator = self.discriminator_network(input_shape)
        self.discriminator.compile(loss='categorical_crossentropy', optimizer=optimizer)

        # disable training for discriminator when training refiner
        self.discriminator.trainable = False

        # build refiner network
        self.refiner = self.refiner_network(input_shape)
        self.refiner.compile(loss=self.self_regularization_loss, optimizer=optimizer)

        # refine a set of synthetic images and run them through the discriminator
        inputs = Input(shape=input_shape)
        refined_inputs = self.refiner(inputs)
        refined_probs = self.discriminator(refined_inputs)

        # build adversarial network
        self.adversarial = Model(inputs=inputs, outputs=[refined_inputs, refined_probs],
                                 name='adversarial_model')
        self.adversarial.compile(loss=[self.self_regularization_loss, 'categorical_crossentropy'],
                                 optimizer=optimizer)


    def refiner_network(self, refiner_input_shape):
        """
        Refiner network of SimGAN meant for images of input size 224x224.
        """
        def resnet_block(inputs):
            """
            ResNet Block with skip connection.
            """
            x = Conv2D(64, 3, strides=1, padding='same', activation='relu')(inputs)
            x = Conv2D(64, 3, strides=1, padding='same')(x)
            skip = Add()([x, inputs])
            out = ReLU()(skip)

            return out

        inputs = Input(shape=refiner_input_shape)

        net = Conv2D(64, 7, strides=1, padding='same', activation='relu')(inputs)

        for _ in range(10):
            net = resnet_block(net)

        output_map = Conv2D(1, 1, strides=1, activation='tanh')(net)

        return Model(inputs=inputs, outputs=output_map, name='refiner')


    def discriminator_network(self, discriminator_input_shape):
        """
        Discriminator network of SimGAN meant for images of input size 224x224.
        """
        inputs = Input(shape=discriminator_input_shape)

        net = Conv2D(96, 7, strides=4, padding='same', activation='relu')(inputs)
        net = Conv2D(64, 5, strides=2, padding='same', activation='relu')(net)
        net = MaxPool2D(pool_size=3, strides=2, padding='same')(net)
        net = Conv2D(32, 3, strides=2, padding='same', activation='relu')(net)
        net = Conv2D(32, 1, strides=1, activation='relu')(net)

        output_map = Conv2D(2, 1, strides=1, activation='softmax')(net)

        return Model(inputs=inputs, outputs=output_map, name='discriminator')


    # def discriminator_loss(self, D_R_x, D_y, epsilon=1e-12):
    #     """
    #     Discriminator loss is Eq. 2 in: https://arxiv.org/pdf/1612.07828.pdf
    #     """
    #     loss = -1 * (K.mean(K.log(D_R_x + epsilon)) + K.mean(K.log(1 - D_y + epsilon)))
    #
    #     return loss
    #
    #
    # def refiner_loss(self, x, R_x, D_R_x, lambda_reg=10.0, epsilon=1e-12):
    #     """
    #     Refiner loss contains a realism loss and regularizer loss.
    #
    #     Eq. 4 in: https://arxiv.org/pdf/1612.07828.pdf
    #     """
    #     realism_loss = -1 * K.mean(K.log(1 - D_R_x + epsilon))
    #     regularizer_loss = self.self_regularization_loss(x, R_x, lambda_reg=lambda_reg)
    #
    #     return realism_loss + regularizer_loss


    def self_regularization_loss(self, x, R_x):
        """
        L1 norm performed on simulated image, x, and the refined image, R(x).

        This loss helps to preserve the structure of the original simulated image
        while refining it.
        """
        l1_norm = K.mean(K.abs(R_x - x))

        return self.lambda_reg * l1_norm
