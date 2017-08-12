# coding=utf-8
"""
    Colorful model
"""
import keras.backend as keras_backend
from keras.regularizers import l2
from keras.layers import Conv2D, concatenate, Input, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Activation, Lambda
from keras.models import Model
from keras.layers.convolutional import UpSampling2D


class ConvNetModel(object):
    """
        Colorful class
        Define and create a CNN model with Keras
    """

    def __init__(self):
        """
        Constructor
        """
        self._block_name = None
        self.filters = None
        self.blocks = None
        self.padding = None
        self.strides = None
        self.dilation_rate = None

    def residual_block(self, last_layer, nb_filter, block_idx, batch_normalization=True, weight_decay=0):
        """
        Create the residual block and then concatenate him with identity
        :param last_layer: input shape
        :param nb_filter: number of filters
        :param block_idx: block id
        :param batch_normalization: flag to use batch normalization 
        :param weight_decay: 
        :return: 
        """

        # create 1st Convolution layer
        self._block_name = "block{}_conv2D_{}".format(block_idx, "a")
        weight_region = l2(weight_decay)

        layer = Conv2D(nb_filter, kernel_size=(3, 3), padding="same", kernel_regularizer=weight_region, name=self._block_name)(last_layer)

        if batch_normalization:
            layer = BatchNormalization(axis=1, name="block{}_bn_{}".format(block_idx, "a"))(layer)

        layer = Activation("relu", name="block{}_relu_{}".format(block_idx, "a"))(layer)

        # create 2nd Convolution layer
        self._block_name = "block{}_conv2D_{}".format(block_idx, "b")
        weight_region = l2(weight_decay)

        layer = Conv2D(nb_filter, kernel_size=(3, 3), padding="same", kernel_regularizer=weight_region, name=self._block_name)(layer)
        if batch_normalization:
            layer = BatchNormalization(axis=1, name="block{}_bn_{}".format(block_idx, "b"))(layer)

        layer = Activation("relu", name="block{}_relu_{}".format(block_idx, "b"))(layer)

        # Concatenate residual block and identity
        last_layer = concatenate(inputs=[last_layer, layer], name="block{}_merge".format(block_idx), axis=1)

        return last_layer

    def convolutional_block(self, layer, block_idx, nb_filter, nb_conv_blocks, subsample):
        """
        Create a big convolutional layer for CNN
        Each conv layer refers to a block of 2 or 3 repeated cov and ReLU layers, followed by a BatchNorm layer
        :param layer: The last layer created
        :param block_idx: The index of current block
        :param nb_filter: Number of filters to have
        :param nb_conv_blocks: How much blocks to create
        :param subsample: 
        :return: 
        """
        print("Creating Conv {}:".format(block_idx))

        for i in range(nb_conv_blocks):

            self._block_name = "conv{}_{}".format(block_idx, i + 1)
            print("\t Creating block: {}".format(self._block_name))

            if i < nb_conv_blocks - 1:
                layer = Conv2D(nb_filter, kernel_size=(3, 3), padding="same", name=self._block_name)(layer)
                layer = Activation("relu", name='relu{}_{}'.format(block_idx, i + 1))(layer)
                layer = BatchNormalization(axis=1, name='conv{}_{}norm'.format(block_idx, i + 1))(layer)
            else:
                layer = Conv2D(nb_filter, kernel_size=(3, 3), strides=subsample, padding="same", name=self._block_name)(layer)
                layer = Activation("relu", name='relu{}_{}'.format(block_idx, i + 1))(layer)
                layer = BatchNormalization(axis=1, name='conv{}_{}norm'.format(block_idx, i + 1))(layer)
        print('-----------------------------------------------')
        return layer

    def atrous_block(self, x, block_idx, nb_filter, nb_conv_blocks):
        """
        AtrousConvolution2D
        :param x: The last layer created
        :param block_idx: The index of current block
        :param nb_filter: Number of filters to have
        :param nb_conv_blocks: How much blocks to create
        :return: 
        """
        print("Creating Conv {}:".format(block_idx))

        for i in range(nb_conv_blocks):
            self._block_name = "conv{}_{}".format(block_idx, i + 1)
            print("\t Creating block: {}".format(self._block_name))

            x = Conv2D(nb_filter, kernel_size=(3, 3), padding="same", dilation_rate=1, name=self._block_name)(x)
            x = Activation("relu", name='relu{}_{}'.format(block_idx, i + 1))(x)
            x = BatchNormalization(axis=1, name='conv{}_{}norm'.format(block_idx, i + 1))(x)

        print('-----------------------------------------------')
        return x

    def colorful_model(self, nb_classes, image_dim, batch_size, model_name="colorful_model"):
        """
        This model represents the CNN model its a copy of VGG model with some modifications
        :param nb_classes: Number of classes
        :param image_dim: Image dimension
        :param batch_size: Batch size to process
        :param model_name: A name that will be given to this model
        :return: 
        """

        height, width = image_dim[1:]
        x_input = Input(shape=image_dim, name="input")

        # keep track the  size of image
        current_height, current_width = image_dim[1:]

        # convolutional parameters
        filter_size_list = [64, 128, 256, 512, 512]

        block_size_list = [2, 2, 3, 3, 3]
        subsample_list = [(2, 2), (2, 2), (2, 2), (1, 1), (1, 1)]

        # create atrous blocks parameters
        atrous_filter_size_list = [512, 512]
        atrous_block_size_list = [3, 3]

        block_idx = 1

        # Create the first block
        _filters, _blocks, _subsample = filter_size_list[0], block_size_list[0], subsample_list[0]
        x = self.convolutional_block(x_input, block_idx, _filters, _blocks, _subsample)
        block_idx += 1

        # make spatial downsampling
        current_height, current_width = current_height / _subsample[0], current_width / _subsample[1]

        # create next blocks
        for _filters, _blocks, _subsample in zip(filter_size_list[1:-1], block_size_list[1:-1], subsample_list[1:-1]):
            x = self.convolutional_block(x, block_idx, _filters, _blocks, _subsample)
            block_idx += 1
            # make spatial downsampling
            current_height, current_width = int(current_height) / _subsample[0], int(current_width) / _subsample[1]

        # next blocks are the atrous blocks
        for idx, (_filters, _blocks) in enumerate(zip(atrous_filter_size_list, atrous_block_size_list)):
            x = self.atrous_block(x, block_idx, _filters, _blocks)
            block_idx += 1

        # Until now we have created first 6 blocks
        # Create block 7
        _filters, _blocks, _subsample = filter_size_list[-1], block_size_list[-1], subsample_list[-1]
        x = self.convolutional_block(x, block_idx, _filters, _blocks, _subsample)
        block_idx += 1

        # make spatial downsampling
        current_height, current_width = int(current_height) / _subsample[0], int(current_width) / _subsample[1]

        # create last block: Block 8
        x = UpSampling2D(size=(2, 2), name="UpSampling2D")(x)

        # TODO self.convolutional_block(x, block_idx, 128, 3, (1, 1))
        x = self.convolutional_block(x, block_idx, 256, 2, (1, 1))

        # block_idx += 1
        # x = self.convolutional_block(x, block_idx, 128, 2, (1, 1))
        block_idx += 1

        # make spatial upsampling
        current_height, current_width = int(current_height * 2), int(current_width * 2)

        # create the final convolutional block
        x = Conv2D(nb_classes, kernel_size=(1, 1), padding="same", name="conv2D_final", data_format="channels_first")(x)

        # Reshape Softmax
        def output_shape(input_shape):
            """
            Use this function to define output shape for softmax
            :param input_shape: 
            :return: 
            """
            shape = (batch_size, height, width, nb_classes + 1)
            return shape

        def reshape_softmax(x_param):
            """
            Use this function to define output shape for softmax
            :param x_param: The input
            :return:
            """
            x_param = keras_backend.permute_dimensions(x_param, [0, 2, 3, 1])  # last dimension is number of filters

            shape = (batch_size * height * width, nb_classes)
            x_param = keras_backend.reshape(x_param, shape=shape)
            x_param = keras_backend.softmax(x_param)

            # Add a zero column so that x has the same dimension as the target (313 classes + 1 weight)
            xc = keras_backend.zeros((batch_size * height * width, 1))
            x_param = keras_backend.concatenate([x_param, xc], axis=1)

            # Reshape back to (batch_size, height, width, nb_classes + 1) to satisfy keras shape checks
            shape = (batch_size, height, width, nb_classes + 1)
            x_param = keras_backend.reshape(x_param, shape=shape)

            return x_param

        _reshape_softmax_callback = Lambda(lambda z: reshape_softmax(z), output_shape=output_shape, name="ReshapeSoftmax")
        x = _reshape_softmax_callback(x)

        # build the model
        _model = Model(inputs=[x_input], outputs=[x], name=model_name)

        return _model

    def create_vgg_conv_block(self, layer, block_idx, nb_filters, nb_conv_blocks, _strides=None, _padding=None, _kernel_size=(1, 1)):
        """
        Create a big convolutional layer for CNN
        Each conv layer refers to a block of 2 or 3 repeated conv and ReLU layers, followed by a BatchNorm layer
        :param layer: The last layer created
        :param block_idx: The index of current block
        :param nb_filters: Number of output filters
        :param nb_conv_blocks: How many blocks to create in this Conv block
        :param _strides: Int or a tuple if you want to use stride, leave None to not use
        :param _padding: Int or a tuple if you want to use padding, leave None to not use
        :param _kernel_size: Int or tuple; represents the size of filter window
        :type layer: Layer
        :type block_idx: int
        :type nb_filters: int
        :type nb_conv_blocks: int
        :type _strides: int | tuple
        :type _padding: int | tuple | None
        :type _kernel_size: tuple
        :return: A bunch of layers who represents a big convolution block
        """
        print("Creating Conv {}:".format(block_idx))

        for i in range(nb_conv_blocks):

            self._block_name = "conv{}_{}".format(block_idx, i + 1)
            print("\t Creating block: {}".format(self._block_name))

            if i < nb_conv_blocks - 1:

                if _padding:
                    layer = ZeroPadding2D(padding=_padding, name='padding{}_{}'.format(block_idx, i + 1))(layer)

                layer = Conv2D(nb_filters, kernel_size=_kernel_size, padding="same", name=self._block_name)(layer)
                layer = Activation("relu", name='relu{}_{}'.format(block_idx, i + 1))(layer)

            else:

                if _padding:
                    layer = ZeroPadding2D(padding=_padding, name='padding{}_{}'.format(block_idx, i + 1))(layer)

                if _strides:
                    layer = Conv2D(nb_filters, kernel_size=_kernel_size, padding="same", name=self._block_name, strides=_strides)(layer)
                else:
                    layer = Conv2D(nb_filters, kernel_size=_kernel_size, padding="same", name=self._block_name)(layer)

                layer = Activation("relu", name='relu{}_{}'.format(block_idx, i + 1))(layer)
                layer = BatchNormalization(axis=1, name='conv{}_{}norm'.format(block_idx, i + 1))(layer)

        print('-----------------------------------------------')
        return layer

    def create_vgg_special_conv_block(self, layer, block_idx, nb_filters, nb_conv_blocks, _strides=None, _padding=None, _kernel_size=(1, 1),
                                      dilation_rate=None):
        """
        Create a big convolutional layer for CNN
        Each conv layer refers to a block of 2 or 3 repeated conv and ReLU layers, followed by a BatchNorm layer
        This is called 'special' because of dilation rate and strides combination
        :param dilation_rate: Dilation rate
        :param layer: The last layer created
        :param block_idx: The index of current block
        :param nb_filters: Number of output filters
        :param nb_conv_blocks: How many blocks to create in this Conv block
        :param _strides: Int or a tuple if you want to use stride, leave None to not use
        :param _padding: Int or a tuple if you want to use padding, leave None to not use
        :param _kernel_size: Int or tuple; represents the size of filter window
        :type layer: Layer
        :type block_idx: int
        :type nb_filters: int
        :type nb_conv_blocks: int
        :type _strides: int | tuple
        :type _padding: int | tuple | None
        :type _kernel_size: tuple
        :return:
        """
        print("Creating Conv {}:".format(block_idx))

        for i in range(nb_conv_blocks):

            self._block_name = "conv{}_{}".format(block_idx, i + 1)
            print("\t Creating block: {}".format(self._block_name))

            if i < nb_conv_blocks - 1:

                if _padding:
                    layer = ZeroPadding2D(padding=_padding, name='padding{}_{}'.format(block_idx, i + 1))(layer)

                if _strides and dilation_rate:
                    layer = Conv2D(nb_filters, kernel_size=_kernel_size, padding="same", name=self._block_name, strides=_strides,
                                   dilation_rate=dilation_rate)(
                        layer)

                elif dilation_rate:
                    layer = Conv2D(nb_filters, kernel_size=_kernel_size, padding="same", name=self._block_name, dilation_rate=dilation_rate)(layer)

                layer = Activation("relu", name='relu{}_{}'.format(block_idx, i + 1))(layer)

            else:

                if _padding:
                    layer = ZeroPadding2D(padding=_padding, name='padding{}_{}'.format(block_idx, i + 1))(layer)

                if _strides and dilation_rate:
                    layer = Conv2D(nb_filters, kernel_size=_kernel_size, padding="same", name=self._block_name, strides=_strides,
                                   dilation_rate=dilation_rate)(layer)

                elif dilation_rate:
                    layer = Conv2D(nb_filters, kernel_size=_kernel_size, padding="same", name=self._block_name, dilation_rate=dilation_rate)(layer)

                layer = Activation("relu", name='relu{}_{}'.format(block_idx, i + 1))(layer)
                layer = BatchNormalization(axis=1, name='conv{}_{}norm'.format(block_idx, i + 1))(layer)

        print('-----------------------------------------------')
        return layer

    def vgg_model(self, nb_classes, image_dim, batch_size, model_name="vgg_model"):
        """
        This model represents the CNN VGG model with some modifications
        :param nb_classes: Number of classes
        :param image_dim: Image dimension
        :param batch_size: Batch size to process
        :param model_name: A name that will be given to this model
        :return:
        """
        height, width = image_dim[1:]

        # define the Input
        layer_input = Input(shape=image_dim, name='data_l')

        # define network parameters
        self.filters = [64, 128, 256, 512, 512, 512, 512]
        self.blocks = [2, 2, 3, 3, 3, 3, 3]
        self.padding = [None for i in range(0, 8)]  # [1, 1, 1, 1, 2, 2, 1]
        self.strides = [2, 2, 2, 1, 1, None, None]
        self.dilation_rate = [None, None, None, 1, 2, 2, 1]

        # Create Conv 1
        block_idx = 1
        layer = self.create_vgg_conv_block(layer=layer_input, block_idx=block_idx, nb_filters=self.filters[0], nb_conv_blocks=self.blocks[0],
                                           _strides=self.strides[0], _padding=self.padding[0], _kernel_size=(3, 3))

        block_idx += 1

        # Create next blocks from Conv2 -> Conv7
        for _blocks, _filters, _padding, _strides, _dilation_rate in zip(self.blocks[1:], self.filters[1:], self.padding[1:], self.strides[1:],
                                                                         self.dilation_rate[1:]):

            if block_idx not in [4, 5, 6, 7]:
                layer = self.create_vgg_conv_block(layer=layer, block_idx=block_idx, nb_filters=_filters, nb_conv_blocks=_blocks,
                                                   _strides=_strides, _padding=_padding, _kernel_size=(3, 3))
            else:
                layer = self.create_vgg_special_conv_block(layer=layer, block_idx=block_idx, nb_filters=_filters, nb_conv_blocks=_blocks,
                                                           _strides=_strides, _padding=_padding, _kernel_size=(3, 3), dilation_rate=_dilation_rate)
            block_idx += 1

        # Conv8
        layer = UpSampling2D(size=(2, 2), name="UpSampling2D")(layer)
        # layer = self.convolutional_block(layer, 8, 256, 3, (1, 1))
        layer = self.create_vgg_special_conv_block(layer=layer, block_idx=8, nb_filters=256, nb_conv_blocks=3, _kernel_size=(3, 3),
                                                   dilation_rate=1)  # or padding=1

        # create the final convolutional block
        layer = Conv2D(nb_classes, kernel_size=(1, 1), padding="same", name="conv2D_final", data_format="channels_first")(layer)

        # Softmax
        def output_shape(input_shape):
            """
            Use this function to define output shape for softmax
            :param input_shape:
            :return:
            """
            shape = (batch_size, height, width, nb_classes + 1)
            return shape

        def reshape_softmax(x_layer):
            """
            The softmax layer
            :param x_layer: Last output of last layer from ConvNet
            :return:
            """
            # last dimension is number of filters
            x_layer = keras_backend.permute_dimensions(x_layer, [0, 2, 3, 1])

            shape = (batch_size * height * width, nb_classes)
            x_layer = keras_backend.reshape(x_layer, shape=shape)
            x_layer = keras_backend.softmax(x_layer)

            # Add a zero column so that x has the same dimension as the target (313 classes + 1 weight)
            xc = keras_backend.zeros((batch_size * height * width, 1))
            x_layer = keras_backend.concatenate([x_layer, xc], axis=1)

            # Reshape back to (batch_size, height, width, nb_classes + 1) to satisfy keras shape checks
            shape = (batch_size, height, width, nb_classes + 1)
            x_layer = keras_backend.reshape(x_layer, shape=shape)

            return x_layer

        _reshape_softmax_callback = Lambda(lambda z: reshape_softmax(z), output_shape=output_shape, name="ReshapeSoftmax")
        layer = _reshape_softmax_callback(layer)

        # Create the model
        _model = Model(inputs=[layer_input], outputs=[layer], name=model_name)

        return _model

    def load_model(self, model_name, nb_classes, image_dim, batch_size):
        """
        Load a specific model
        :param model_name: 
        :param nb_classes: 
        :param image_dim: 
        :param batch_size: 
        :return: 
        """

        if model_name == "colorful_model":
            model = self.colorful_model(nb_classes, image_dim, batch_size, model_name)
        else:
            model = self.vgg_model(nb_classes, image_dim, batch_size, model_name)

        return model
