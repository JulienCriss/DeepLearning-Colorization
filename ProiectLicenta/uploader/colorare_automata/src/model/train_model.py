# coding=utf-8
"""
    Train the CNN model
"""
import glob
import os
import h5py
import numpy as np
import time
import sklearn.neighbors as nn
import keras.backend as keras_backend
from keras.optimizers import Adam, RMSprop, Adadelta
from keras.losses import categorical_crossentropy
from keras.utils import plot_model

from . import colorization_model
from ..utils import batch_utils
from ..utils import colorful_utils


def categorical_crossentropy_color(y_true, y_pred):
    """
    Custom categorical cross entropy function for Theano backend
    :param y_true: 
    :param y_pred: 
    :return: 
    """
    n, height, width, q = y_true.shape
    y_true = keras_backend.reshape(y_true, (n * height * width, q))
    y_pred = keras_backend.reshape(y_pred, (n * height * width, q))

    weighs = y_true[:, 313:]
    weighs = keras_backend.concatenate([weighs] * 313, axis=1)

    # remove last column for y_true and y_pred
    y_true = y_true[:, :-1]
    y_pred = y_pred[:, :-1]

    y_true = y_true * weighs
    cross_entropy = keras_backend.categorical_crossentropy(y_pred, y_true)
    cross_entropy = keras_backend.mean(cross_entropy, axis=-1)

    return cross_entropy


def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='=', train_result=None):  # fill='█'
    """
    Call in a loop to create terminal progress bar
    :param iteration: Required : current iteration
    :param total: Required  : total iterations
    :param prefix: Optional  : prefix string
    :param suffix: Optional  : suffix string
    :param decimals: Optional  : positive number of decimals in percent complete
    :param length: Optional  : character length of bar
    :param fill: Optional  : bar fill character
    :param train_result: Optional : train loss
    :type iteration: int
    :type total: int
    :type prefix: str
    :type suffix: str
    :type decimals: int
    :type length: int
    :type train_result: list
    """
    if train_result is None:
        train_result = [0, 0]

    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration / total)
    bar = fill * filled_length + '>' + '-' * (length - filled_length)
    print('%s |%s| %s%% %s | Loss(%s) Acc(%s) | Iteration %s / %s' % (prefix, bar, percent, suffix, "{0:.7f}".format(float(train_result[0])),
                                                                      "{0:.7f}".format(float(train_result[1])),
                                                                      iteration, total))

    # Print New Line on Complete
    if iteration == total:
        print()


class TrainModel(object):
    """
    Train a model
    """

    def __init__(self, **kwargs):
        """
        Constructor
        :param kwargs: 
        """
        self._batch_size = kwargs["batch_size"]
        self._nb_batch_per_epoch = kwargs["n_batch_per_epoch"]
        self._nb_epoch = kwargs["nb_epoch"]
        self._hdf5_data_file = kwargs["data_file"]
        self._nb_neighbors = kwargs["nb_neighbors"]
        self._model_name = kwargs["model_name"]
        self._training_mode = kwargs["training_mode"]
        self._optimizer = kwargs["optimizer"]
        self._epoch = kwargs["epoch"]

        # noinspection PyTypeChecker
        self._image_size = int(os.path.basename(self._hdf5_data_file).split("_")[2].split(".")[0])

    def train_model(self):
        """
        Train the model
        :return: 
        """
        # Setup directories to save model, architecture etc
        colorful_utils.setup_logging(self._model_name)

        # Create a batch generator for the color data
        data_generator = batch_utils.DataBatchGenerator(self._hdf5_data_file, self._batch_size)

        channels, height, width = data_generator.configuration["data_shape"][1:]

        # load the array of quantized ab values
        quantized_ab = np.load("../../data/processed/pts_in_hull.npy")
        nb_quantized = quantized_ab.shape[0]

        # fit a nearest neighbor to a quantized ab
        nn_finder = nn.NearestNeighbors(n_neighbors=self._nb_neighbors, algorithm="ball_tree").fit(quantized_ab)

        # Load the color prior factor that encourages rare colors
        prior_factor = np.load("../../data/processed/DataSet_prior_factor_{}.npy".format(self._image_size))

        x_train = None

        # Load and rescale data
        if self._training_mode == "in_memory":
            with h5py.File(self._hdf5_data_file, "r") as hf_handler:
                x_train = hf_handler["training_lab_data"][:]

        # Remove possible previous figures to avoid confusion
        for _file in glob.glob("../../figures/*.png"):
            os.remove(_file)

        try:
            # Create and optimizer
            _optimizer = None
            print('------------------ SUMMARY --------------------')
            print("\t Optimizer: ", self._optimizer)
            print("\t Batch size: ", self._batch_size)
            print('-----------------------------------------------')

            if self._optimizer == 'adam':
                # Adam(lr=0.000003, epsilon=1e-08, decay=10 ** (-3))  # 1e-4
                _optimizer = Adam(lr=1e-4, epsilon=1e-08)

            elif self._optimizer == 'adadelta':
                _optimizer = Adadelta(lr=1e-4, rho=0.95, epsilon=1e-08)

            elif self._optimizer == 'rms_prop':
                _optimizer = RMSprop(lr=1e-4, rho=0.9, epsilon=1e-08)

            # Load the CNN model
            _model = colorization_model.ConvNetModel().load_model(self._model_name, nb_quantized, (1, height, width), self._batch_size)

            if keras_backend.backend() == "tensorflow":
                _model.compile(optimizer=_optimizer, loss=categorical_crossentropy, metrics=['accuracy'])

            elif keras_backend.backend() == "theano":
                _model.compile(optimizer=_optimizer, loss=categorical_crossentropy_color, metrics=['accuracy'])

            _model.summary()
            plot_model(_model, to_file='../../figures/cnn_model.png', show_shapes=True, show_layer_names=True)

            x_batch_black, x_batch_color = None, None

            training_loop = range(self._nb_epoch)

            if self._epoch > -1:
                print("Use weights: ")
                weights_path_last = os.path.join('../../models/%s_v16/%s_weights_epoch_%s.h5' % (self._model_name, self._model_name, self._epoch))
                print("-- Loading weights from: {}".format(weights_path_last))
                _model.load_weights(weights_path_last)

                training_loop = range(self._epoch + 1, self._nb_epoch)

            # Training loop
            for epoch in training_loop:

                batch_counter = 0
                start_time = time.time()
                print("** Start Epoch %s/%s" % (epoch, self._nb_epoch))
                print_progress_bar(batch_counter, self._nb_batch_per_epoch, prefix='Progress:', suffix='Complete', length=50, train_result=[0, 0])
                batch_counter += 1

                if self._training_mode == "in_memory":
                    _batch_generator = data_generator.generate_batch_in_memory(x_train, nn_finder, nb_quantized, prior_factor)
                else:
                    _batch_generator = data_generator.generate_batch(nn_finder, nb_quantized, prior_factor)

                # for batch in _batch_generator:
                #     x_batch_black, x_batch_color, y_batch = batch
                #
                #     # train the model with fit method
                #     _model.fit(x_batch_black, y_batch, batch_size=self._batch_size, epochs=self._nb_batch_per_epoch, verbose=1)
                #     break
                # weights = _model.get_weights()
                # _model.set_weights(weights)

                for batch in _batch_generator:

                    x_batch_black, x_batch_color, y_batch = batch
                    # train on batch method
                    train_loss = _model.train_on_batch(x_batch_black / 100., y_batch)

                    print_progress_bar(batch_counter, self._nb_batch_per_epoch, prefix='Progress:', suffix='Complete', length=50,
                                       train_result=train_loss)

                    batch_counter += 1
                    if batch_counter > self._nb_batch_per_epoch:
                        break

                print("")
                print('** End of Epoch %s/%s, Time: %s seconds' % (epoch, self._nb_epoch, time.time() - start_time))

                # Plot some data with original, black and white and colorized versions side by side
                colorful_utils.plot_batch_on_train(_model, quantized_ab, x_batch_black, x_batch_color, self._batch_size, height, width, nb_quantized, epoch)

                # Save weights every 8 epoch
                if epoch % 8 == 0:
                    weights_path = os.path.join('../../models/%s/%s_weights_epoch_%s.h5' % (self._model_name, self._model_name, epoch))
                    _model.save_weights(weights_path, overwrite=True)

        except KeyboardInterrupt:
            pass
