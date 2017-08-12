# coding=utf-8
"""
    Evaluate colorful_model model
"""

import os
import numpy as np

from . import colorization_model
from ..utils import batch_utils
from ..utils import colorful_utils


class EvaluateModel(object):
    """
    Evaluate class
    """

    def __init__(self, **kwargs):
        """
        Constructor
        """
        self._data_file = kwargs["data_file"]
        self._model_name = kwargs["model_name"]
        self._epoch = kwargs["epoch"]
        self._t_parameter = kwargs["T"]
        self._batch_size = kwargs["batch_size"]
        self._number_of_neighbors = kwargs["nb_neighbors"]
        self._path_to_gray_scale = kwargs["gray_scale_image"]
        self._root_dir = kwargs["root_dir"]

        # noinspection PyTypeChecker
        self._image_size = int(os.path.basename(self._data_file).split("_")[2].split(".")[0])

    def evaluate_model(self):
        """
        Evaluate the model
        :return:
        """

        # create a batch generator for the color data
        data_generator = batch_utils.DataBatchGenerator(self._data_file, batch_size=self._batch_size, max_processes=1)

        channels, height, width = data_generator.configuration["data_shape"][1:]

        # load the array of quantized ab values
        gamut_points = os.path.join(self._root_dir, 'uploader/colorare_automata/data/processed/pts_in_hull.npy')

        quantized_ab = np.load(gamut_points)
        nb_quantized = quantized_ab.shape[0]

        # Load colorization model
        model_weights = os.path.join(self._root_dir,
                                     "uploader/colorare_automata/models/{}/{}_weights_epoch_{}.h5".format(self._model_name, self._model_name,
                                                                                                          self._epoch))

        network_model = colorization_model.ConvNetModel().load_model(self._model_name, nb_quantized, (1, height, width), self._batch_size)
        network_model.load_weights(model_weights)

        image_l_rs, img_l, (height_orig, width_orig), (height_out, width_out) = data_generator.generate_cnn_input(network_model,
                                                                                                                  self._path_to_gray_scale)

        image_colored = colorful_utils.color_gray_scale_image(network_model, quantized_ab, image_l_rs, self._batch_size, height, width, nb_quantized,
                                                              self._t_parameter, img_l, size_original=(height_orig, width_orig),
                                                              size_out=(height_out, width_out), root_dir=self._root_dir)
        return image_colored
