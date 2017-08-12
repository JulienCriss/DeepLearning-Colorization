# coding=utf-8
"""
    Data generator module
"""

import time
import traceback
import numpy as np
import multiprocessing
import os
import h5py
import skimage.color as color
import caffe


class DataBatchGenerator(object):
    """
    This class will help to generate mini-batches with real-time data parallel augmentation on CPU
    """

    def __init__(self, hdf5_file, batch_size=32, data_set="training", max_processes=8, num_cached=10):
        """
        Constructor
        :param hdf5_file: Path to data in HDF5 format
        :param batch_size: Mini-batch size
        :param data_set: train/test/valid, the name of the data set to iterate over
        :param max_processes: max number of processes to spawn in parallel
        :param num_cached: max number of batches to keep in queue
        :type hdf5_file: str
        :type batch_size: int
        :type data_set: str
        :type max_processes: int
        :type num_cached: int
        """

        # First check if the HDF5 file exists
        assert os.path.isfile(hdf5_file), hdf5_file + " doesn't exists. Please check if the file exists or you provide the correct path"

        self._data_set = data_set
        self._max_processes = max_processes
        self._hdf5_file = hdf5_file
        self._batch_size = batch_size
        self._num_cached = num_cached

        # Dictionary that will store all transformation and their parameters
        self._transformations = {}
        self._queue = None

        # Read the data file to get data set shape information
        with h5py.File(self._hdf5_file, "r") as hf_handler:
            self.x_shape = hf_handler["{}_lab_data".format(self._data_set)].shape

            assert len(self.x_shape) == 4, "\n\n Image data should be formatted as: (n_samples, n_channels, height, width)"

            self.n_samples = hf_handler["{}_lab_data".format(self._data_set)].shape[0]

            # verify if n_channels is at index 1
            assert self.x_shape[-3] < min(self.x_shape[-2:]), "\n\n Image data should be formatted as: (n_samples, n_channels, height, width)"

        # save the class interval variables to a configuration dictionary
        self._config_dict = {"hdf5_file": self._hdf5_file,
                             "batch_size": self._batch_size,
                             "data_set": self._data_set,
                             "num_cached": self._num_cached,
                             "max_processes": self._max_processes,
                             "data_shape": self.x_shape}

    @property
    def configuration(self):
        """
        Return configuration
        :return: 
        """
        return self._config_dict

    @staticmethod
    def get_soft_encoding(x, nn_finder, nb_q):
        """
        Get soft encoding
        :param x: Input
        :param nn_finder: Nearest Neighbor finder
        :param nb_q: 
        :return: 
        """
        sigma_neighbor = 5

        # get the distance to and the index of the nearest neighbor
        dist_neighbor, index_neighbor = nn_finder.kneighbors(x)

        # smooth the weights with a gaussian kernel
        weights = np.exp(-dist_neighbor ** 2 / (2 * sigma_neighbor ** 2))
        weights = weights / np.sum(weights, axis=1)[:, np.newaxis]

        # format the target
        soft_encoding = np.zeros((x.shape[0], nb_q))
        idx_points = np.arange(x.shape[0])[:, np.newaxis]
        soft_encoding[idx_points, index_neighbor] = weights

        return soft_encoding

    def producer(self, nn_finder, nb_quantized, prior_factor):
        """
        Define a producer
        """
        try:
            with h5py.File(self._hdf5_file, "r") as hf_handler:

                # select start index at random for the batch
                index_start = np.random.randint(0, self.x_shape[0] - self._batch_size)
                index_end = index_start + self._batch_size

                # get x and y
                x_batch_color = hf_handler["{}_lab_data".format(self._data_set)][index_start: index_end, :, :, :]
                x_batch_black = x_batch_color[:, :1, :, :]
                x_batch_ab = x_batch_color[:, 1:, :, :]

                npts, channels, height, width = x_batch_ab.shape
                x_a = np.ravel(x_batch_ab[:, 0, :, :])
                x_b = np.ravel(x_batch_ab[:, 1, :, :])

                x_batch_ab = np.vstack((x_a, x_b)).T

                y_batch = self.get_soft_encoding(x_batch_ab, nn_finder, nb_quantized)

                # add priority weight to y_batch
                index_max = np.argmax(y_batch, axis=1)
                weights = prior_factor[index_max].reshape(y_batch.shape[0], 1)
                y_batch = np.concatenate((y_batch, weights), axis=1)

                # reshape the y_batch
                y_batch = y_batch.reshape((npts, height, width, nb_quantized + 1))

                # add data in queue
                self._queue.put((x_batch_black, x_batch_color, y_batch))
        except Exception as exc:
            print(traceback.format_exc(exc))

    def generate_batch(self, nn_finder, nb_quantized, prior_factor):
        """
        Use multiprocessing to generate batches in parallel
        :param nn_finder: Nearest neighbor finder
        :param nb_quantized: Quantized ab values
        :param prior_factor: 
        :return: 
        """
        processes = []
        self._queue = multiprocessing.Queue(maxsize=self._num_cached)

        try:

            def start_process():
                """
                Start a process for data generator
                """

                for i in range(len(processes), self._max_processes):
                    np.random.seed()

                    thread = multiprocessing.Process(target=self.producer, args=(nn_finder, nb_quantized, prior_factor,))
                    time.sleep(0.01)

                    thread.start()
                    processes.append(thread)

            # try to consume all processes
            while True:

                processes = [process for process in processes if process.is_alive()]

                if len(processes) < self._max_processes:
                    start_process()

                yield self._queue.get()
        except Exception as exc:

            for process in processes:
                process.terminate()
            self._queue.close()

            raise Exception("Error when generating mini-batches: {}".format(exc))

    def generate_batch_in_memory(self, x, nn_finder, nb_q, prior_factor):
        """
        Generate batch in memory
        :param x: 
        :param nn_finder: 
        :param nb_q: 
        :param prior_factor: 
        :return: 
        """

        while True:
            random_idx = np.random.choice(x.shape[0], self._batch_size, replace=False)

            x_batch_color = x[random_idx]
            x_batch_black = x_batch_color[:, :1, :, :]
            x_batch_ab = x_batch_color[:, 1:, :, :]

            npts, channels, height, width = x_batch_ab.shape
            x_a = np.ravel(x_batch_ab[:, 0, :, :])
            x_b = np.ravel(x_batch_ab[:, 1, :, :])
            x_batch_ab = np.vstack((x_a, x_b)).T

            y_batch = self.get_soft_encoding(x_batch_ab, nn_finder, nb_q)

            max_idx = np.argmax(y_batch, axis=1)
            weights = prior_factor[max_idx].reshape(y_batch.shape[0], 1)

            y_batch = np.concatenate((y_batch, weights), axis=1)

            # Reshape y_batch
            y_batch = y_batch.reshape((npts, height, width, nb_q + 1))

            yield x_batch_black, x_batch_color, y_batch

    @staticmethod
    def generate_cnn_input(cnn_model, image_path):
        """
        Generate the input for CNN model
        :param cnn_model:
        :param image_path:
        :return:
        """
        try:
            # Get input shape of CNN
            height_in, width_in = cnn_model.input_shape[2:]
            height_out, width_out = cnn_model.output_shape[1:-1]

            # Load the original image
            image_rgb = caffe.io.load_image(image_path)

            # Convert image to Lab color space
            image_lab = color.rgb2lab(image_rgb)

            # Pull out L channel
            image_l = image_lab[:, :, 0]

            # Get original image size
            height_original, width_original = image_rgb.shape[:2]

            # Create gray scale version of image (just for displaying)
            image_lab_bw = image_lab.copy()
            image_lab_bw[:, :, 1:] = 0

            # Resize image to network input size
            image_rs = caffe.io.resize_image(image_rgb, (height_in, width_in))
            image_lab_rs = color.rgb2lab(image_rs)
            image_l_rs = image_lab_rs[:, :, 0]

            image_l_rs = image_l_rs.reshape((1, height_in, width_in, 1))
            image_l_rs = image_l_rs.transpose(0, 3, 1, 2)

            # Create the input for CNN (just the image_l_rs will be used in CNN)
            cnn_input = (image_l_rs, image_l, (height_original, width_original), (height_out, width_out))

            return cnn_input

        except Exception as exc:
            print(exc)

    def __getstate__(self):
        """
        Get process state
        :return:
        """
        self_dict = self.__dict__.copy()
        return self_dict

    def __setstate__(self, state):
        """
        Set process state
        :param state: 
        :return: 
        """
        self.__dict__.update(state)
