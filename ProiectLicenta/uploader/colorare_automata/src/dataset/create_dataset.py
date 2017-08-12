# coding=utf-8
"""
    Create data entry from images for Neural Network
"""
import os
import cv2
import h5py
import parmap
import numpy as np
import _pickle as pickle

from skimage import color
from tqdm import tqdm as tqdm
import sklearn.neighbors as nn
import matplotlib.pylab as plt
from matplotlib.colors import LogNorm
import matplotlib.gridspec as gridspec
from scipy.interpolate import interp1d
from scipy.signal import gaussian, convolve


class DataSet(object):
    """
    Data set class
    Build data set which is the input for CNN
    Read color images and extract Lab color space, black & white and RGB version of ground truth image
    This data set file, lately will be the input for our ConvNet
    """

    def __init__(self, raw_directory, data_directory, train_data_directory="train_data", img_size=224, make_plot=False, chunk_size=1000,
                 evaluation_list=None):
        """
        Constructor
        :param raw_directory: Where is raw data
        :param data_directory: Where to save processed data
        :param train_data_directory: Where is train data (images)
        :param img_size: The size do you want to resize the images
        :param make_plot: If you want to plot some information when creating the data set
        """
        self.evaluation_list = evaluation_list
        self._raw_dir = raw_directory
        self._data_dir = data_directory
        self._train_data_images = train_data_directory
        self._img_size = img_size
        self._make_plot = make_plot
        self._chunk_size = chunk_size

        self._sigma = 5
        self._gamma = 0.5
        self._alpha = 1

    @staticmethod
    def format_image(image_path, image_size):
        """
        Load a image with opencv library and reshape the original image to a size that user specifies
        :param image_path: Path to image
        :param image_size: Resize the image to a dimension WxH
        :rtype img_path: str
        :rtype image_size: int
        :return: Three images : color, Lab colors and the gray-scale
        """
        color_image = cv2.imread(image_path)
        color_image = color_image[:, :, ::-1]
        black_image = cv2.imread(image_path, 0)

        # resize the images
        color_image = cv2.resize(color_image, (image_size, image_size), interpolation=cv2.INTER_AREA)
        black_image = cv2.resize(black_image, (image_size, image_size), interpolation=cv2.INTER_AREA)

        # get the Lab color
        lab_image = color.rgb2lab(color_image)

        # reshape the images
        lab_image = lab_image.reshape((1, image_size, image_size, 3)).transpose(0, 3, 1, 2)
        color_image = color_image.reshape((1, image_size, image_size, 3)).transpose(0, 3, 1, 2)
        black_image = black_image.reshape((1, image_size, image_size, 1)).transpose(0, 3, 1, 2)

        return color_image, lab_image, black_image

    def build_hdf5_file(self):
        """
        Gather the data in a single HDF5 file which will be the input data for CNN.
        Read evaluation file, build it if it does not exist
        In evaluation status
            - "0" represents training image
            - "1" represents validation image
            - "2" represents testing image
        """
        print(" -- Build HDF5 file ...")
        data_partitions = {}

        with open(os.path.join(self._raw_dir, "eval/{}".format(self.evaluation_list))) as file_handler:
            lines = file_handler.readlines()

            for line in lines:
                line = line.rstrip().split()
                _image = line[0]
                data_set_type = int(line[1])
                data_partitions[_image] = data_set_type

        with open(os.path.join(self._data_dir, "data_partition.pickle"), "wb") as file_handler:
            pickle.dump(data_partitions, file_handler)

        hdf5_file = os.path.join(self._data_dir, "HDF5_data_{}.h5".format(self._img_size))
        with h5py.File(hdf5_file, "w") as hfw_handler:

            for data_set_idx, data_set_type, in enumerate(["training", "validation", "test"]):
                images_list = []

                for _image in data_partitions.keys():
                    if data_partitions[_image] == data_set_idx:
                        images_list.append(os.path.join(self._raw_dir, self._train_data_images, _image))

                images_list.sort()
                images_list = np.array(images_list)

                data_color = hfw_handler.create_dataset("{}_color_data".format(data_set_type), (0, 3, self._img_size, self._img_size),
                                                        maxshape=(None, 3, self._img_size, self._img_size), dtype=np.uint8)

                data_lab = hfw_handler.create_dataset("{}_lab_data".format(data_set_type), (0, 3, self._img_size, self._img_size),
                                                      maxshape=(None, 3, self._img_size, self._img_size), dtype=np.float64)

                data_black = hfw_handler.create_dataset("{}_black_data".format(data_set_type), (0, 1, self._img_size, self._img_size),
                                                        maxshape=(None, 1, self._img_size, self._img_size), dtype=np.uint8)

                num_of_images = len(images_list)
                num_chunks = int(num_of_images / self._chunk_size) if int(num_of_images / self._chunk_size) > 0 else 10
                arr_chunks = np.array_split(np.arange(num_of_images), num_chunks)

                print("\t\t Creating data set {}    Number of chunks arrays: {}".format(data_set_type, len(arr_chunks)))

                for chunk_idx in tqdm(arr_chunks):
                    list_img_path = images_list[chunk_idx].tolist()
                    list_img_path.sort()

                    outputs = parmap.map(self.format_image, list_img_path, self._img_size, parallel=True)

                    arr_img_color = np.vstack([_output[0] for _output in outputs if _output[0].shape[0] > 0])
                    arr_img_lab = np.vstack([_output[1] for _output in outputs if _output[0].shape[0] > 0])
                    arr_img_black = np.vstack([_output[2] for _output in outputs if _output[0].shape[0] > 0])

                    # Resize HDF5 data set
                    data_color.resize(data_color.shape[0] + arr_img_color.shape[0], axis=0)
                    data_lab.resize(data_lab.shape[0] + arr_img_lab.shape[0], axis=0)
                    data_black.resize(data_black.shape[0] + arr_img_black.shape[0], axis=0)

                    data_color[-arr_img_color.shape[0]:] = arr_img_color.astype(np.uint8)
                    data_lab[-arr_img_lab.shape[0]:] = arr_img_lab.astype(np.float64)
                    data_black[-arr_img_black.shape[0]:] = arr_img_black.astype(np.uint8)

    def compute_color_priority(self):
        """
        Compute the color priority
        :return: 
        """
        print(" -- Compute color priority ...")
        # Load the gamut points location
        quantized_ab = np.load(os.path.join(self._data_dir, "pts_in_hull.npy"))

        if self._make_plot:
            plt.figure(figsize=(15, 15))
            grid_spec = gridspec.GridSpec(1, 1)
            ax = plt.subplot(grid_spec[0])

            for i in range(quantized_ab.shape[0]):
                ax.scatter(quantized_ab[:, 0], quantized_ab[:, 1])
                ax.annotate(str(i), (quantized_ab[i, 0], quantized_ab[i, 1]), fontsize=6)
                ax.set_xlim([-110, 110])
                ax.set_ylim([-110, 110])

        # Compute the color priority over a subset of the training set
        with h5py.File(os.path.join(self._data_dir, "HDF5_data_{}.h5".format(self._img_size)), "a") as hf_handler:

            x_ab = hf_handler["training_lab_data"][:100000][:, 1:, :, :]

            x_a = np.ravel(x_ab[:, 0, :, :])
            x_b = np.ravel(x_ab[:, 1, :, :])
            x_ab = np.vstack((x_a, x_b)).T

            if self._make_plot:
                plt.hist2d(x_ab[:, 0], x_ab[:, 1], bins=100, norm=LogNorm())
                plt.xlim([-110, 110])
                plt.ylim([-110, 110])
                plt.colorbar()
                plt.show()
                plt.clf()
                plt.close()

            # Create nearest neighbor instance with index = quantized_ab
            nearest_neighbor = 1
            nearest = nn.NearestNeighbors(n_neighbors=nearest_neighbor, algorithm="ball_tree").fit(quantized_ab)

            # Find index of nearest neighbor for x_ab
            distance, index = nearest.kneighbors(x_ab)

            # We now count the number of occurrences of each color
            index = np.ravel(index)
            counts = np.bincount(index)
            indexes = np.nonzero(counts)[0]
            priority_probability = np.zeros((quantized_ab.shape[0]))

            for i in range(quantized_ab.shape[0]):
                priority_probability[indexes] = counts[indexes]

            # We turn this into a color probability
            priority_probability = priority_probability / (1.0 * np.sum(priority_probability))

            # Save the probability
            np.save(os.path.join(self._data_dir, "DataSet_prior_probability_{}.npy".format(self._img_size)), priority_probability)

            if self._make_plot:
                plt.hist(priority_probability, bins=100)
                plt.yscale("log")
                plt.show()

    def smooth_color_priority(self):
        """
        Smooth the color prior
        :return: 
        """
        print(" -- Smooth color priority ...")

        priority_probability = np.load(os.path.join(self._data_dir, "DataSet_prior_probability_{}.npy".format(self._img_size)))

        # Add an epsilon to prior prob to avoid 0 values and possible NaN
        priority_probability += 1e-3 * np.min(priority_probability)

        # Re-normalize
        priority_probability = priority_probability / (1.0 * np.sum(priority_probability))

        # Smooth with gaussian
        f = interp1d(np.arange(priority_probability.shape[0]), priority_probability)
        xx = np.linspace(0, priority_probability.shape[0] - 1, 1000)
        yy = f(xx)

        window = gaussian(2000, self._sigma)
        smoothed = convolve(yy, window / window.sum(), mode='same')
        fout = interp1d(xx, smoothed)

        prior_probability_smoothed = np.array([fout(i) for i in range(priority_probability.shape[0])])
        prior_probability_smoothed = prior_probability_smoothed / np.sum(prior_probability_smoothed)

        # Save on disk
        file_name = os.path.join(self._data_dir, "DataSet_prior_probability_smoothed_{}.npy".format(self._img_size))
        np.save(file_name, prior_probability_smoothed)

        if self._make_plot:
            plt.plot(priority_probability)
            plt.plot(prior_probability_smoothed, "g--")
            plt.plot(xx, smoothed, "r-")
            plt.yscale("log")
            plt.show()

    def compute_priority_factor(self):
        """
        Compute the priority factor
        :return: 
        """
        print(" -- Compute priority factor ...")

        file_name = os.path.join(self._data_dir, "DataSet_prior_probability_smoothed_{}.npy".format(self._img_size))
        prior_prob_smoothed = np.load(file_name)

        unique = np.ones_like(prior_prob_smoothed)

        # noinspection PyTypeChecker
        unique /= np.sum(1.0 * unique)

        prior_factor = (1 - self._gamma) * prior_prob_smoothed + self._gamma * unique
        prior_factor = np.power(prior_factor, -self._alpha)

        # Re-normalize
        prior_factor /= np.sum(prior_factor * prior_prob_smoothed)

        # Save on disk
        file_name = os.path.join(self._data_dir, "DataSet_prior_factor_{}.npy".format(self._img_size))
        np.save(file_name, prior_factor)

        if self._make_plot:
            plt.plot(prior_factor)
            plt.yscale("log")
            plt.show()

    def check_hdf5_file(self):
        """
        Plot images with landmarks to check the processing
        :return: 
        """
        hdf5_file = os.path.join(self._data_dir, "HDF5_data_{}.h5".format(self._img_size))
        with h5py.File(hdf5_file, "r") as hf:
            data_color = hf["training_color_data"]
            data_lab = hf["training_lab_data"]
            data_black = hf["training_black_data"]

            for i in range(data_color.shape[0]):
                fig = plt.figure()
                grid_spec = gridspec.GridSpec(3, 1)

                for k in range(3):
                    ax = plt.subplot(grid_spec[k])

                    if k == 0:
                        img = data_color[i, :, :, :].transpose(1, 2, 0)
                        ax.imshow(img)

                    elif k == 1:
                        img = data_lab[i, :, :, :].transpose(1, 2, 0)
                        img = color.lab2rgb(img)
                        ax.imshow(img)

                    elif k == 2:
                        img = data_black[i, 0, :, :] / 255.
                        ax.imshow(img, cmap="gray")
                grid_spec.tight_layout(fig)
                plt.show()
                plt.clf()
                plt.close()


if __name__ == '__main__':

    raw_dir = "../../data/raw"
    data_dir = "D:\\processed"
    train_data = "D:\\train2"  # "../../data/raw/train/img_align_celeba"

    temp_obj = DataSet(raw_dir, data_dir, train_data, img_size=64, evaluation_list='list_eval_partition_3.txt', make_plot=True)

    for d in [raw_dir, data_dir]:
        if not os.path.exists(d):
            os.makedirs(d)

    temp_obj.build_hdf5_file()
    temp_obj.compute_color_priority()
    temp_obj.smooth_color_priority()
    temp_obj.compute_priority_factor()
