# coding=utf-8
"""
    ???
"""

from . import train_model
from . import evaluate_model


class CNN(object):
    """
    Convolutional Neural Network class
    From here you can train or evaluate a CNN model
    """

    def __init__(self, mode=None, hdf5_data_file=None, model_name='vgg_model', training_mode='in_memory', batch_size=32, n_batch_per_epoch=150,
                 nb_epoch=10000, nb_resblocks=3, nb_neighbors=10, load_weight_epoch=-1, t_parameter=0.38, optimizer='adam', path_to_gs_img=None,
                 root_dir=None):
        """
        Constructor
        :param mode: Choose the mode: train or eval
        :param hdf5_data_file: Path to HDF5 containing the data
        :param model_name: Model name. Choose simple_colorful or colorful_model
        :param training_mode: Choose in_memory to load all the data in memory and train. Choose on_demand to load batches from disk at each step
        :param batch_size: Batch size
        :param n_batch_per_epoch: Number of iterations per epoch
        :param nb_epoch: Number of total epochs
        :param nb_resblocks: Number of residual blocks for simple model
        :param nb_neighbors: Number of nearest neighbors for soft encoding
        :param load_weight_epoch: Epoch at which weights were saved for evaluation
        :param t_parameter: Temperature to change color balance in evaluation phase. If T = 1: desaturated. If T~0 vivid
        :param optimizer: What optimizer do you want to use
                          Adam will run with next parameters: (lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
                          Adadelta will run with next parameters: (lr=1.0, rho=0.95, epsilon=1e-08, decay=0.0)
                          RMSProp will run with next parameters: (lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
        :param path_to_gs_img: The path to gray scale image that will be colorized
        """
        self.__can_run = False
        self._model = None

        assert mode in ["train", "eval"], "\n\n No mode selected. Chose train or eval."
        assert training_mode in ["in_memory", "on_demand"], "\n\n Chose training_mode between 'in_memory' and 'on_demand'."
        assert model_name in ["colorful_model", "vgg_model"], "\n\n Chose training_mode between 'colorful_model' and 'vgg_model'."
        assert optimizer in ["adam", "adadelta", "rms_prop"], "\n\n Chose optimizer between 'adam', 'adadelta' and 'rms_prop'."

        self.__parameters = {"data_file": hdf5_data_file,
                             "batch_size": batch_size,
                             "n_batch_per_epoch": n_batch_per_epoch,
                             "nb_epoch": nb_epoch,
                             "nb_resblocks": nb_resblocks,
                             "training_mode": training_mode,
                             "model_name": model_name,
                             "nb_neighbors": nb_neighbors,
                             "epoch": load_weight_epoch,
                             "T": t_parameter,
                             "optimizer": optimizer,
                             "gray_scale_image": path_to_gs_img,
                             "root_dir": root_dir
                             }

        self.__can_run = True
        self.__mode = mode
        self.__new_image = None

    @property
    def new_image(self):
        """
        Get path to the new image
        :return:
        """
        return self.__new_image

    def __launch_training(self, **kwargs):
        """
        Launch training mode
        :param kwargs:
        :return:
        """
        if self.__can_run:
            self._model = train_model.TrainModel(**kwargs)
            self._model.train_model()
        else:
            raise Exception("Can not launch train mode, because parameters are not set.")

    def __launch_eval(self, **kwargs):
        """
        Launch evaluation mode
        :param kwargs: Parameters
        :return:
        """
        if self.__can_run:
            self._model = evaluate_model.EvaluateModel(**kwargs)
            self.__new_image = self._model.evaluate_model()
        else:
            raise Exception("Can not launch evaluation mode, because parameters are not set.")

    def run_cnn(self):
        """
        Run the CNN
        :return:
        """

        if self.__mode == 'train':
            self.__launch_training(**self.__parameters)

        if self.__mode == 'eval':
            self.__launch_eval(**self.__parameters)

# if __name__ == "__main__":
#     # cnn = CNN('eval', '../../data/processed/HDF5_data_64.h5', nb_neighbors=5, load_weight_epoch=3936,
#     #           path_to_gs_img=r'C:\Users\Julian\Desktop\testare poze\Mona_Lisa_GS2.jpg', batch_size=1)
#
#     cnn = CNN('train', '../../data/processed/HDF5_data_64.h5', nb_neighbors=5, load_weight_epoch=3936, n_batch_per_epoch=200)
#     cnn.run_cnn()
