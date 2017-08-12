# coding=utf-8
"""
    Utils module
"""
import os
import uuid

import cv2
import scipy.misc
import numpy as np
from skimage import color
import scipy.ndimage.interpolation as sni
import matplotlib.pylab as plt


def remove_files(files):
    """
    Remove files from disk
    :param files: A string or a list or o tuple of files that you want to delete them
    :type files: str | tuple | str
    """

    if isinstance(files, (list, tuple)):
        for _file in files:
            if os.path.isfile(os.path.expanduser(_file)):
                os.remove(_file)
    elif isinstance(files, str):
        if os.path.isfile(os.path.expanduser(files)):
            os.remove(files)


def create_directories(dirs):
    """
    Create directory
    :param dirs: A string or a list or a tuple with strings to create directories
    :type dirs: str | list | tuple
    """

    if isinstance(dirs, (list, tuple)):
        for d in dirs:
            if not os.path.exists(os.path.expanduser(d)):
                os.makedirs(d)
    elif isinstance(dirs, str):
        if not os.path.exists(os.path.expanduser(dirs)):
            os.makedirs(dirs)


def setup_logging(model_name):
    """
    Configure were to store experimental data
    :param model_name: 
    :return: 
    """

    model_dir = "../../models"
    # Output path where we store experiment log and weights
    model_dir = os.path.join(model_dir, model_name)

    fig_dir = "../../figures"

    # Create if it does not exist
    create_directories([model_dir, fig_dir])


def plot_batch_on_train(color_model, quantized_ab, x_batch_black, x_batch_color, batch_size, height, width, nb_q, epoch):
    """
    Plot the image from a batch on train mode
    :param color_model:
    :param quantized_ab:
    :param x_batch_black:
    :param x_batch_color:
    :param batch_size:
    :param height:
    :param width:
    :param nb_q:
    :param epoch:
    :return:
    """

    # Format x_colorized
    x_colorized = color_model.predict(x_batch_black / 100.)[:, :, :, :-1]
    x_colorized = x_colorized.reshape((batch_size * height * width, nb_q))
    x_colorized = quantized_ab[np.argmax(x_colorized, 1)]

    x_a = x_colorized[:, 0].reshape((batch_size, 1, height, width))
    x_b = x_colorized[:, 1].reshape((batch_size, 1, height, width))

    x_colorized = np.concatenate((x_batch_black, x_a, x_b), axis=1).transpose(0, 2, 3, 1)
    x_colorized = [np.expand_dims(color.lab2rgb(im), 0) for im in x_colorized]
    x_colorized = np.concatenate(x_colorized, 0).transpose(0, 3, 1, 2)

    x_batch_color = [np.expand_dims(color.lab2rgb(im.transpose(1, 2, 0)), 0) for im in x_batch_color]
    x_batch_color = np.concatenate(x_batch_color, 0).transpose(0, 3, 1, 2)

    list_img = []
    for i, img in enumerate(x_colorized[:min(32, batch_size)]):  # 32
        # noinspection PyTypeChecker
        arr = np.concatenate([x_batch_color[i], np.repeat(x_batch_black[i] / 100., 3, axis=0), img], axis=2)
        list_img.append(arr)

    list_img = [np.concatenate(list_img[4 * i: 4 * (i + 1)], axis=2) for i in range(int(len(list_img) / 4))]
    arr = np.concatenate(list_img, axis=1)

    img = arr.transpose(1, 2, 0)
    scipy.misc.imsave("../../figures/fig_epoch_%s.png" % epoch, img)


def plot_batch_on_eval(color_model, quantized_ab, x_batch_black, x_batch_color, batch_size, height, width, nb_q, t_parameter):
    """
    Plot the image from a batch in evaluation state
    :param color_model: The model do you want to plot
    :param quantized_ab:
    :param x_batch_black: 
    :param x_batch_color: 
    :param batch_size: 
    :param height: 
    :param width: 
    :param nb_q: 
    :param t_parameter: 
    :return: 
    """

    # Format X_colorized
    x_colorized = color_model.predict(x_batch_black / 100.)[:, :, :, :-1]
    x_colorized = x_colorized.reshape((batch_size * height * width, nb_q))

    # Reweight probabilities
    x_colorized = np.exp(np.log(x_colorized) / t_parameter)
    x_colorized = x_colorized / np.sum(x_colorized, 1)[:, np.newaxis]

    # Reweighted
    q_a = quantized_ab[:, 0].reshape((1, 313))
    q_b = quantized_ab[:, 1].reshape((1, 313))

    x_a = np.sum(x_colorized * q_a, 1).reshape((batch_size, 1, height, width))
    x_b = np.sum(x_colorized * q_b, 1).reshape((batch_size, 1, height, width))

    x_colorized = np.concatenate((x_batch_black, x_a, x_b), axis=1).transpose((0, 2, 3, 1))
    x_colorized = [np.expand_dims(color.lab2rgb(im), 0) for im in x_colorized]
    x_colorized = np.concatenate(x_colorized, 0).transpose((0, 3, 1, 2))

    x_batch_color = [np.expand_dims(color.lab2rgb(im.transpose(1, 2, 0)), 0) for im in x_batch_color]
    x_batch_color = np.concatenate(x_batch_color, 0).transpose((0, 3, 1, 2))

    list_img = []
    for i, img in enumerate(x_colorized[:min(32, batch_size)]):  # 32
        # noinspection PyTypeChecker
        arr = np.concatenate([x_batch_color[i], np.repeat(x_batch_black[i] / 100., 3, axis=0), img], axis=2)
        list_img.append(arr)

    list_img = [np.concatenate(list_img[4 * i: 4 * (i + 1)], axis=2) for i in range(int(len(list_img) / 4))]
    arr = np.concatenate(list_img, axis=1)

    file_name = uuid.uuid4()
    img = arr.transpose((1, 2, 0))
    scipy.misc.imsave("../../evaluation/fig_%s.png" % file_name, img)


def colorize_gray_scale_image(color_model, quantized_ab, x_batch_black, batch_size, height, width, nb_q, t_parameter, size_original=None):
    """
    Plot the image from a batch in evaluation state
    :param size_original:
    :param color_model: The model do you want to plot
    :param quantized_ab:
    :param x_batch_black:
    :param batch_size:
    :param height:
    :param width:
    :param nb_q:
    :param t_parameter:
    :return:
    """

    # Format X_colorized
    ab_prediction = color_model.predict(x_batch_black / 100.)[:, :, :, :-1]
    ab_prediction = ab_prediction.reshape((batch_size * height * width, nb_q))

    # Reweight probabilities
    ab_prediction = np.exp(np.log(ab_prediction) / t_parameter)
    ab_prediction = ab_prediction / np.sum(ab_prediction, 1)[:, np.newaxis]

    # Reweighted
    q_a = quantized_ab[:, 0].reshape((1, 313))
    q_b = quantized_ab[:, 1].reshape((1, 313))

    x_a = np.sum(ab_prediction * q_a, 1).reshape((batch_size, 1, height, width))
    x_b = np.sum(ab_prediction * q_b, 1).reshape((batch_size, 1, height, width))

    ab_prediction = np.concatenate((x_batch_black, x_a, x_b), axis=1).transpose((0, 2, 3, 1))

    ab_prediction = [np.expand_dims(color.lab2rgb(im), axis=0) for im in ab_prediction]
    ab_prediction = np.concatenate(ab_prediction, 0).transpose((0, 3, 1, 2))
    list_img = []

    for i, img in enumerate(ab_prediction[:min(1, batch_size)]):  # 32

        # noinspection PyTypeChecker
        arr = np.concatenate([np.repeat(x_batch_black[i] / 100., 3, axis=0), img], axis=2)
        list_img.append(arr)

    arr = np.concatenate(list_img, axis=1)

    file_name = uuid.uuid4()
    img = arr.transpose((1, 2, 0))
    img = sni.zoom(img, (2. * size_original[0] / img.shape[0], 1. * size_original[1] / img.shape[1], 1))

    img = cv2.resize(img, (img.shape[0], img.shape[1]), interpolation=cv2.INTER_AREA)
    scipy.misc.imsave("../../evaluation/fig_%s.png" % file_name, img)


def color_gray_scale_image(color_model, quantized_ab, x_batch_black, batch_size, height, width, nb_q, t_parameter, img_l, size_original=None,
                           size_out=None, root_dir=None):
    """
    Predict colors for a gray-scale image
    :param root_dir:
    :param size_out:
    :param size_original:
    :param img_l:
    :param color_model: The model do you want to plot
    :param quantized_ab:
    :param x_batch_black:
    :param batch_size:
    :param height:
    :param width:
    :param nb_q:
    :param t_parameter:
    :return:
    """

    # Format X_colorized
    ab_prediction = color_model.predict(x_batch_black / 100.)[:, :, :, :-1]
    ab_prediction = ab_prediction.reshape((batch_size * height * width, nb_q))

    # Reweight probabilities
    ab_prediction = np.exp(np.log(ab_prediction) / t_parameter)
    ab_prediction = ab_prediction / np.sum(ab_prediction, 1)[:, np.newaxis]

    # Reweighted
    q_a = quantized_ab[:, 0].reshape((1, 313))
    q_b = quantized_ab[:, 1].reshape((1, 313))

    x_a = np.sum(ab_prediction * q_a, 1).reshape((batch_size, 1, height, width))
    x_b = np.sum(ab_prediction * q_b, 1).reshape((batch_size, 1, height, width))

    img = np.concatenate((x_a, x_b), axis=1).transpose((0, 2, 3, 1))

    img = img[0, :, :, :]  # this is our result
    img = sni.zoom(img, (1. * size_original[0] / size_out[0], 1. * size_original[1] / size_out[1], 1))  # upsample to match size of original image L

    img_lab_out = np.concatenate((img_l[:, :, np.newaxis], img), axis=2)  # concatenate with original image L
    img_lab_out = color.lab2rgb(img_lab_out)  # convert back to rgb
    img_rgb_out = (255 * np.clip(img_lab_out, 0, 1)).astype('uint8')

    file_name = uuid.uuid4()

    final_result = '/media/colorized/image_%s.png' % file_name

    file_name = os.path.join(root_dir, 'media/colorized/image_%s.png' % file_name)
    plt.imsave(file_name, img_rgb_out)

    return final_result
