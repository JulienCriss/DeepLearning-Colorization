# coding=utf-8
"""
    Upload an image
"""
import os

from django.shortcuts import render
from django.core.files.storage import FileSystemStorage

from .colorare_automata.src.model.colorize_cnn import CNN

DATA_DIR = os.path.dirname(os.path.dirname(__file__))


def simple_upload(request):
    """
    Index upload Image
    :param request:
    :return:
    """
    if request.method == 'POST' and request.FILES['userImage']:
        image_file = request.FILES['userImage']
        fs = FileSystemStorage()
        filename = fs.save(image_file.name, image_file)
        uploaded_file_url = fs.url(filename)

        hdf5_file = os.path.join(DATA_DIR, 'uploader/colorare_automata/data/processed/HDF5_data_64.h5')

        image_gs = os.path.join(DATA_DIR, uploaded_file_url[1:])
        cnn = CNN('eval', hdf5_file, nb_neighbors=5, load_weight_epoch=4544,
                  path_to_gs_img=image_gs, batch_size=1, root_dir=DATA_DIR)
        cnn.run_cnn()

        return render(request, 'colorized.html', {'uploaded_file_url': uploaded_file_url, 'colorized_image': cnn.new_image})
    return render(request, 'index.html', {'uploaded_file_url': None})
