import os

import geojson
import rasterio

import numpy as np

from shapely.geometry import shape, Point

def one_hot_2_class(array, num_classes):
    # Get the shape of the input array
    shape = array.shape

    # Create an empty 3D array with the required shape
    one_hot_array = np.zeros((shape[0], shape[1], num_classes))

    # Fill the one-hot array
    for i in range(shape[0]):
        for j in range(shape[1]):
            class_index = array[i, j]
            one_hot_array[i, j, class_index] = 1

    return one_hot_array


def tile_to_rgb(tile):
    rgb = tile[[4, 2, 1], :, :]
    rgb[rgb >= 2000] = 2000
    rgb = rgb / 2000.0
    rgb = np.rollaxis(rgb, 0, 3)

    return rgb

def label_from_3band(path, label_type="mask"):
    if label_type not in ("geojson", "mask"):
        raise
    # Get the label folder
    folder = os.path.dirname(path)
    up = "/".join(path.split("/")[0:-2])
    label_path = os.path.join(up, label_type)

    # Get the filename
    filename = os.path.basename(path)
    keyname = filename.split(".")[0]

    if label_type == "geojson":
        label_name = keyname.replace("8band_", "Geo_") + ".geojson"
    else:
        label_name = keyname + "_mask.png"

    # Combine them
    return os.path.join(label_path, label_name)

def read_rgb_tile(path):
    # Read the data
    with rasterio.open(path, "r") as f:
        data = f.read()
        data = np.rollaxis(data, 0, 3)
        transform = f.transform

    return data, transform
