import numpy as np
from skimage import io, color

from constants import *

class ImageUtil(object):
    """
    Utility class for processing image files and data.
    """

    def __init__(self):
        io.use_plugin('matplotlib')

    def read_scaled_Lab_image(self, filepath):
        """
        filepath: full/relative path to image file.
        Returns: Numpy array representing image in Lab color space (https://en.wikipedia.org/wiki/CIELAB_color_space)
        with Lab components all scaled to between 0 and 1.
        """
        rgb = io.imread(filepath)
        lab = color.rgb2lab(rgb)
        # scale so that L, a, and b components are between 0 and 1
        lab[:, :, 0] = lab[:, :, 0] / 100
        lab[:, :, 1:] = np.maximum(np.minimum(lab[:, :, 1:], 100), -100)
        lab[:, :, 1:] = (lab[:, :, 1:] + 100) / 200
        return lab

    def class_one_hot_from_filename(self, filename):
        """
        filename: image file name (without parent directories).
        Returns: one-hot class label vector.
        """
        class_label = int(filename.split("_")[0])
        one_hot = np.zeros(NUM_CLASSES)
        one_hot[class_label] = 1.0
        return one_hot

    def save_result_images(self, images):
        """
        Saves the result images so they can be manually inspected later.
        """
        # Will need to turn from LAB into RGB.
        # Use io.imsave(filename, image).
        pass
