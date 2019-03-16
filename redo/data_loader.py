import os
import numpy as np

from constants import *

class ImageDataLoader(object):
    """
    Handles data loading and batching.
    """

    def __init__(self, root_image_dir, image_util):
        self.root_image_dir = root_image_dir
        self.image_util = image_util

    def get_dataset(self, dataset_type, num_images):
        """
        dataset_type: train/test/val
        Returns: dict containing greyscale image arrays (L component),
        color image array (a/b components), and class labels
        """
        greyscale_images = np.zeros([num_images, IMAGE_SIZE, IMAGE_SIZE, 1])
        color_images = np.zeros([num_images, IMAGE_SIZE, IMAGE_SIZE, 2])
        class_labels = np.zeros([num_images, NUM_CLASSES])
        image_dir = os.path.join(self.root_image_dir, dataset_type)
        filenames = [filename for filename in os.listdir(image_dir) if filename.endswith('.png')]
        # Randomize order that images are read.
        training_images_index = np.random.permutation(num_images)
        for i in xrange(num_images):
            file_index = training_images_index[i]
            filename = filenames[file_index]
            class_labels[i] = self.image_util.class_one_hot_from_filename(filename)
            image = self.image_util.read_scaled_Lab_image(os.path.join(image_dir, filename))
            greyscale_images[i,:,:,:] = image[:, :, 0].reshape((-1, IMAGE_SIZE, IMAGE_SIZE, 1))
            color_images[i,:,:,:] = image[:, :, 1:].reshape((-1, IMAGE_SIZE, IMAGE_SIZE, 2))
        return {"greyscale": greyscale_images, "color": color_images, "class": class_labels}
