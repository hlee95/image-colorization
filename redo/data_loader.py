"""
Handles data loading and batching.
"""

from tensorflow.data import Dataset
from constants import *
import os

class ImageDataLoader(object):

  def __init__(self, images_dir, color_helper, sess):
    self.root_images_dir = images_dir
    self.image_util = image_util
    self.sess = sess

  def get_train_dataset(self):
    """
    Returns a dataset that holds all of the training data.
    """
    bw_images = np.zeros([NUM_TRAINING_IMAGES, IMAGE_SIZE, IMAGE_SIZE, 1])
    color_features = np.zeros([NUM_TRAINING_IMAGES, IMAGE_SIZE, IMAGE_SIZE, 2])
    class_labels = np.zeros([NUM_TRAINING_IMAGES, NUM_CLASSES])
    train_dir = self.root_images_dir + "train/"
    filenames = os.listdir(train_dir)
    # Randomize order that they are read.
    training_images_index = np.random.permutation(NUM_TRAIN_IMAGES)
    for i in xrange(NUM_TRAINING_IMAGES):
      file_index = training_images_index[file_index]
      filename = filenames[i]
      label = self.image_util.class_one_hot_from_filename(filename)
      class_labels[j] = label
      im = self.image_util.read_scaled_color_image_LAB(filename)
      im_bw = im[:,:,0].reshape((-1, IMAGE_SIZE, IMAGE_SIZE, 1))
      im_color = im[:,:,1:].reshape((-1, IMAGE_SIZE, IMAGE_SIZE, 2))
      bw_images[i] = im_bw
      color_features[i] = im_color
    # Downsize the color features so that they are the same size as expected output of
    # the last layer of our model.
    color_downsampled = tf.image.resize_nearest_neighbor(color_features, [IMAGE_SIZE/2, IMAGE_SIZE/2]).eval(session=self.sess)
    # TODO: will need to combine the color and classify sets so that they can be passed as one tensor
    # into the loss function.
    return Dataset.from_tensor_slices(("bw": bw_images, "color": color_downsampled, "class": class_labels))

  def get_val_dataset(self):
    return self._get_eval_dataset_helper("val/", NUM_VAL_IMAGES)

  def get_test_dataset(self):
    return self._get_eval_dataset_helper("test/", NUM_TEST_IMAGES)

  def _get_eval_dataset_helper(self, dirpath, num_images):
    val_bw = np.zeros([num_images, IMAGE_SIZE, IMAGE_SIZE, 1]) # TODO: need to set dtype=np.float32?
    val_color = np.zeros([num_images, IMAGE_SIZE, IMAGE_SIZE, 2])

    val_dir = self.root_images_dir + dirpath
    filenames = os.listdir(val_dir)
    for i in xrange(num_images):
      im = self.image_util.read_scaled_color_image_LAB(filenames[i])
      im_bw = im[:,:,0].reshape((-1, IMAGE_SIZE, IMAGE_SIZE, 1))
      im_color = im[:,:,1:].reshape((-1, IMAGE_SIZE, IMAGE_SIZE, 2))
      val_bw[i] = im_bw
      val_color[i] = im_color
    return Dataset.from_tensor_slices((im_bw, im_color))

