"""
Handles data loading and batching.
"""

from tensorflow.data import Dataset

class ImageDataLoader(object):

  def __init__(self, images_dir):
    self.root_images_dir = images_dir
  #   self._train = Dataset()
  #   self._test = Dataset()

  #   self.init_train_data()
  #   self.init_test_data()

  # def get_train_dataset(self):
  #   return self._train

  # def get_test_dataset(self):
  #   return self._test

  # def init_train_data(self):
  #   # Need to load up all the images into the dataset so that it can be returned.
  #   # One hot vectors identifying each image.
  #   # Need to get the colored image and the black/white image.
  #   # To follow dataset pattern, load all data, then call dataset.batch(), then dataset.repeat()
  #   pass
  #   file_idx = training_images_index[start_idx + i]
  #   filename = train_filenames[file_idx]
  #   label = one_hot_from_filename(filename)
  #   class_labels[i] = label
  #   assert(np.linalg.norm(class_labels[i]) == 1)
  #   im = read_scaled_color_image_Lab(train_dir + filename)
  #   im_bw = im[:,:,0].reshape((-1,IMAGE_SIZE,IMAGE_SIZE,1))
  #   im_c = im[:,:,1:].reshape((-1,IMAGE_SIZE,IMAGE_SIZE,2))
  #   bw_images[i] = im_bw
  #   color_features[i] = im_c

  # def init_test_data(self):
  #   pass



