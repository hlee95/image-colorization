from constants import *

class ImageUtil(object):
  """
  This class is a utility class for helper functions that operate on
  image data or image files.
  """

  def __init__(self):
    pass

  def read_scaled_color_image_LAB(self, filepath):
    """
    Returns the image at filepath represented in LAB color scheme.
    """
    pass

  def class_one_hot_from_filename(self, filename):
    """
    Returns a one-hot vector representing the class label of the image
    with the given filename.

    The size of the vector will be equal to the constant NUM_CLASSES.
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
