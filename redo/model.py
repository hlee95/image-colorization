"""
The main image colorization model.
"""

import math
from tensorflow import TensorShape
from tensorflow.keras import Model
from tensorflow.keras import layers

class ImageColorization(Model):
  def __init__(self, constants):
    super(ImageColorization).__init__(name="image_colorization")

    # Save the constants that were passed in.
    self.c = constants

    # Define all layers.
    self.ll1 = layers.Conv2D(self.c.depth, self.c.conv_kernel_size, strides=(2,2), input_shape=(self.c.image_size, self.c.image_size, 1))
    self.ll2 = layers.Conv2D(2 * self.c.depth, self.c.conv_kernel_size, strides=(1,1))

  def call(self, inputs):
    # Forward pass, use layers.
    # The only complicated part is the fusion layer.
    low_level_features = self.ll1(inputs)
    # ...
    mid_level_features = self.ml1(low_level_features)
    global_features = self.g1(low_level_features)
    fusion = None # Reshape mid_level_featureus and global_features together.

    output = self.c1(fusion)
    upsample = None # Upsample result of c2.
    output = self.c3(upsample)
    output = self.c4(output)
    return output

  def compute_output_shape(self, input_shape):
    # Return a TensorShape representing the output shape.
    pass


class ImageColorizationConstants(object):
  def __init__(self):
    self.depth = 64
    self.image_size = 128
    self.conv_kernel_size = 3


