"""
The main image colorization model.
"""

import math
import numpy as np
import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras import layers

from constants import *

class ImageColorization(Model):
  def __init__(self):
    super(ImageColorization).__init__(name="image_colorization")

    # Save the constants that were passed in.
    self.c = ImageColorizationConstants()

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
    # TODO: also need to return the classify logis, in addition to the color logits.
    return output

  def compute_output_shape(self, input_shape):
    # Return a tf.TensorShape representing the output shape.
    pass

def color_classify_loss(truth, predictions):
  """
  Computes the loss given the truth and the predictions.
  Both truth and predictions are the concatanation of the color logits
  and the classify logits.

  For colors, the loss is mean squared error.
  For classification, the loss is cross-entropy loss.
  """
  color_truth = truth[...]
  color_predictions = predictions[...]
  color_loss = tf.reduce_sum(tf.square(color_truth - color_predictions))

  classify_truth = truth[...]
  classify_predictions = predictions[...]
  classify_loss = ALPHA * tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(classify_truth, classify_predictions))

  # Combine to form overall loss.
  return color_loss + classify_loss


class ImageColorizationConstants(object):
  def __init__(self):
    self.depth = 64
    self.image_size = 128
    self.conv_kernel_size = 3


