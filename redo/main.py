import math
import numpy as np
import tensorflow as tf

from model import ImageColorization
from model import color_classify_loss
from data_loader import ImageDataLoader
from constants import *

def main():
  # TODO: create a tf session.
  image_util = ImageUtil()
  dataloader = ImageDataLoader(IMAGES_DIR, image_util, sess)

  train_dataset = dataloader.get_train_datset()
  val_dataset = dataloader.get_val_dataset()
  test_dataset = dataloader.get_test_dataset()

  model = ImageColorization()
  model.compile(optimizer=tf.train.AdamOptimizer(LR),
                loss=color_classify_loss)

  # Train. TODO: put this in a loop.
  model.fit(train_dataset, epochs=NUM_EPOCHS, steps_per_epoch=30)
  # Train with val data (need this?)
  model.fit(train_dataset, epochs=NUM_EPOCHS, steps_per_epoch=30, validation_data=val_dataset)

  # Test it out!
  results = model.predict(test_dataset)
  print(results.shape)
  # Need to upsample the results and only take the images part not the classified labels part.
  upsampled_results = tf.image.resize_nearest_neighbor(...)
  image_util.save_result_images(upsampled_results[only take the image part])


if __name__ == "__main__":
  # Parse arguments (such as the image dir?)

  main()

  print "\nDone."
