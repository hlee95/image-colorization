import math
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential # Model.Sequential, then model.fit()

from model import ImageColorization
from data_loader import ImageDataLoader

IMAGES_DIR = "fake"

DEPTH = 64 # Used to parameterize the depth of each output layer.
NUM_TRAIN_IMAGES = 100000 # True value is 100000, change to train on fewer.
NUM_CLASSES = 100

NUM_EPOCHS = 1
LR = 1e-5


def train_model_one_epoch(data_loader, weights, sess):
  num_batches = NUM_TRAIN_IMAGES / BATCH_SIZE
  for batch in xrange(num_batches):
    bw_images, color_features, class_labels = data_loader.get_next_training_batch(BATCH_SIZE)
    y_downsampled = tf.image.resize_nearest_neighbor(color_features, [IMAGE_SIZE/2, IMAGE_SIZE/2]).eval(session=sess)
    feed_dict = {
      train_data_node: bw_images,
      train_colors_node: y_downsampled,
      train_class_node: class_labels
    }
    run_model_one_batch(feed_dict, weights, sess, train=True)

def main():
  data_loader = ImageDataLoader(IMAGES_DIR)
  model = Sequential()
  model.add(layers.Dense(64, activation="relu"))
  model.add(layers.Dense(10, activation="softmax"))
  model.compile(optimizer=tf.train.AdamOptimizer(LR),
                loss="categorical_crossentropy",
                metrics=["accuracy"])
  data = np.random.random((1000, 32))
  labels = np.random.random((1000, 10))
  model.fit(data, labels, epochs=10, batch_size=32)

  val_data = np.random.random((100, 32))
  val_labels = np.random.random((100, 10))
  model.fit(data, labels, epochs=10, batch_size=32, validation_data=(val_data, val_labels))

  result = model.predict(data, batch_size=32)
  print(result.shape)


if __name__ == "__main__":
  # Parse arguments (such as the image dir?)

  main()

  print "\nDone."