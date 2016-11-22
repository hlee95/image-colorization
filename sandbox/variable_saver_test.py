# Test the functionality of Variable Savers.
# Following documentation at:
# https://www.tensorflow.org/versions/r0.11/how_tos/variables/index.html
import tensorflow as tf
import numpy as np

# Create two variables.
weights = tf.Variable(tf.random_normal([5, 2], stddev=0.35), name='weights')
biases = tf.Variable(tf.zeros([2]), name='biases')
# Add an op to initialize the variables.
init_op = tf.initialize_all_variables()
# Add ops to save and restore all the variables.
saver= tf.train.Saver()

# Launch model, initialize variables, do work, save variables to disk.
with tf.Session() as sess:
  sess.run(init_op)
  v = sess.run(weights)
  save_path = saver.save(sess, 'model.ckpt')
  print('Model saved in file: %s') % save_path
  print('Saved weights:')
  print(v)


