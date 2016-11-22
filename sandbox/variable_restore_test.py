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

# Launch model, don't need to initialize variables.
# Restore variables.s
with tf.Session() as sess:
  sess.run(init_op)
  v = sess.run(weights)
  print('Weights before restoring:\n')
  print v
  saver.restore(sess, 'model.ckpt')
  print('Model restored.')
  v = sess.run(weights)
  print('Weights after restoring:')
  print v

