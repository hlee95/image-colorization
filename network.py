import tensorflow as tf
import math
import os
import sys
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
from scipy import misc
from skimage import io, color

def color_small():
	'''
	Small test network
	'''
	x = tf.placeholder(tf.float32, [None, 64*64])
	x = tf.reshape(x, [-1,64,64,1])

	W = tf.Variable(tf.truncated_normal(
									[5, 5, 1, 3], stddev=0.1))
	b = tf.Variable(tf.zeros([3]))

	level = tf.nn.conv2d(x, W, [1, 1, 1, 1], padding='SAME')
	level = tf.nn.relu(level + b)
	y_ = tf.placeholder(tf.float32, [None, 64, 64, 3])

	loss = tf.reduce_mean(tf.square(level-y_))
	train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
	init = tf.initialize_all_variables()
	sess = tf.Session()
	sess.run(init)
	sailboat_c = tf.image.decode_png(tf.read_file('sailboat_c.png'),channels=3).eval(session=sess)
	sailboat_c = sailboat_c.reshape((-1,64,64,3))
	sailboat_bw = tf.image.decode_png(tf.read_file('sailboat_bw.png'),channels=1).eval(session=sess)
	sailboat_bw = sailboat_bw.reshape((-1,64,64,1))
	lake_c = tf.image.decode_png(tf.read_file('lake_c.png'),channels=3).eval(session=sess)
	lake_c = sailboat_c.reshape((-1,64,64,3))
	lake_bw = tf.image.decode_png(tf.read_file('lake_bw.png'),channels=1).eval(session=sess)
	lake_bw = sailboat_bw.reshape((-1,64,64,1))
	print lake_bw.shape
	x_train = np.stack((sailboat_bw,lake_bw),-1)
	y_train = np.stack((sailboat_c,lake_c),-1)
	sess.run(train_step, feed_dict={x: sailboat_bw, y_: sailboat_c})
	#sess.close()
	#correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
	#accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	#print(sess.run(accuracy, feed_dict={x: mnist.test.images.reshape(-1,28,28,1), y_: mnist.test.labels}))

SEED = 66478 				# Set to None for random seed.
NUM_TRAIN_IMAGES = 100000 		# Should be 100,000 for actual dataset.
NUM_TEST_IMAGES = 100 		# Should be 10,000 for actual dataset.
NUM_VAL_IMAGES = 100			# Should be 10,000 for actual dataset.
BATCH_SIZE = 128 			# Should be 128 for actual dataset.
EVAL_BATCH_SIZE = 128
EVAL_FREQUENCY = 100			# Subject to change...
IMAGE_SIZE = 128
DEPTH = 64 				# Used to parameterize the depth of each output layer.
FINAL_DEPTH = 2 			# The final depth should always be 2.
epsilon = 1e-3
ALPHA = 1.0/300				# Weight of classification loss
NUM_CLASSES = 100			# Number of classes for classification
IMAGES_DIR = 'data/images/' 		# Relative or absolute path to directory where images are.
                 			# IMAGES_DIR should have 3 subdirectories: train, val, test
# Set up random order to access training images.
TRAINING_IMAGES_INDEX = np.random.permutation(NUM_TRAIN_IMAGES)

def read_scaled_color_image_Lab(filename):
	# Read image, cut off alpha channel, only keep rgb.
	rgb = io.imread(filename)[:,:,:3]
	# Resize to 128x128 (temporary until we ensure inputs are 128x128)
	rgb = misc.imresize(rgb, (128,128))
	lab = color.rgb2lab(rgb).astype(np.float32)
	# rescale a and b so that they are in the range (0,1) of the sigmoid function
	a_min = np.min(lab[:,:,1])-1.0
	a_max = np.max(lab[:,:,1])+1.0
	b_min = np.min(lab[:,:,2])-1.0
	b_max = np.max(lab[:,:,2])+1.0
	lab[:,:,1] = (lab[:,:,1]-a_min)/(a_max-a_min)
	lab[:,:,2] = (lab[:,:,2]-b_min)/(b_max-b_min)
	return lab

# Returns a one-hot vector given an image filename.
# Used to get labels for the classification network.
def one_hot_from_filename(filename):
	pos = int(filename.split('_')[0])
        one_hot = np.zeros(NUM_CLASSES)
        one_hot[pos] = 1.0
        return one_hot

def batch_norm(inputs, train, axes = 3, decay = 0.999):
    scale = tf.Variable(tf.ones([inputs.get_shape()[-1]]))
    beta = tf.Variable(tf.zeros([inputs.get_shape()[-1]]))
    pop_mean = tf.Variable(tf.zeros([inputs.get_shape()[-1]]), trainable=False)
    pop_var = tf.Variable(tf.ones([inputs.get_shape()[-1]]), trainable=False)

    if train:
        batch_mean, batch_var = tf.nn.moments(inputs,[0])
        if axes == 3:
        	batch_mean, batch_var = tf.nn.moments(inputs,[0,1,2])
        train_mean = tf.assign(pop_mean,
                               pop_mean * decay + batch_mean * (1 - decay))
        train_var = tf.assign(pop_var,
                              pop_var * decay + batch_var * (1 - decay))
        with tf.control_dependencies([train_mean, train_var]):
            return tf.nn.batch_normalization(inputs,
                batch_mean, batch_var, beta, scale, epsilon)
    else:
        return tf.nn.batch_normalization(inputs,
            pop_mean, pop_var, beta, scale, epsilon)

def error_rate(predictions, labels):
  """Return the error rate based on dense predictions and sparse labels."""
  return 100.0 - (
      100.0 *
      np.sum(np.argmax(predictions, 1) == labels) /
      predictions.shape[0])

def main():
	sess = tf.Session()

	# This is where training samples and labels are fed to the graph.
	# These placeholder nodes will be fed a batch of training data at each
	# training step using the {feed_dict} argument to the Run() call below.
	train_data_node = tf.placeholder(
	  	tf.float32,
	  	shape=(BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 1))
	train_colors_node = tf.placeholder(
		tf.float32,
		shape=(BATCH_SIZE, IMAGE_SIZE/2, IMAGE_SIZE/2, 2))
	train_class_node = tf.placeholder(tf.float32, shape=(BATCH_SIZE, NUM_CLASSES))
	eval_data = tf.placeholder(
	  	tf.float32,
	  	shape=(EVAL_BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 1))

	######
	# Low level feature hyperparameters: 4 convolutional layers.
	######

	# Input is IMAGE_SIZE x IMAGE_SIZE x 1
	# Output is IMAGE_SIZE/2 x IMAGE_SIZE/2 x DEPTH
	ll1_filter_size = 3
	ll1_depth = DEPTH
	ll1_stride = 2

	# Output is IMAGE_SIZE/2 x IMAGE_SIZE/2 x 2*DEPTH
	ll2_filter_size = 3
	ll2_depth = 2*DEPTH
	ll2_stride = 1

	# Output is IMAGE_SIZE/4 x IMAGE_SIZE/4 x 2*DEPTH
	ll3_filter_size = 3
	ll3_depth = 2*DEPTH
	ll3_stride = 2

	# Output is IMAGE_SIZE/4 x IMAGE_SIZE/4 x 4*DEPTH
	ll4_filter_size = 3
	ll4_depth = 4*DEPTH
	ll4_stride = 1

	# experiment with different values for the standard deviation
	ll1_weights = tf.Variable(tf.truncated_normal(
									[ll1_filter_size, ll1_filter_size, 1, ll1_depth], stddev=0.1))
	ll1_biases = tf.Variable(tf.zeros([ll1_depth]))
	ll1_feat_map_size = int(math.ceil(float(IMAGE_SIZE) / ll1_stride))

	ll2_weights = tf.Variable(tf.truncated_normal(
									[ll2_filter_size, ll2_filter_size, ll1_depth, ll2_depth], stddev=0.1))
	ll2_biases = tf.Variable(tf.zeros([ll2_depth]))
	ll2_feat_map_size = int(math.ceil(float(ll1_feat_map_size) / ll2_stride))

	ll3_weights = tf.Variable(tf.truncated_normal(
									[ll3_filter_size, ll3_filter_size, ll2_depth, ll3_depth], stddev=0.1))
	ll3_biases = tf.Variable(tf.zeros([ll3_depth]))
	ll3_feat_map_size = int(math.ceil(float(ll2_feat_map_size) / ll3_stride))

	ll4_weights = tf.Variable(tf.truncated_normal(
									[ll4_filter_size, ll4_filter_size, ll3_depth, ll4_depth], stddev=0.1))
	ll4_biases = tf.Variable(tf.zeros([ll4_depth]))
	ll4_feat_map_size = int(math.ceil(float(ll3_feat_map_size) / ll4_stride))

	######
	# Global feature hyperparameters: two convolutional layers, three FC layers.
	######

	# Input is IMAGE_SIZE/4 x IMAGE_SIZE/4 x 4*DEPTH
	# Output is IMAGE_SIZE/8 x IMAGE_SIZE/8 x 4*DEPTH
	g1_filter_size = 3
	g1_depth = 4*DEPTH
	g1_stride = 2

	# Output is IMAGE_SIZE/8 x IMAGE_SIZE/8 x 4*DEPTH
	g2_filter_size = 3
	g2_depth = 4*DEPTH
	g2_stride = 1

	# Output is IMAGE_SIZE/8 x IMAGE_SIZE/8 x 4*DEPTH
	g3_filter_size = 3
	g3_depth = 4*DEPTH
	g3_stride = 2

	# Output is IMAGE_SIZES/8 x IMAGE_SIZE/8 x 4*DEPTH
	g4_filter_size = 3
	g4_depth = 4*DEPTH
	g4_stride = 1

	# First fully connected layer, outputs 8*DEPTH.
	g5_num_hidden = 8*DEPTH
	# Second fully connected layer.
	g6_num_hidden = 4*DEPTH
	# Third fully connected layer.
	g7_num_hidden = 2*DEPTH

	g1_weights = tf.Variable(tf.truncated_normal(
									[g1_filter_size, g1_filter_size, ll4_depth, g1_depth], stddev=0.1))
	g1_biases = tf.Variable(tf.zeros([g1_depth]))
	g1_feat_map_size = int(math.ceil(float(ll4_feat_map_size) / g1_stride))

	g2_weights = tf.Variable(tf.truncated_normal(
									[g2_filter_size, g2_filter_size, g1_depth, g2_depth], stddev=0.1))
	g2_biases = tf.Variable(tf.zeros([g2_depth]))
	g2_feat_map_size = int(math.ceil(float(g1_feat_map_size) / g2_stride))

	g3_weights = tf.Variable(tf.truncated_normal(
									[g3_filter_size, g3_filter_size, g2_depth, g3_depth], stddev=0.1))
	g3_biases = tf.Variable(tf.zeros([g3_depth]))
	g3_feat_map_size = int(math.ceil(float(g2_feat_map_size) / g3_stride))

	g4_weights = tf.Variable(tf.truncated_normal(
									[g4_filter_size, g4_filter_size, g3_depth, g4_depth], stddev=0.1))
	g4_biases = tf.Variable(tf.zeros([g4_depth]))
	g4_feat_map_size = int(math.ceil(float(g3_feat_map_size) / g4_stride))

	g5_weights = tf.Variable(tf.truncated_normal(
									[g4_feat_map_size * g4_feat_map_size * g4_depth, g5_num_hidden], stddev=0.1))
	g5_biases = tf.Variable(tf.zeros([g5_num_hidden]))

	g6_weights = tf.Variable(tf.truncated_normal(
									[g5_num_hidden, g6_num_hidden], stddev=0.1))
	g6_biases = tf.Variable(tf.zeros([g6_num_hidden]))

	g7_weights = tf.Variable(tf.truncated_normal(
									[g6_num_hidden, g7_num_hidden], stddev=0.1))
	g7_biases = tf.Variable(tf.zeros([g7_num_hidden]))

	######
	# Mid-level feature hyperparameters: two convolutional layers.
	######

	# Input is IMAGE_SIZE/4 x IMAGE_SIZE/4 x 4*DEPTH
	# Output is IMAGE_SIZE/4 x IMAGE_SIZE/4 x 4*DEPTH
	ml1_filter_size = 3
	ml1_depth = 4*DEPTH
	ml1_stride = 1

	# Output is IMAGE_SIZE/4 x IMAGE_SIZE/4 x 2*DEPTH
	ml2_filter_size = 3
	ml2_depth = 2*DEPTH
	ml2_stride = 1

	ml1_weights = tf.Variable(tf.truncated_normal(
									[ml1_filter_size, ml1_filter_size, ll4_depth, ml1_depth], stddev=0.1))
	ml1_biases = tf.Variable(tf.zeros([ml1_depth]))
	ml1_feat_map_size = int(math.ceil(float(ll4_feat_map_size) / ml1_stride))

	ml2_weights = tf.Variable(tf.truncated_normal(
									[ml2_filter_size, ml2_filter_size, ml1_depth, ml2_depth], stddev=0.1))
	ml2_biases = tf.Variable(tf.zeros([ml2_depth]))
	ml2_feat_map_size = int(math.ceil(float(ml1_feat_map_size) / ml2_stride))

	######
	# Colorization layer hyperparameters: one fusion layer, one convolutional layer,
	# one upsample, two more convolutional layers.
	# Expected output size is IMAGE_SIZE/4 x IMAGE_SIZE/4 x 2
	######

	# Input consists of two parts:
	# global features: 2*DEPTH x 1
	# mid level features: IMAGE_SIZE/4 x IMAGE_SIZE/4 x 2*DEPTH
	# Output is IMAGE_SIZE/4 x IMAGE_SIZE/4 x 2*DEPTH
	c1_num_hidden = 2*DEPTH
	c1_filter_size = 3
	c1_stride = 1

	# Output is IMAGE_SIZE/4 x IMAGE_SIZE/4 x DEPTH
	c2_filter_size = 3
	c2_depth = DEPTH
	c2_stride = 1

	# Output is IMAGE_SIZE/2 x IMAGE_SIZE/2 x DEPTH
	c3_upsample_factor = 2
	c3_depth = DEPTH

	# Output is IMAGE_SIZE/2 x IMAGE_SIZE/2 x DEPTH/2
	c4_filter_size = 3
	c4_depth = DEPTH/2
	c4_stride = 1

	# Output is IMAGE_SIZE/2 x IMAGE_SIZE/2 x 2
	c5_filter_size = 3
	c5_depth = FINAL_DEPTH
	c5_stride = 1

	# Output is IMAGE_SIZE x IMAGE_SIZE x 2. Represents the chrominance.
	# Only used for testing and producing results, not for training.
	c6_upsample_factor = 2
	c6_depth = FINAL_DEPTH

	# Set up the weights for the fusion layer.
	# W should be IMAGE_SIZE/4 x IMAGE_SIZE/4 x 4*DEPTH x 2*DEPTH
	c1_weights = tf.Variable(tf.truncated_normal(
									[c1_filter_size, c1_filter_size, 2*c1_num_hidden, c1_num_hidden], stddev=0.1))
	c1_biases = tf.Variable(tf.zeros([c1_num_hidden]))

	c2_weights = tf.Variable(tf.truncated_normal(
									[c2_filter_size, c2_filter_size, c1_num_hidden, c2_depth], stddev=0.1))
	c2_biases = tf.Variable(tf.zeros([c2_depth]))
	c2_feat_map_size = int(math.ceil(float(ml2_feat_map_size) / c2_stride))

	c3_feat_map_size = c3_upsample_factor * c2_feat_map_size

	c4_weights = tf.Variable(tf.truncated_normal(
									[c4_filter_size, c4_filter_size, c3_depth, c4_depth], stddev=0.1))
	c4_biases = tf.Variable(tf.zeros([c4_depth]))
	c4_feat_map_size = int(math.ceil(float(c3_feat_map_size) / c4_stride))

	c5_weights = tf.Variable(tf.truncated_normal(
									[c5_filter_size, c5_filter_size, c4_depth, c5_depth], stddev=0.1))
	c5_biases = tf.Variable(tf.zeros([c5_depth]))
	c5_feat_map_size = int(math.ceil(float(c4_feat_map_size) / c5_stride))

	c6_feat_map_size = c6_upsample_factor * c5_feat_map_size

	######
	# Classification layer hyperparameters: two fully connected layers.
	# Expected output size is NUM_CLASSES
	######
	# Input consists of second-to-last global feature layer: 4*DEPTH x 1
	# First fully connected layer.
	class1_num_hidden = DEPTH*4
	# Second fully connected layer.
	class2_num_hidden = NUM_CLASSES

	class1_weights = tf.Variable(tf.truncated_normal(
									[g6_num_hidden, class1_num_hidden], stddev=0.1))
	class1_biases = tf.Variable(tf.zeros([class1_num_hidden]))

	class2_weights = tf.Variable(tf.truncated_normal(
									[class1_num_hidden, class2_num_hidden], stddev=0.1))
	class2_biases = tf.Variable(tf.zeros([class2_num_hidden]))


	def model(data, train=False):
		# Low level feature network.
		ll1 = tf.nn.conv2d(data, ll1_weights, [1, ll1_stride, ll1_stride, 1], padding='SAME')
		ll1 = tf.nn.relu(batch_norm(ll1 + ll1_biases,train))

		ll2 = tf.nn.conv2d(ll1, ll2_weights, [1, ll2_stride, ll2_stride, 1], padding='SAME')
		ll2 = tf.nn.relu(batch_norm(ll2 + ll2_biases,train))

		ll3 = tf.nn.conv2d(ll2, ll3_weights, [1, ll3_stride, ll3_stride, 1], padding='SAME')
		ll3 = tf.nn.relu(batch_norm(ll3 + ll3_biases,train))

		ll4 = tf.nn.conv2d(ll3, ll4_weights, [1, ll4_stride, ll4_stride, 1], padding='SAME')
		ll4 = tf.nn.relu(batch_norm(ll4 + ll4_biases,train))
		# print 'low level features output shape:', ll4.get_shape().as_list()

		# Global features network.
		g1 = tf.nn.conv2d(ll4, g1_weights, [1, g1_stride, g1_stride, 1], padding='SAME')
		g1 = tf.nn.relu(batch_norm(g1 + g1_biases,train))

		g2 = tf.nn.conv2d(g1, g2_weights, [1, g2_stride, g2_stride, 1], padding='SAME')
		g2 = tf.nn.relu(batch_norm(g2 + g2_biases,train))

		g3 = tf.nn.conv2d(g2, g3_weights, [1, g3_stride, g3_stride, 1], padding='SAME')
		g3 = tf.nn.relu(batch_norm(g3 + g3_biases,train))

		g4 = tf.nn.conv2d(g3, g4_weights, [1, g4_stride, g4_stride, 1], padding='SAME')
		g4 = tf.nn.relu(batch_norm(g4 + g4_biases,train))

		shape = g4.get_shape().as_list()
		# print 'g4 shape, should be 8 x 8 x 256:', shape
		g4 = tf.reshape(g4, [shape[0], shape[1] * shape[2] * shape[3]])
		# print 'g4 shape after reshaping to pass into fully connected layers'
		# print g4.get_shape().as_list()

		# print 'g5 weights shape', g5_weights.get_shape().as_list()
		g5 = tf.nn.relu(batch_norm(tf.matmul(g4, g5_weights) + g5_biases,train,1))
		# print 'g5 shape:', g5.get_shape().as_list()

		g6 = tf.nn.relu(batch_norm(tf.matmul(g5, g6_weights) + g6_biases,train,1))

		g7 = tf.nn.relu(batch_norm(tf.matmul(g6, g7_weights) + g7_biases,train,1))

		# Mid level features network.
		ml1 = tf.nn.conv2d(ll4, ml1_weights, [1, ml1_stride, ml1_stride, 1], padding='SAME')
		ml1 = tf.nn.relu(batch_norm(ml1 + ml1_biases,train))

		ml2 = tf.nn.conv2d(ml1, ml2_weights, [1, ml2_stride, ml2_stride, 1], padding='SAME')
		ml2 = tf.nn.relu(batch_norm(ml2 + ml2_biases,train))
		ml2_shape = ml2.get_shape().as_list()
		# Check that the fusion layer works.
		# print 'g7 shape:', g7.get_shape().as_list()
		# print 'ml2 shape:', ml2.get_shape().as_list()

		# For fusion layer, the intended input should be IMAGE_SIZE/4 x IMAGE_SIZE/4 x 4*DEPTH,
		# which is ml2_feat_map_size x ml2_feat_map_size x (ml2_depth + g7_num_hidden)
		fusion = tf.concat(3,[tf.reshape(tf.tile(g7,[1,ml2_feat_map_size**2]),[ml2_shape[0],ml2_feat_map_size,ml2_feat_map_size,g7_num_hidden]),ml2])
		shape = fusion.get_shape().as_list()
		# print 'fusion shape, should be 32 x 32 x 256:', shape

		# print 'c1 weights shape', c1_weights.get_shape().as_list()
		c1 = tf.nn.conv2d(fusion, c1_weights, [1, c1_stride, c1_stride, 1], padding='SAME')
		c1 = tf.nn.relu(batch_norm(c1 + c1_biases,train))

		# print 'c1 shape', c1.get_shape().as_list()

		c2 = tf.nn.conv2d(c1, c2_weights, [1, c2_stride, c2_stride, 1], padding='SAME')
		c2 = tf.nn.relu(batch_norm(c2 + c2_biases,train))
		c2_shape = c2.get_shape().as_list()
		# print 'c2 shape:', c2_shape

		# Upsample.
		c3 = tf.image.resize_nearest_neighbor(c2, [2*c2_shape[1], 2*c2_shape[2]])
		# print 'c3 shape (after upsample):', c3.get_shape().as_list()

		c4 = tf.nn.conv2d(c3, c4_weights, [1, c4_stride, c4_stride, 1], padding='SAME')
		c4 = tf.nn.relu(batch_norm(c4 + c4_biases,train))
		# print 'c4 shape:', c4.get_shape().as_list()

		# Note that this uses Sigmoid transfer function instead of ReLU.
		c5 = tf.nn.conv2d(c4, c5_weights, [1, c5_stride, c5_stride, 1], padding='SAME')
		c5 = tf.nn.sigmoid(batch_norm(c5 + c5_biases,train))
		c5_shape = c5.get_shape().as_list()
		# print 'c5 shape:', c5_shape

		# Only need classification layer during training.
		if train:
			class1 = tf.nn.relu(batch_norm(tf.matmul(g6, class1_weights) + class1_biases,train,1))
			print class1.get_shape()
			class2 = tf.matmul(class1, class2_weights) + class2_biases
			print class2.get_shape()

			# Dropout training.
			c5 = tf.nn.dropout(c5, .5, seed=SEED)
			return (c5, class2)
		else:
			# ONLY DURING TESTING, NOT TRAINING:
			# Upsample again, then merge with original image.
			c6 = tf.image.resize_nearest_neighbor(c5, [2*c5_shape[1], 2*c5_shape[2]])
			# print 'not training, c6 shape:', c6.get_shape().as_list()
			return c6

	# Use the model to get logits.
	train_color_logits, train_classify_logits = model(train_data_node, train=True)
	# Use mean squared error for loss for colorization network and cross-entropy loss in classification network.
	loss = tf.reduce_sum(tf.square(train_colors_node - train_color_logits)) + ALPHA * tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(train_classify_logits, train_class_node))
	optimizer = tf.train.AdadeltaOptimizer(learning_rate=.01).minimize(loss)

	train_prediction = tf.nn.softmax(train_classify_logits)
	eval_prediction = tf.nn.softmax(model(eval_data, train=False))

	# Initialize variables.
	init = tf.initialize_all_variables()
	sess.run(init)

	# Get training data, in batches.
	train_dir = IMAGES_DIR + 'train/'
	train_filenames = os.listdir(train_dir)
	num_images = len(train_filenames)
	assert(num_images == NUM_TRAIN_IMAGES)
	num_batches = int(math.ceil(float(num_images/BATCH_SIZE)))
	# Iterate through batches.
	for batch in xrange(num_batches):
		# Create zeroed arrays that we will fill with appropriate image data.
		bw_images = np.zeros([BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 1], dtype=np.float32)
		color_features = np.zeros([BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 2], dtype=np.float32)
		class_labels = np.zeros([BATCH_SIZE, NUM_CLASSES])
		# Determine where in the global list of files we should start for this batch.
		start_idx = batch * BATCH_SIZE
		for i in xrange(BATCH_SIZE):
                        # Get the next file to look at, based on precalculated random order.
			file_idx = TRAINING_IMAGES_INDEX[start_idx + i]
			filename = train_filenames[file_idx]
			label = one_hot_from_filename(filename)
			class_labels[i] = label
			im = read_scaled_color_image_Lab(train_dir + filename)
			im_bw = im[:,:,0].reshape((-1,IMAGE_SIZE,IMAGE_SIZE,1))
			im_c = im[:,:,1:].reshape((-1,IMAGE_SIZE,IMAGE_SIZE,2))
			bw_images[i] = im_bw
			color_features[i] = im_c

		x = bw_images
		y = color_features
		y_downsample = tf.image.resize_nearest_neighbor(y, [IMAGE_SIZE/2, IMAGE_SIZE/2]).eval(session=sess)
		feed_dict = {train_data_node: x,
					 train_colors_node: y_downsample,
					 train_class_node: class_labels}

		# Train the model.
		_, l = sess.run([optimizer, loss], feed_dict=feed_dict)

		# Every so often, evaluate.
		if batch % EVAL_FREQUENCY == 0:
			# Load the validation data for this batch..
			val_dir = IMAGES_DIR + 'val/'
			val_data = np.zeros([EVAL_BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 1])
			val_color_labels = np.zeros([EVAl_BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 2])
			val_filenames = os.listdir(val_dir)
			eval_start_idx = batch // EVAL_FREQUENCY * EVAL_BATCH_SIZE
			for i in xrange(EVAL_BATCH_SIZE):
				file_idx = eval_start_idx + i
				filename = val_filenames[file_idx]
				im = read_scaled_color_image_Lab(val_dir + filename)
				im_bw = im[:,:,0].reshape((-1,IMAGE_SIZE,IMAGE_SIZE,1))
				im_c = im[:,:,1:].reshape((-1,IMAGE_SIZE,IMAGE_SIZE,2))
				val_data[file_idx] = im_bw
				val_color_labels[file_idx] = im_c

			feed_dict = {eval_data: val_data}
			eval_predictions = np.array(sess.run([eval_prediction], feed_dict=feed_dict))
			error = error_rate(eval_predictions, val_color_labels)
			print('Step: %d' % batch)
			print('Validation error: %f' % error)

	# Ready to test!
	# Load the test data.
	test_dir = IMAGES_DIR + 'test/'
	test_data = np.zeros([NUM_TEST_IMAGES, IMAGE_SIZE, IMAGE_SIZE, 1], dtype=np.float32)
	test_color_labels = np.zeros([NUM_TEST_IMAGES, IMAGE_SIZE, IMAGE_SIZE, 2], dtype=np.float32)
	test_filenames = os.listdir(test_dir)
	for file_idx in xrange(len(test_filenames)):
		filename = test_filenames[file_idx]
		im = read_scaled_color_image_Lab(val_dir + filename)
		im_bw = im[:,:,0].reshape((-1,IMAGE_SIZE,IMAGE_SIZE,1))
		im_c = im[:,:,1:].reshape((-1,IMAGE_SIZE,IMAGE_SIZE,2))
		test_data[file_idx] = im_bw
		test_color_labels[file_idx] = im_c
	feed_dict = {eval_data: test_data}
	test_predictions = np.array(sess.run([eval_prediction], feed_dict=feed_dict))
	test_error = error_rate(test_predictions, test_color_labels)
	print 'Test error: %f' % test_error

if __name__ == '__main__':
	main()
	#color_small()
