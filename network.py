import tensorflow as tf
import math
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
from scipy import misc

def color_small():
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
	x_train = np.stack((sailboat_bw,lake_bw),-1)
	y_train = np.stack((sailboat_c,lake_c),-1)
	sess.run(train_step, feed_dict={x: sailboat_bw, y_: sailboat_c})
	#sess.close()
	#correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
	#accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	#print(sess.run(accuracy, feed_dict={x: mnist.test.images.reshape(-1,28,28,1), y_: mnist.test.labels}))

#color_small()

NUM_IMAGES = 2
IMAGE_SIZE = 64
DEPTH = 3

# low level feature hyperparameters
ll1_filter_size = 3
ll1_depth = DEPTH
ll1_stride = 2

ll2_filter_size = 3
ll2_depth = 2*DEPTH
ll2_stride = 1

ll3_filter_size = 3
ll3_depth = 2*DEPTH
ll3_stride = 2

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

# global feature hyperparameters

g1_filter_size = 3
g1_depth = 4*DEPTH
g1_stride = 2

g2_filter_size = 3
g2_depth = 4*DEPTH
g2_stride = 1

g3_num_hidden = 8*DEPTH

g4_num_hidden = 4*DEPTH

g5_num_hidden = 2*DEPTH

g1_weights = tf.Variable(tf.truncated_normal(
				[g1_filter_size, g1_filter_size, ll4_depth, g1_depth], stddev=0.1))
g1_biases = tf.Variable(tf.zeros([g1_depth]))
g1_feat_map_size = int(math.ceil(float(ll4_feat_map_size) / g1_stride))

g2_weights = tf.Variable(tf.truncated_normal(
				[g2_filter_size, g2_filter_size, g1_depth, g2_depth], stddev=0.1))
g2_biases = tf.Variable(tf.zeros([g2_depth]))
g2_feat_map_size = int(math.ceil(float(g1_feat_map_size) / g2_stride))

g3_weights = tf.Variable(tf.truncated_normal(
				[g2_feat_map_size * g2_feat_map_size * g2_depth, g3_num_hidden], stddev=0.1))
g3_biases = tf.Variable(tf.zeros([g3_num_hidden]))

g4_weights = tf.Variable(tf.truncated_normal(
				[g3_num_hidden, g4_num_hidden], stddev=0.1))
g4_biases = tf.Variable(tf.zeros([g4_num_hidden]))

g5_weights = tf.Variable(tf.truncated_normal(
				[g4_num_hidden, g5_num_hidden], stddev=0.1))
g5_biases = tf.Variable(tf.zeros([g5_num_hidden]))

# mid-level feature hyperparameters

ml1_filter_size = 3
ml1_depth = 4*DEPTH
ml1_stride = 1

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

# colorization layer hyperparameters

c1_num_hidden = 2*DEPTH

c2_filter_size = 3
c2_depth = 64
c2_stride = 1

c3_filter_size = 3

c4_filter_size = 3
c4_depth = 64
c4_stride = 1

c5_filter_size = 3
c5_depth = 64
c5_stride = 1

c1_weights = tf.Variable(tf.truncated_normal(
				[ml2_feat_map_size, ml2_feat_map_size, 2*c1_num_hidden, c1_num_hidden], stddev=0.1))
c1_biases = tf.Variable(tf.zeros([c1_num_hidden]))

c2_weights = tf.Variable(tf.truncated_normal(
				[c2_filter_size, c2_filter_size, c1_num_hidden, c2_depth], stddev=0.1))
c2_biases = tf.Variable(tf.zeros([c2_depth]))
c2_feat_map_size = int(math.ceil(float(ml2_feat_map_size) / c2_stride))

'''

c4_weights = tf.Variable(tf.truncated_normal(
				[m2_filter_size, m1_filter_size, c1_num_hidden, c2_depth], stddev=0.1))
c4_biases = tf.Variable(tf.zeros([m1_depth]))
c4_feat_map_size = int(math.ceil(float(ll4_feat_map_size) / m1_stride))

c5_weights = tf.Variable(tf.truncated_normal(
				[m2_filter_size, m1_filter_size, c1_num_hidden, c2_depth], stddev=0.1))
c5_biases = tf.Variable(tf.zeros([m1_depth]))
c5_feat_map_size = int(math.ceil(float(ll4_feat_map_size) / m1_stride))
'''
# classification layer hyperparameters

x = tf.placeholder(tf.float32, [NUM_IMAGES, IMAGE_SIZE*IMAGE_SIZE])
x = tf.reshape(x, [-1,IMAGE_SIZE,IMAGE_SIZE,1])

y_ = tf.placeholder(tf.float32, [None, IMAGE_SIZE, IMAGE_SIZE, 2])

ll1 = tf.nn.conv2d(x, ll1_weights, [1, ll1_stride, ll1_stride, 1], padding='SAME')
ll1 = tf.nn.relu(ll1 + ll1_biases)

ll2 = tf.nn.conv2d(ll1, ll2_weights, [1, ll2_stride, ll2_stride, 1], padding='SAME')
ll2 = tf.nn.relu(ll2 + ll2_biases)

ll3 = tf.nn.conv2d(ll2, ll3_weights, [1, ll3_stride, ll3_stride, 1], padding='SAME')
ll3 = tf.nn.relu(ll3 + ll3_biases)

ll4 = tf.nn.conv2d(ll3, ll4_weights, [1, ll4_stride, ll4_stride, 1], padding='SAME')
ll4 = tf.nn.relu(ll4 + ll4_biases)

g1 = tf.nn.conv2d(ll4, g1_weights, [1, g1_stride, g1_stride, 1], padding='SAME')
g1 = tf.nn.relu(g1 + g1_biases)

g2 = tf.nn.conv2d(g1, g2_weights, [1, g2_stride, g2_stride, 1], padding='SAME')
g2 = tf.nn.relu(g2 + g2_biases)

shape = g2.get_shape().as_list()
g2 = tf.reshape(g2, [shape[0], shape[1] * shape[2] * shape[3]])
g3 = tf.nn.relu(tf.matmul(g2, g3_weights) + g3_biases)

g4 = tf.nn.relu(tf.matmul(g3, g4_weights) + g4_biases)

g5 = tf.nn.relu(tf.matmul(g4, g5_weights) + g5_biases)

ml1 = tf.nn.conv2d(ll4, ml1_weights, [1, ml1_stride, ml1_stride, 1], padding='SAME')
ml1 = tf.nn.relu(ml1 + ml1_biases)

ml2 = tf.nn.conv2d(ml1, ml2_weights, [1, ml2_stride, ml2_stride, 1], padding='SAME')
ml2 = tf.nn.relu(ml2 + ml2_biases)

# check to make sure this actually works
fusion = tf.concat(0,[tf.reshape(tf.tile(g5,[16,16]),[2,16,16,6]),ml2])
shape = fusion.get_shape().as_list()
print shape
fusion = tf.reshape(fusion, [shape[0], shape[1] * shape[2] * shape[3]])
c1 = tf.nn.relu(tf.matmul(fusion, c1_weights) + c1_biases)

c2 = tf.nn.conv2d(c1, c2_weights, [1, c2_stride, c2_stride, 1], padding='SAME')
c2 = tf.nn.relu(c2 + c2_biases)

'''
c3 = tf.image.resize_nearest_neighbor(images, size, align_corners=None, name=None)

hidden = tf.nn.sigmoid(conv1 + layer1_biases)

tf.image.resize_nearest_neighbor(images, size, align_corners=None, name=None)

loss = tf.reduce_mean(tf.square(g4, tf_train_labels))
optimizer = tf.train.AdadeltaOptimizer(learning_rate).minimize(loss)

init = tf.initialize_all_variables()
sess = tf.Session()
result = sess.run(init)

sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
#print(result)
'''