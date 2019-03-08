IMAGES_DIR = "../data"

# Dataset parameters
# dataset used is CIFAR-10 from https://www.cs.toronto.edu/~kriz/cifar.html
# change NUM_CLASSES and IMAGE_SIZE if using different dataset
IMAGE_SIZE = 32
NUM_CLASSES = 10

# Model parameters
DEPTH = 64  # Used to parameterize the depth of each output layer.
CONV_KERNEL_SIZE = 3

# Training/inference parameters
NUM_TRAIN_IMAGES = 100 
NUM_VAL_IMAGES = 100
NUM_TEST_IMAGES = 100
NUM_EPOCHS = 1
BATCH_SIZE = 32
LR = 1e-5
