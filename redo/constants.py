IMAGES_DIR = "fake"

DEPTH = 64 # Used to parameterize the depth of each output layer.
NUM_TRAIN_IMAGES = 100000 # True value is 100000, change to train on fewer.
NUM_VAL_IMAGES = 1000     # Should be 10,000 for actual dataset.
NUM_TEST_IMAGES = NUM_VAL_IMAGES      # Should be 10,000 for actual dataset.
NUM_CLASSES = 100

NUM_EPOCHS = 30
BATCH_SIZE = 32
TRAIN_DATASET_REPEAT_FACTOR = 100 # Not sure about this...
LR = 1e-5
IMAGE_SIZE = 128