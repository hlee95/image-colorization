import math
import numpy as np
import tensorflow as tf
from skimage import io, color
import matplotlib.pyplot as plt
from matplotlib.colors import NoNorm

from image_util import ImageUtil
from model import ImageColorization
from data_loader import ImageDataLoader
from constants import *

def main():
    sess = tf.Session()
    image_util = ImageUtil()
    dataloader = ImageDataLoader(IMAGES_DIR, image_util)

    train_dataset = dataloader.get_dataset('train', NUM_TRAIN_IMAGES)
    val_dataset = dataloader.get_dataset('val', NUM_VAL_IMAGES)
    test_dataset = dataloader.get_dataset('test', NUM_TEST_IMAGES)

    model = ImageColorization().model
    model.compile(optimizer=tf.train.AdamOptimizer(),
                  loss={'colorization_output': 'mean_squared_error', 'classification_output': 'categorical_crossentropy'},
                loss_weights={'colorization_output': 1., 'classification_output': 1./300})

    print 'Starting training'
    for i in range(NUM_EPOCHS):
        model.fit(train_dataset['greyscale'], {'colorization_output': train_dataset['color'], 'classification_output': train_dataset['class']}, epochs=1)
        ab_unscaled = model.predict(val_dataset['greyscale'])[0]
        lab = np.concatenate([val_dataset['greyscale']*100, ab_unscaled*200-100], axis=3)
        show_predicted_values(ab_unscaled, lab)
        
    print 'Finished training'

    results = model.predict(test_dataset['greyscale'])
    # image_util.save_result_images(upsampled_results[only take the image part])

def show_predicted_values(ab_unscaled, lab):
    num_images = lab.shape[0]
    num_rows = int(math.sqrt(num_images))
    num_columns = num_images/num_rows

    # reshape predictions into image grid
    a_grid = np.zeros([IMAGE_SIZE*num_rows,IMAGE_SIZE*num_columns,1])
    b_grid = np.zeros([IMAGE_SIZE*num_rows,IMAGE_SIZE*num_columns,1])
    rgb_grid = np.zeros([IMAGE_SIZE*num_rows,IMAGE_SIZE*num_columns,3])

    for i in range(num_rows):
        for j in range(num_columns):
            a_grid[IMAGE_SIZE*i:IMAGE_SIZE*(i+1),IMAGE_SIZE*j:IMAGE_SIZE*(j+1),:] = ab_unscaled[num_columns*i+j,:,:,0:1]
            b_grid[IMAGE_SIZE*i:IMAGE_SIZE*(i+1),IMAGE_SIZE*j:IMAGE_SIZE*(j+1),:] = ab_unscaled[num_columns*i+j,:,:,1:2]
            rgb_grid[IMAGE_SIZE*i:IMAGE_SIZE*(i+1),IMAGE_SIZE*j:IMAGE_SIZE*(j+1),:] = color.lab2rgb(lab[10*i+j,:,:,:])
    
    plt.imshow(a_grid[:,:,0], cmap='gray', norm=NoNorm())
    plt.show()
    plt.title("Histogram of predicted a values (min 0, max 1)")
    plt.hist(a_grid.flatten())
    plt.show()

    plt.imshow(b_grid[:,:,0], cmap='gray', norm=NoNorm())
    plt.show()
    plt.title("Histogram of predicted b values (min 0, max 1)")
    plt.hist(b_grid.flatten())
    plt.show()
    
    plt.imshow(rgb_grid)
    plt.show()

if __name__ == "__main__":
    main()
    print 'Done.'
