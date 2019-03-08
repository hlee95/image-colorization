import tensorflow as tf

from image_util import ImageUtil
from model import ImageColorization
from data_loader import ImageDataLoader
from constants import *


def main():
    # TODO: create a tf session.
    sess = tf.Session()
    image_util = ImageUtil()
    dataloader = ImageDataLoader(IMAGES_DIR, image_util)

    train_dataset = dataloader.get_dataset('train')
    val_dataset = dataloader.get_dataset('val')
    test_dataset = dataloader.get_dataset('test')

    model = ImageColorization().model
    model.compile(optimizer=tf.train.AdamOptimizer(),
                  loss={'colorization_output': 'mean_squared_error', 'classification_output': 'categorical_crossentropy'},
                loss_weights={'colorization_output': 1., 'classification_output': 1./300})

    print 'Starting training'
    model.fit(train_dataset['greyscale'],
        {'colorization_output': train_dataset['color'], 'classification_output': train_dataset['class']},
        epochs=NUM_EPOCHS, steps_per_epoch=30)
    print 'Finished training'

    results = model.predict(test_dataset['greyscale'])
    #image_util.save_result_images(upsampled_results[only take the image part])


if __name__ == "__main__":
    main()
    print 'Done.'
