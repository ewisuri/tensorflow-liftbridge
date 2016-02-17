import os

import tensorflow.python.platform
import tensorflow as tf

from tensorflow.python.platform import gfile


# Processing definitions


IMAGE_SIZE = 96

NUM_CLASSES = 2 # [UP, DOWN]
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 24
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 12


def read_liftbridge(filename_queue):
    class LiftbridgeRecord(object):
        pass
    result = LiftbridgeRecord()
    
    label_bytes = 1
    result.height = 128
    result.width = 128
    result.depth = 3
    image_bytes = result.height * result.width * result.depth
    
    record_bytes = label_bytes + image_bytes
    
    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
    result.key, value = reader.read(filename_queue)
    
    record_bytes = tf.decode_raw(value, tf.uint8)
    
    result.label = tf.cast(
        tf.slice(record_bytes, [0], [label_bytes]), tf.int32)
    
    depth_major = tf.reshape(tf.slice(record_bytes, [label_bytes], [image_bytes]),
                             [result.depth, result.height, result.width])
    
    result.uint8image = tf.transpose(depth_major, [1, 2, 0])
    
    return result
    

def _generate_image_and_label_batch(image, label, min_queue_examples,
                                    batch_size):
    num_preprocess_threads = 16
    images, label_batch = tf.train.shuffle_batch(
        [image, label],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity = min_queue_examples + 3 * batch_size,
        min_after_dequeue=min_queue_examples)
    
    tf.image_summary('images', images)
    
    return images, tf.reshape(label_batch, [batch_size])


def distorted_inputs(data_dir, batch_size):
    filenames = ['train.bin']
    for f in filenames:
        if not gfile.Exists(f):
            raise ValueError('Failed to find file: ' + f)
    
    filename_queue = tf.train.string_input_producer(filenames)
    
    read_input = read_liftbridge(filename_queue)
    reshaped_image = tf.cast(read_input.uint8image, tf.float32)
    
    height = IMAGE_SIZE
    width = IMAGE_SIZE
    
    # Randomly crop a [height, width] section of the image.
    distorted_image = tf.image.random_crop(reshaped_image, [height, width])
    
    # Randomly flip the image horizontally
    distorted_image = tf.image.random_flip_left_right(distorted_image)
    
    # Because these operations are not commutative, consider randomizing
    # the order of the operation
    distorted_image = tf.image.random_brightness(distorted_image,
                                                 max_delta=63)
    distorted_image = tf.image.random_contrast(distorted_image,
                                               lower=0.2, upper=1.8)
    
    # Subtract off the mean and divide by the variance of the pixels.
    float_image = tf.image.per_image_whitening(distorted_image)

    # Ensure that the random shuffling has good mixing properties.
    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
                            min_fraction_of_examples_in_queue)
    print ('Filling queue with %d LIFTBRIDGE images before starting to train. '
            'This will take a few minutes.' % min_queue_examples)

    # Generate a batch of images and labels by building up a queue of examples.
    return _generate_image_and_label_batch(float_image, read_input.label,
                                           min_queue_examples, batch_size)


def inputs(eval_data, data_dir, batch_size):
    if not eval_data:
        filenames = [os.path.join(data_dir, 'train.bin')]
        num_examples_per_epoc = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
    else:
        filenames = [os.path.join(data_dir, 'eval.bin')]
        num_examples_per_epoc = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL
        
    for f in filenames:
        if not gfile.Exists(f):
            raise ValueError('Failed to find file: ' + f)
        
    filename_queue = tf.train.string_input_producer(filenames)
    
    read_input = read_liftbridge(filename_queue)
    reshaped_image = tf.cast(read_input.uint8image, tf.float32)
    
    height = IMAGE_SIZE
    width = IMAGE_SIZE
    
    float_image = tf.image.per_image_whitening(resized_image)
    
    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(num_examples_per_epoc *
                             min_fraction_of_examples_in_queue)
    
    return _generate_image_and_label_batch(float_image, read_input.label,
                                           min_queue_examples, batch_size)