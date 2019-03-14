# Created by Yanbing Jiang Copyright 2019.

# This file converts images' bottlenecks to TFRecords in multi thread to work efficiently

'''
The raw ImageNet data set is expected to reside in JPEG files located in the
following directory structure.

  bottleneck_dir/daisy/*****************.JPEG
  bottleneck_dir/sunflowers/###############.JPEG
  ...

where 'daisy' is the unique synset label associated with these images.

By specify the percentage of validation set and testing set, the following files will 
be generated:
  TFRecords_directory/flowers_train_00000-of-00005
  TFRecords_directory/flowers_train_00001-of-00005
  ...
  TFRecords_directory/flowers_validation_00000-of-00005
  ...
  TFRecords_directory/flowers_test_00000-of-00005

Each validation TFRecord file contains the bottleneck values and the ground truth array
e.g. [0, 0, 1, 0, 0] means number 3 is the ground truth

Things to specify in the FLAG:
bottleneck_dir = bottleneck in .txt location
tfrecord_dir   = desired bottleneck in .tfrecord location
validation_percentage = percetage of validation set from the data set
testing_percentage    = percetage of testing set from the data set
num_shards = number of shards you want for tfrecords i.e. how many tfrecord files you want
num_threads = number of threads to process on

!!!!!! NOTE THAT num_shards % num_threads == 0 to make it work

'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime  
import argparse  
import os
import random
import sys
import threading

import numpy as np
import tensorflow as tf
import time
import math

# Seed for repeatability.
_RANDOM_SEED = 0

def ensure_dir_exists(dir_name):
  """Makes sure the folder exists on disk.

  Args:
    dir_name: Path string to the folder we want to create.
  """  
  if not os.path.exists(dir_name):  
    os.makedirs(dir_name)  

def int64_feature(values):
  if not isinstance(values,  (tuple, list)):
    values = [values]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=values))

def _bytes_feature(value):
  """Wrapper for inserting bytes features into Example proto."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  """Wrapper for inserting float features into Example proto."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def image_to_tfexample(image_bottleneck, ground_truth):
    return tf.train.Example(features=tf.train.Features(feature={
        "bottleneck": _float_feature(image_bottleneck),
        # "bottleneck": _bytes_feature(tf.compat.as_bytes(image_bottleneck)),
        "image/class/label": _float_feature(ground_truth),
    }))

def _get_filenames_and_classes(dataset_dir):
  """Returns a list of filenames and inferred class names.

  Args:
    dataset_dir: A directory containing a set of subdirectories representing
      class names. Each subdirectory should contain PNG or JPG encoded images.

  Returns:
    A list of image file paths, relative to `dataset_dir` and the list of
    subdirectories, representing class names.
  """
  # flower_root = os.path.join(dataset_dir, 'flower_photos')
  flower_root = dataset_dir
  directories = []
  class_names = []
  for filename in os.listdir(flower_root):
    path = os.path.join(flower_root, filename)
    if os.path.isdir(path):
      directories.append(path)
      class_names.append(filename)

  photo_filenames = []
  for directory in directories:
    for filename in os.listdir(directory):
      path = os.path.join(directory, filename)
      photo_filenames.append(path)

  return photo_filenames, sorted(class_names)

def _get_dataset_filename(dataset_dir, split_name, shard_id):
    output_filename = 'flowers_%s_%05d-of-%05d.tfrecord' % (split_name, shard_id, FLAGS.num_shards)
    return os.path.join(dataset_dir, output_filename)

def _convert_dataset(split_name, filenames, class_names_to_ids, dataset_dir, class_count, thread_index, ranges):
      
    assert split_name in ['train', 'validation', 'test']

    num_threads_c = len(ranges)

    assert not FLAGS.num_shards % num_threads_c
    num_shards_per_batch = int(FLAGS.num_shards / num_threads_c)

    shard_ranges = np.linspace(ranges[thread_index][0],
                               ranges[thread_index][1],
                               num_shards_per_batch + 1).astype(int)
    num_files_in_thread = ranges[thread_index][1] - ranges[thread_index][0]

    counter = 0
    for s in range(num_shards_per_batch):
      # Generate a sharded version of the file name, e.g. 'flowers_train-00001-of-00005'
      shard = thread_index * num_shards_per_batch + s
      output_file = _get_dataset_filename(dataset_dir, split_name, shard)
      tfrecord_writer = tf.python_io.TFRecordWriter(output_file)

      shard_counter = 0
      files_in_shard = np.arange(shard_ranges[s], shard_ranges[s + 1], dtype=int)
      for i in files_in_shard:
        with open(filenames[i], 'r') as bottleneck_file:  
          bottleneck_string = bottleneck_file.read()  

        bottleneck_values = [float(x) for x in bottleneck_string.split(',')]

        class_name = os.path.basename(os.path.dirname(filenames[i]))
        class_id = class_names_to_ids[class_name]

    
        ground_truth = np.zeros(class_count, dtype=np.float32)  
        ground_truth[class_id] = 1.0 

        example = image_to_tfexample(bottleneck_values, ground_truth)
        tfrecord_writer.write(example.SerializeToString())

        shard_counter += 1
        counter += 1

      tfrecord_writer.close()
      print('[thread %d]: Writing %d images to %s' %
            (thread_index, shard_counter, output_file))
      sys.stdout.flush()
      shard_counter = 0
    print('[thread %d]: Wrote %d images to %d shards.' %
          (thread_index, counter, num_files_in_thread))
    sys.stdout.flush()


def _process_image_files(name, filenames, class_names_to_ids, dataset_dir, class_count):
  """Process and save list of images as TFRecord of Example protos.

  Args:
    name: string, unique identifier specifying the data set
    filenames: list of strings; each string is a path to an image file
    texts: list of strings; each string is human readable, e.g. 'dog'
    labels: list of integer; each integer identifies the ground truth
    num_shards: integer number of shards for this data set.
  """
  # assert len(filenames) == len(texts)
  # assert len(filenames) == len(labels)

  # Break all images into batches with a [ranges[i][0], ranges[i][1]].
  spacing = np.linspace(0, len(filenames), FLAGS.num_threads + 1).astype(np.int)
  ranges = []
  for i in range(len(spacing) - 1):
    ranges.append([spacing[i], spacing[i+1]])

  # Launch a thread for each batch.
  print('Launching %d threads for spacings: %s' % (FLAGS.num_threads, ranges))
  sys.stdout.flush()

  # Create a mechanism for monitoring when all threads are finished.
  coord = tf.train.Coordinator()

  threads = []

  for thread_index in range(len(ranges)):
    args = (name, filenames, class_names_to_ids, dataset_dir, class_count, thread_index, ranges)
    t = threading.Thread(target=_convert_dataset, args=args)
    t.start()
    threads.append(t)

  # Wait for all the threads to terminate.
  coord.join(threads)
  print('%s: Finished writing all %d images in data set.' %
        (datetime.now(), len(filenames)))
  sys.stdout.flush()

def main(unused_argv):

    startTime = time.time() # Starting Time, for time consumption calculation purpose

    ensure_dir_exists(FLAGS.tfrecord_dir) # Guarantee TFRecords can be stored in the location indicated

    bottleneck_filenames, class_names = _get_filenames_and_classes(FLAGS.bottleneck_dir)
    class_names_to_ids = dict(zip(class_names, range(len(class_names))))

    total_num_files = len(bottleneck_filenames) # Get the total number of files

    # Get the number of files in each category
    num_validation = int(total_num_files*(FLAGS.validation_percentage/100))
    num_test       = int(total_num_files*(FLAGS.testing_percentage/100))

    class_count = len(class_names)

    # Divide into train, validation and test:
    random.seed(_RANDOM_SEED)
    random.shuffle(bottleneck_filenames)
    training_filenames = bottleneck_filenames[(num_validation + num_test):]
    validation_filenames = bottleneck_filenames[:num_validation]
    testing_files = bottleneck_filenames[num_validation:(num_validation + num_test)]


    _process_image_files('train', training_filenames, class_names_to_ids, FLAGS.tfrecord_dir, class_count)
    _process_image_files('validation', validation_filenames, class_names_to_ids, FLAGS.tfrecord_dir, class_count)
    _process_image_files('test', testing_files, class_names_to_ids, FLAGS.tfrecord_dir, class_count)

    print("Time taken: %f" % (time.time() - startTime))

    # First, convert the training and validation sets.
    # _convert_dataset('train', training_filenames, class_names_to_ids, TFRECORD_PATH, class_count)
    # _convert_dataset('validation', validation_filenames, class_names_to_ids, TFRECORD_PATH, class_count)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()  
  parser.add_argument(  
      '--bottleneck_dir',  
      type=str,  
      default='./bottleneck',  
      help='Location of bottleneck in .txt previously generated'  
  )
  parser.add_argument(  
      '--tfrecord_dir',  
      type=str,  
      default='./bottleneck_tf',  
      help='Location of bottleneck in .tfrecord desired'  
  )
  parser.add_argument(  
      '--validation_percentage',  
      type=int,  
      default=10,  
      help='Percetage of validation set from the data set'  
  )
  parser.add_argument(  
      '--testing_percentage',  
      type=int,  
      default=10,  
      help='Percetage of testing set from the data set'  
  )
  parser.add_argument(  
      '--num_shards',  
      type=int,  
      default=5,  
      help='Number of shards.'  
  )
  parser.add_argument(  
      '--num_threads',  
      type=int,  
      default=5,  
      help='Number of shards.'  
  )
  FLAGS, unparsed = parser.parse_known_args()  
tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)  
  

    
