# Created by Yanbing Jiang Copyright 2019.

# This file converts images' bottlenecks to TFRecords 

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

'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse  
import os
import random
import sys

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
  if not isinstance(values, (tuple, list)):
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

def _convert_dataset(split_name, filenames, class_names_to_ids, dataset_dir, class_count):
      
    assert split_name in ['train', 'validation', 'test']

    num_per_shard = int(math.ceil(len(filenames) / float(FLAGS.num_shards)))

    for shard_id in range(FLAGS.num_shards):
      output_filename = _get_dataset_filename(dataset_dir, split_name, shard_id)

      with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
        start_ndx = shard_id * num_per_shard
        end_ndx = min((shard_id+1) * num_per_shard, len(filenames))
        for i in range(start_ndx, end_ndx):
          sys.stdout.write('\r>> Converting %s image %d/%d shard %d' % (split_name, i+1, len(filenames), shard_id))
          sys.stdout.flush()

      # Read the filename:

          with open(filenames[i], 'r') as bottleneck_file:  
            bottleneck_string = bottleneck_file.read()  

          bottleneck_values = [float(x) for x in bottleneck_string.split(',')]

          # print(type(bottleneck_values[0]))
          # exit(0)

          class_name = os.path.basename(os.path.dirname(filenames[i]))
          class_id = class_names_to_ids[class_name]

      
          ground_truth = np.zeros(class_count, dtype=np.float32)  
          ground_truth[class_id] = 1.0 

          example = image_to_tfexample(bottleneck_values, ground_truth)
          tfrecord_writer.write(example.SerializeToString())

    sys.stdout.write('\n')
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

    # First, convert the training and validation sets.
    _convert_dataset('train', training_filenames, class_names_to_ids, FLAGS.tfrecord_dir, class_count)
    _convert_dataset('validation', validation_filenames, class_names_to_ids, FLAGS.tfrecord_dir, class_count)
    _convert_dataset('test', testing_files, class_names_to_ids, FLAGS.tfrecord_dir, class_count)

    print("Time taken to convert TFRecord: %f" % (time.time() - startTime))

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
  FLAGS, unparsed = parser.parse_known_args()  
tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)  
    
