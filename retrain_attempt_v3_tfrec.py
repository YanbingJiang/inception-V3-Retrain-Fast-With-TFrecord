# Created by Yanbing Jiang Copyright 2019.

# This file train. validate and test transfer learning withTFRecords 

'''

TFRecords are stored as following:

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
image_dir = original_image directory for get class # bottleneck # purpose
tfrecord_dir   = desired bottleneck in .tfrecord location
validation_percentage = percetage of validation set from the data set
testing_percentage    = percetage of testing set from the data set
training/validation batch size
Total steps
Evaluation step
Learning Rate


###### NOTICE AN IMPROVEMENT could be made at 'tf.case', will be updated later  ##############
'''

# os.environ["CUDA_VISIBLE_DEVICES"]="-1"  
from __future__ import absolute_import  
from __future__ import division  
from __future__ import print_function  
  
import argparse  
from datetime import datetime  
import hashlib  
import os.path
import os
import random  
import re  
import struct  
import sys  
import tarfile
import time
  
import numpy as np  
from six.moves import urllib
import tensorflow as tf  
  
from tensorflow.python.framework import graph_util  
from tensorflow.python.framework import tensor_shape  
from tensorflow.python.platform import gfile  
from tensorflow.python.util import compat  
import csv
import pandas as pd

FLAGS = None  
  
# pylint：disable=line-too-long  
DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'

# file list folder
FILE_LIST_FOLDER = './file_list/'

# pylint: enable=line-too-long  
BOTTLENECK_TENSOR_NAME = 'pool_3/_reshape:0'  
BOTTLENECK_TENSOR_SIZE = 2048  
MODEL_INPUT_WIDTH = 299  
MODEL_INPUT_HEIGHT = 299  
MODEL_INPUT_DEPTH = 3  
JPEG_DATA_TENSOR_NAME = 'DecodeJpeg/contents:0'  
RESIZED_INPUT_TENSOR_NAME = 'ResizeBilinear:0'  
MAX_NUM_IMAGES_PER_CLASS = 2 ** 27 - 1  # ~134M  

TFRECORD_PATH   = '.\\bottleneck_tf'
  
def create_image_lists(image_dir):  

  if not gfile.Exists(image_dir):  
    print("Image directory '" + image_dir + "' not found.")  
    return None  
  result = {}  
  sub_dirs = [x[0] for x in gfile.Walk(image_dir)]

  # The root directory comes first, so skip it.
  is_root_dir = True
  counter = 0

  for sub_dir in sub_dirs:  
    if is_root_dir:  
      is_root_dir = False  
      continue

    # Different types of extensions
    extensions = ['jpg', 'jpeg', 'JPG', 'JPEG']  
    file_list = []  
    dir_name = os.path.basename(sub_dir)  
    if dir_name == image_dir:  
      continue

    # New For Stable
    file_list_path = FILE_LIST_FOLDER + 'file_list_' + dir_name +'.csv'
    file_list_exist = False
    if os.path.isfile(file_list_path):
      file_list_exist = True

    if not file_list_exist: 
      resultFyle = open(file_list_path,'w')
      # Create Writer Object
      wr = csv.writer(resultFyle, dialect='excel')

    print("Looking for images in '" + dir_name + "'")

    # For quick reading, image_list pre_created as a CSV file
    # Incredible speed up if you have millions of images and work on SSD 
    for extension in extensions:  
      if not file_list_exist:
        file_glob = os.path.join(image_dir, dir_name, '*.' + extension)  
        file_list.extend(gfile.Glob(file_glob))
      else:
        # Read the CSV and convert it into string
        file_list = pd.read_csv(file_list_path)
        file_list = file_list['key'].astype(str).values.tolist()

    if not file_list:  
      print('No files found')  
      continue  
    if len(file_list) < 20:  
      print('WARNING: Folder has less than 20 images, which may cause issues.')  
    elif len(file_list) > MAX_NUM_IMAGES_PER_CLASS:  
      print('WARNING: Folder {} has more than {} images. Some images will '  
            'never be selected.'.format(dir_name, MAX_NUM_IMAGES_PER_CLASS))  
    label_name = re.sub(r'[^a-z0-9]+', ' ', dir_name.lower())

    # Saving File List as CSV
    if not file_list_exist:
      wr.writerow(['key'])
      for item in file_list:
        wr.writerow([item])

    # Obtain the image list for all in the folder
    all_images = []  

    for file_name in file_list:
      counter += 1
      base_name = os.path.basename(file_name)  

      all_images.append(base_name)

    result[label_name] = {  
        'dir': dir_name,  
        'all': all_images,   
    }

  return result, counter
  
def create_inception_graph():
  """"Create a graph from the stored GraphDef file and return a graph object
  Returns：
    A graph contains the inception CNN and all the tensors we needs 
  """ 
  with tf.Session() as sess:  
    model_filename = os.path.join(  
        FLAGS.model_dir, 'classify_image_graph_def.pb')  
    with gfile.FastGFile(model_filename, 'rb') as f:  
      graph_def = tf.GraphDef()  
      graph_def.ParseFromString(f.read())  
      bottleneck_tensor, jpeg_data_tensor, resized_input_tensor = (  
          tf.import_graph_def(graph_def, name='', return_elements=[  
              BOTTLENECK_TENSOR_NAME, JPEG_DATA_TENSOR_NAME,  
              RESIZED_INPUT_TENSOR_NAME]))  
  return sess.graph, bottleneck_tensor, jpeg_data_tensor, resized_input_tensor  
  
def run_bottleneck_on_image(sess, image_data, image_data_tensor, bottleneck_tensor):
  """Runs inference on an image to extract the 'bottleneck' summary layer.
  Args:
    sess: Current active TensorFlow Session.
    image_data: String of raw JPEG data.
    image_data_tensor: Input data layer in the graph.
    bottleneck_tensor: Layer before the final softmax.

  Returns:
    Numpy array of bottleneck values.
  """  
  bottleneck_values = sess.run(bottleneck_tensor, {image_data_tensor: image_data})  
  bottleneck_values = np.squeeze(bottleneck_values)  
  return bottleneck_values  
  

def maybe_download_and_extract():
  """Download and extract the tar file for inception model
     If the model is not in the --model_dir folder, this function will download
     a pre-trained from from tensorflow.org and extract it to --model_dir 
  """    
  dest_directory = FLAGS.model_dir  
  if not os.path.exists(dest_directory):  
    os.makedirs(dest_directory)  
  filename = DATA_URL.split('/')[-1]  
  filepath = os.path.join(dest_directory, filename)  
  if not os.path.exists(filepath):  
  
    def _progress(count, block_size, total_size):  
      sys.stdout.write('\r>> Downloading %s %.1f%%' %  
                       (filename,  
                        float(count * block_size) / float(total_size) * 100.0))  
      sys.stdout.flush()  
  
    filepath, _ = urllib.request.urlretrieve(DATA_URL,  
                                             filepath,  
                                             _progress)  
    print()  
    statinfo = os.stat(filepath)  
    print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')  
  tarfile.open(filepath, 'r:gz').extractall(dest_directory)  

def ensure_dir_exists(dir_name):
  """Makes sure the folder exists on disk.

  Args:
    dir_name: Path string to the folder we want to create.
  """  
  if not os.path.exists(dir_name):  
    os.makedirs(dir_name)


def read_and_decode(filename_queue, class_count):
    """Decode the TFRecords as a queue

       Returns: bottles and labels that decdoed
    """
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(serialized_example, features = {
        "bottleneck": tf.FixedLenFeature([BOTTLENECK_TENSOR_SIZE], tf.float32),
        "image/class/label": tf.FixedLenFeature([class_count], tf.float32),})

    bottleneck = features["bottleneck"]
    label = features["image/class/label"]
    return bottleneck, label

def get_random_cached_bottlenecks_tfrec(how_many, category, total_num_bottleneck, class_count, if_random = True):  
  if(category == 'training'):
    filenames = [os.path.join(TFRECORD_PATH, "flowers_train_0000%d-of-0000"  % i + str(FLAGS.num_shards) + ".tfrecord") for i in range(0, FLAGS.num_shards)]
  elif(category == 'validation'):
    filenames = [os.path.join(TFRECORD_PATH, "flowers_validation_0000%d-of-0000"  % i + str(FLAGS.num_shards) + ".tfrecord") for i in range(0, FLAGS.num_shards)]
  else:
    filenames = [os.path.join(TFRECORD_PATH, "flowers_test_0000%d-of-0000"  % i + str(FLAGS.num_shards) + ".tfrecord") for i in range(0, FLAGS.num_shards)]

  for f in filenames:
    if not tf.gfile.Exists(f):
      raise ValueError("Failed to find file: " + f)
  
  filename_queue = tf.train.string_input_producer(filenames)
  bottleneck, label = read_and_decode(filename_queue, class_count)

  # Whehter to shuffle the queue
  if(if_random):
    min_fraction_of_examples_in_queue = 0.4
    TRAINING_SET_SIZE = total_num_bottleneck - int(total_num_bottleneck *FLAGS.validation_percentage/100) - int(total_num_bottleneck *FLAGS.testing_percentage/100)
    min_queue_examples = int(TRAINING_SET_SIZE * min_fraction_of_examples_in_queue)
    print("Filling queue with %d images before starting to train. " "This will take a few minutes." % min_queue_examples)
    num_preprocess_threads = 1
    bottleneck_batch, label_batch = tf.train.shuffle_batch(
      [bottleneck, label],
      batch_size = how_many,
      num_threads = num_preprocess_threads,
      capacity = min_queue_examples + 3 * how_many,
      min_after_dequeue = min_queue_examples)
    return bottleneck_batch, label_batch
  else:
    bottleneck_batch, label_batch = tf.train.batch(
      [bottleneck, label],
      batch_size = how_many,
      num_threads = 4)
    return bottleneck_batch, label_batch 
  
"""Summaries for Tensorboard for visual process """ 
def variable_summaries(var):   
  with tf.name_scope('summaries'):  
    mean = tf.reduce_mean(var)  
    tf.summary.scalar('mean', mean)  
    with tf.name_scope('stddev'):  
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))  
      tf.summary.scalar('stddev', stddev)  
      tf.summary.scalar('max', tf.reduce_max(var))  
      tf.summary.scalar('min', tf.reduce_min(var))  
      tf.summary.histogram('histogram', var)  
  
def add_final_training_ops(class_count, final_tensor_name, bottleneck_tensor, bottleneck_input, ground_truth_input):

  """
  Here get rid of feed_dict by input the bottleneck_input and ground_truth_input as function argument to speed up

  """
  """Adds a new softmax and fully-connected layer for training and eval.

  We need to retrain the top layer to identify our new classes, so this function
  adds the right operations to the graph, along with some variables to hold the
  weights, and then sets up all the gradients for the backward pass.

  The set up for the softmax and fully-connected layers is based on:
  https://www.tensorflow.org/tutorials/mnist/beginners/index.html


  Returns:
    The tensors for the training and cross entropy results, and tensors for the
    bottleneck input and ground truth input.
  """
  layer_name = 'final_training_ops'  
  with tf.name_scope(layer_name):
    with tf.name_scope('weights'):  
      layer_weights = tf.Variable(tf.truncated_normal([BOTTLENECK_TENSOR_SIZE, class_count], stddev=0.001), name='final_weights')  
      variable_summaries(layer_weights)  
    with tf.name_scope('biases'):  
      layer_biases = tf.Variable(tf.zeros([class_count]), name='final_biases')  
      variable_summaries(layer_biases)  
    with tf.name_scope('Wx_plus_b'):  
      logits = tf.matmul(bottleneck_input, layer_weights) + layer_biases  
      tf.summary.histogram('pre_activations', logits)  
  
  final_tensor = tf.nn.softmax(logits, name=final_tensor_name)  
  tf.summary.histogram('activations', final_tensor)  
  
  with tf.name_scope('cross_entropy'):  
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(  
        labels=ground_truth_input, logits=logits)  
    with tf.name_scope('total'):  
      cross_entropy_mean = tf.reduce_mean(cross_entropy)  
  tf.summary.scalar('cross_entropy', cross_entropy_mean)  
  
  with tf.name_scope('train'):  
    train_step = tf.train.GradientDescentOptimizer(FLAGS.learning_rate).minimize(  
        cross_entropy_mean)  
  
  return (train_step, cross_entropy_mean, bottleneck_input, ground_truth_input, final_tensor)  
  
def add_evaluation_step(result_tensor, ground_truth_tensor):
  """Inserts the operations we need to evaluate the accuracy of our results.

  Args:
    result_tensor: The new final node that produces results.
    ground_truth_tensor: The node we feed ground truth data into.

  Returns:
    Tuple of (evaluation step, prediction).
  """ 
  with tf.name_scope('accuracy'):  
    with tf.name_scope('correct_prediction'):  
      prediction = tf.argmax(result_tensor, 1)  
      correct_prediction = tf.equal(  
      prediction, tf.argmax(ground_truth_tensor, 1))  
    with tf.name_scope('accuracy'):  
      evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))  
      tf.summary.scalar('accuracy', evaluation_step)  
  return evaluation_step, prediction  
  
def main(_):  
  if tf.gfile.Exists(FLAGS.summaries_dir):  
    tf.gfile.DeleteRecursively(FLAGS.summaries_dir)  
    tf.gfile.MakeDirs(FLAGS.summaries_dir)  
  
  # Double Check of model download
  maybe_download_and_extract()  
  graph, bottleneck_tensor, jpeg_data_tensor, resized_image_tensor = (create_inception_graph())  
  
  # Imagelist for class counter and bottleneck files counter
  image_lists, total_bn = create_image_lists(FLAGS.image_dir)
  class_count = len(image_lists.keys())

  if class_count == 0:  
    print('No valid folders of images found at ' + FLAGS.image_dir)  
    return -1  
  if class_count == 1:  
    print('Only one valid folder of images found at ' + FLAGS.image_dir + ' - multiple classes are needed for classification.')  
    return -1  
  
  sess = tf.Session() 

  startTime = time.time() 

  test_batch_size = int(total_bn * FLAGS.testing_percentage/100)

  (train_bottlenecks, train_ground_truth) = get_random_cached_bottlenecks_tfrec(FLAGS.train_batch_size, 'training', total_bn, class_count, if_random = True)
  (validation_bottlenecks, validation_ground_truth) = get_random_cached_bottlenecks_tfrec(FLAGS.validation_batch_size, 'validation', total_bn, class_count, if_random = True)
  (test_bottlenecks, test_ground_truth) = get_random_cached_bottlenecks_tfrec(test_batch_size, 'testing', total_bn, class_count, if_random = False)

  category = tf.placeholder(tf.string, name="category_queue")
  
  (bottleneck_input, ground_truth_input) = tf.case({tf.equal(category, 'training'): lambda:(train_bottlenecks, train_ground_truth), 
      tf.equal(category, 'validation'): lambda: (validation_bottlenecks, validation_ground_truth),
      tf.equal(category, 'testing'): lambda: (test_bottlenecks, test_ground_truth)}, 
      default= lambda: (train_bottlenecks, train_ground_truth), exclusive=True)

  # Add the new layer that we'll be training.
  (train_step, cross_entropy, bottleneck_input, ground_truth_input, final_tensor) = add_final_training_ops(len(image_lists.keys()), 
    FLAGS.final_tensor_name, bottleneck_tensor, bottleneck_input, ground_truth_input)  
  
  # Create the operations we need to evaluate the accuracy of our new layer
  evaluation_step, prediction = add_evaluation_step(final_tensor, ground_truth_input)  
  
  # Merge all the summaries and write them out to the summaries_dir
  merged = tf.summary.merge_all()  
  train_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/train', sess.graph)  
  validation_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/validation')  
  
  # variable initialization  
  init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
  sess.run(init)

  startTime = time.time()
  
  # initialization of the queue runner
  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(coord=coord, sess = sess)

  # Start of training
  for i in range(FLAGS.how_many_training_steps):  

    # Obtain the bottlenecks and ground truth from the queue runner and feed into the graph, and run a training
    # step. Capture training summaries for TensorBoard with the `merged` op.
    train_summary, _ = sess.run([merged, train_step], {category: 'training'})  
    train_writer.add_summary(train_summary, i)  

    is_last_step = (i + 1 == FLAGS.how_many_training_steps)

    # Every so often, print out how well the graph is training.
    if (i % FLAGS.eval_step_interval) == 0 or is_last_step:  
      
      train_accuracy, cross_entropy_value = sess.run([evaluation_step, cross_entropy], {category: 'training'})  
      print('%s: Step %d: Train accuracy = %.1f%%' % (datetime.now(), i,  
                                                      train_accuracy * 100))  
      print('%s: Step %d: Cross entropy = %f' % (datetime.now(), i,  
                                                 cross_entropy_value))  

      # Run a validation step and capture training summaries for TensorBoard with the `merged` op.
      validation_summary, validation_accuracy = sess.run([merged, evaluation_step], {category: 'validation'})  
      validation_writer.add_summary(validation_summary, i)  
      print('%s: Step %d: Validation accuracy = %.1f%% (N=%d)' %  
            (datetime.now(), i, validation_accuracy * 100,  
             len(sess.run(validation_bottlenecks))))  
  
  # We've completed all our training, so run a final test evaluation of some new images we haven't used before.
  test_bottlenecks_value, test_ground_truth_value = sess.run([test_bottlenecks, test_ground_truth], {category: 'testing'})   
    
  # Print out the total time cost
  print("Time taken: %f" % (time.time() - startTime))

  # Write out the trained graph and labels with the weights stored as constants.
  output_graph_def = graph_util.convert_variables_to_constants(  
      sess, graph.as_graph_def(), [FLAGS.final_tensor_name])  
  with gfile.FastGFile(FLAGS.output_graph, 'wb') as f:  
    f.write(output_graph_def.SerializeToString())  
  with gfile.FastGFile(FLAGS.output_labels, 'w') as f:  
    f.write('\n'.join(image_lists.keys()) + '\n')  

  # Stop the queue runner
  coord.request_stop()
  coord.join(threads)
  sess.close()
  
if __name__ == '__main__':  
  parser = argparse.ArgumentParser()  
  parser.add_argument(  
      '--image_dir',  
      type=str,  
      default='',  
      help='Path to folders of labeled images.'  
  )
  parser.add_argument(  
      '--tfrecord_dir',  
      type=str,  
      default='./bottleneck_tf',  
      help='Location of bottleneck in .tfrecord desired'  
  )  
  parser.add_argument(  
      '--output_graph', 
      type=str,  
      default='./output_graph.pb',  
      help='Where to save the trained graph.'  
  )  
  parser.add_argument(  
      '--output_labels',  
      type=str,  
      default='./output_labels.txt',  
      help='Where to save the trained graph\'s labels.'  
  )  
  parser.add_argument(  
      '--summaries_dir',  
      type=str,  
      default='./retrain_logs',  
      help='Where to save summary logs for TensorBoard.'  
  )  
  parser.add_argument(  
      '--how_many_training_steps',  
      type=int,  
      default=4000,  
      help='How many training steps to run before ending.'  
  )  
  parser.add_argument(  
      '--learning_rate',  
      type=float,  
      default=0.01,  
      help='How large a learning rate to use when training.'  
  )  
  parser.add_argument(  
      '--eval_step_interval',  
      type=int,  
      default=10,  
      help='How often to evaluate the training results.'  
  )  
  parser.add_argument(  
      '--train_batch_size',  
      type=int,  
      default=100,  
      help='How many images to train on at a time.'  
  )  
  parser.add_argument(  
      '--test_batch_size',  
      type=int,  
      default=-1,  
      help="""\
      How many images to test on. This test set is only used once, to evaluate 
      the final accuracy of the model after training completes. 
      A value of -1 causes the entire test set to be used, which leads to more 
      stable results across runs.\
      """  
  )  
  parser.add_argument(  
      '--validation_batch_size',  
      type=int,  
      default=100,  
      help="""\
      How many images to use in an evaluation batch. This validation set is 
      used much more often than the test set, and is an early indicator of how 
      accurate the model is during training. 
      A value of -1 causes the entire validation set to be used, which leads to 
      more stable results across training iterations, but may be slower on large 
      training sets.\
      """  
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
      '--print_misclassified_test_images',  
      default=False,  
      help="""
      Whether to print out a list of all misclassified test images.
      """,  
      action='store_true'  
  )  
  parser.add_argument(  
      '--model_dir',  
      type=str,  
      default='./imagenet',  
      help="""\
      Path to classify_image_graph_def.pb, 
      imagenet_synset_to_human_label_map.txt, and 
      imagenet_2012_challenge_label_map_proto.pbtxt.\
      """  
  )  
  parser.add_argument(  
      '--final_tensor_name',  
      type=str,  
      default='final_result',  
      help="""\
      The name of the output classification layer in the retrained graph.\
      """  
  )
  parser.add_argument(  
      '--num_shards',  
      type=int,  
      default=5,  
      help='Number of shards.'  
  )
  FLAGS, unparsed = parser.parse_known_args()  
tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)  