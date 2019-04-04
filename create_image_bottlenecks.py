# Copyright 2015 The TensorFlow Authors. All Rights Reserved.

# Revised by Yanbing Jiang Copyright 2019.

"""Simple transfer learning with image modules from local stored pb model
This file generates all image files's Bottlenecks before Softmax Tensor

Assumes you have a folder containing class-named
subfolders, each full of images for each label. The example folder flower_photos
should have a structure like this:

~/flower_photos/daisy/photo1.jpg
~/flower_photos/daisy/photo2.jpg
...
~/flower_photos/rose/anotherphoto77.jpg
...
~/flower_photos/sunflower/somepicture.jpg

The subfolder names are important, since they define what label is applied to
each image, but the filenames themselves don't matter. (For a working example,
download http://download.tensorflow.org/example_images/flower_photos.tgz
and run  tar xzf flower_photos.tgz  to unpack it.)

This file also downloads the Inception-V3 pb frozen model before generating bottlencks
"""
from __future__ import absolute_import  
from __future__ import division  
from __future__ import print_function  
  
import argparse  
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
  
# Underneath are all the parameters, which are corresponding to the Inception V3 Model
# Tensors Including Tensor names, sizes (e.g. Image sizes)
# If you want to use this script to work on your own model, you may need to update those
# tensor name and size

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

BOTTLENECK_PATH = '.\\v3_move_direct\\bottleneck'
  
def create_image_lists(image_dir):  
  """Builds a list of training images from the file system.

  Analyzes the sub folders in the image directory, splits them into stable
  training, testing, and validation sets, and returns a data structure
  describing the lists of images for each label and their paths.

  Args:
    image_dir: String path to a folder containing subfolders of images.
    testing_percentage: Integer percentage of the images to reserve for tests.
    validation_percentage: Integer percentage of images reserved for validation.

  Returns:
    An OrderedDict containing an entry for each label subfolder, with images
    split into training, testing, and validation sets within each label.
    The order of items defines the class indices.
  """
  if not gfile.Exists(image_dir):  
    print("Image directory '" + image_dir + "' not found.")  
    return None  
  result = {}  
  sub_dirs = [x[0] for x in gfile.Walk(image_dir)]

  # The root directory comes first, so skip it.
  is_root_dir = True  
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
      base_name = os.path.basename(file_name)  

      all_images.append(base_name)

    result[label_name] = {  
        'dir': dir_name,  
        'all': all_images,   
    }

  return result  
  
  
def get_image_path(image_lists, label_name, index, image_dir, category):  
  """Returns a path to an image for a label at the given index.

  Args:
    image_lists: OrderedDict of training images for each label.
    label_name: Label string we want to get an image for.
    index: Int offset of the image we want. This will be moduloed by the
    available number of images for the label, so it can be arbitrarily large.
    image_dir: Root folder string of the subfolders containing the training
    images.
    category: Name string of set to pull images from - training, testing, or
    validation.

  Returns:
    File system path string to an image that meets the requested parameters.

  """
  if label_name not in image_lists:  
    tf.logging.fatal('Label does not exist %s.', label_name)  
  label_lists = image_lists[label_name]  
  if category not in label_lists:  
    tf.logging.fatal('Category does not exist %s.', category)  
  category_list = label_lists[category]  
  if not category_list:  
    tf.logging.fatal('Label %s has no images in the category %s.',  
                     label_name, category)  
  mod_index = index % len(category_list)  
  base_name = category_list[mod_index]  
  sub_dir = label_lists['dir']  
  full_path = os.path.join(image_dir, sub_dir, base_name)  
  return full_path  
  
  
def get_bottleneck_path(image_lists, label_name, index, bottleneck_dir,  
                        category):  
  """Returns a path to a bottleneck file for a label at the given index.

  Args:
    image_lists: OrderedDict of training images for each label.
    label_name: Label string we want to get an image for.
    index: Integer offset of the image we want. This will be moduloed by the
    available number of images for the label, so it can be arbitrarily large.
    bottleneck_dir: Folder string holding cached files of bottleneck values.
    category: Name string of set to pull images from - training, testing, or
    validation.
    module_name: The name of the image module being used.

  Returns:
    File system path string to an image that meets the requested parameters.
  """
  return get_image_path(image_lists, label_name, index, bottleneck_dir, category) + '.txt'  

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
  
  
def create_bottleneck_file(bottleneck_path, image_lists, label_name, index, image_dir, category, sess, jpeg_data_tensor, bottleneck_tensor):  
  print('Creating bottleneck at ' + bottleneck_path)  
  image_path = get_image_path(image_lists, label_name, index, image_dir, category)  
  if not gfile.Exists(image_path):  
    tf.logging.fatal('File does not exist %s', image_path)  
  image_data = gfile.FastGFile(image_path, 'rb').read()  
  bottleneck_values = run_bottleneck_on_image(sess, image_data, jpeg_data_tensor, bottleneck_tensor)  
  bottleneck_string = ','.join(str(x) for x in bottleneck_values)  
  with open(bottleneck_path, 'w') as bottleneck_file:  
    bottleneck_file.write(bottleneck_string)

def get_or_create_bottleneck(sess, image_lists, label_name, index, image_dir,category, bottleneck_dir, jpeg_data_tensor, bottleneck_tensor):
  """Retrieves or calculates bottleneck values for an image.

  If a cached version of the bottleneck data exists on-disk, return that,
  otherwise calculate the data and save it to disk for future use.

  Args:
    sess: The current active TensorFlow Session.
    image_lists: OrderedDict of training images for each label.
    label_name: Label string we want to get an image for.
    index: Integer offset of the image we want. This will be modulo-ed by the
    image_dir: Root folder string of the subfolders containing the training images.
    category: Name string of which set to pull images from - training, testing, or validation.
    bottleneck_dir: Folder string holding cached files of bottleneck values.
    jpeg_data_tensor: The tensor to feed loaded jpeg data into.
    bottleneck_tensor: The penultimate output layer of the graph.
    
  Returns:
    Numpy array of values produced by the bottleneck layer for the image.
  """

  label_lists = image_lists[label_name]  
  sub_dir = label_lists['dir']  
  sub_dir_path = os.path.join(bottleneck_dir, sub_dir)  
  ensure_dir_exists(sub_dir_path)  
  bottleneck_path = get_bottleneck_path(image_lists, label_name, index, bottleneck_dir, category)  
  if not os.path.exists(bottleneck_path):  
    create_bottleneck_file(bottleneck_path, image_lists, label_name, index, image_dir, category, sess, jpeg_data_tensor, bottleneck_tensor)  
  with open(bottleneck_path, 'r') as bottleneck_file:  
    bottleneck_string = bottleneck_file.read()  
  did_hit_error = False  
  try:  
    bottleneck_values = [float(x) for x in bottleneck_string.split(',')]  
  except:  
    print("Invalid float found, recreating bottleneck")  
    did_hit_error = True  
  if did_hit_error:  
    create_bottleneck_file(bottleneck_path, image_lists, label_name, index, image_dir, category, sess, jpeg_data_tensor, bottleneck_tensor)  
    with open(bottleneck_path, 'r') as bottleneck_file:  
      bottleneck_string = bottleneck_file.read()  
    # Allow exceptions to propagate here, since they shouldn't happen after a fresh creation
    bottleneck_values = [float(x) for x in bottleneck_string.split(',')]  
  return bottleneck_values  

  
def cache_bottlenecks(sess, image_lists, image_dir, bottleneck_dir, jpeg_data_tensor, bottleneck_tensor):
  """Ensures all the training, testing, and validation bottlenecks are cached.

  Because we're likely to read the same image multiple times (if there are no
  distortions applied during training) it can speed things up a lot if we
  calculate the bottleneck layer values once for each image during
  preprocessing, and then just read those cached values repeatedly during
  training. Here we go through all the images we've found, calculate those
  values, and save them off.

  Args:
    sess: The current active TensorFlow Session.
    image_lists: OrderedDict of training images for each label.
    image_dir: Root folder string of the subfolders containing the training
    images.
    bottleneck_dir: Folder string holding cached files of bottleneck values.
    jpeg_data_tensor: Input tensor for jpeg data from file.
    bottleneck_tensor: The penultimate output layer of the graph.

  Returns:
    Nothing.
  """
  how_many_bottlenecks = 0  
  ensure_dir_exists(bottleneck_dir)  
  for label_name, label_lists in image_lists.items():
    category = 'all'
    category_list = label_lists[category]  
    for index, unused_base_name in enumerate(category_list):  
      get_or_create_bottleneck(sess, image_lists, label_name, index,  
                               image_dir, category, bottleneck_dir,  
                               jpeg_data_tensor, bottleneck_tensor)  

      how_many_bottlenecks += 1  
      if how_many_bottlenecks % 100 == 0:  
        print(str(how_many_bottlenecks) + ' bottleneck files created.')  
  

def main(_):  
  # Download and extract the inception v3 graph
  maybe_download_and_extract()  
  graph, bottleneck_tensor, jpeg_data_tensor, resized_image_tensor = (create_inception_graph())  
  
  # Look up the file list and create image list
  image_lists = create_image_lists(FLAGS.image_dir)  
  class_count = len(image_lists.keys())

  # special number of classes solution
  if class_count == 0:  
    print('No valid folders of images found at ' + FLAGS.image_dir)  
    return -1  
  if class_count == 1:  
    print('Only one valid folder of images found at ' + FLAGS.image_dir + ' - multiple classes are needed for classification.')  
    return -1  

  # Session starts  
  sess = tf.Session() 

  cache_bottlenecks(sess, image_lists, FLAGS.image_dir, FLAGS.bottleneck_dir, jpeg_data_tensor, bottleneck_tensor)
  print("\nBottlenecks' Generation is done, located at ", FLAGS.bottleneck_dir)

  # Print out the time needed to create bottlenecks
  print("Time taken: %f" % (time.time() - startTime))

  
if __name__ == '__main__':  
  parser = argparse.ArgumentParser()  
  parser.add_argument(  
      '--image_dir',  
      type=str,  
      default='',  
      help='Path to folders of labeled images.'  
  )   
  parser.add_argument(  
      '--bottleneck_dir',  
      type=str,  
      default='./bottleneck',  
      help='Path to cache bottleneck layer values as files.'  
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
  FLAGS, unparsed = parser.parse_known_args()  
tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)  