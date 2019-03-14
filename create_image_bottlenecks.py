# Copyright 2015 The TensorFlow Authors. All Rights Reserved.

# Revised by Yanbing Jiang Copyright 2019

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
  
# 这些是所有的参数，我们使用这些参数绑定到特定的Inceptionv3_move_direct模型结构。  
#这些包括张量名称和它们的尺寸。如果您想使此脚本与其他模型相适应，您将需要更新这些映射你在网络中使用的值。  
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
  """从文件系统生成训练图像列表。分析图像目录中的子文件夹，将其分割成稳定的训练、测试和验证集，并返回数据结构，描述每个标签及其路径的图像列表。 
  Args： 
    image_dir：一个包含图片子文件夹的文件夹的字符串路径。 
    testing_percentage：预留测试图像的整数百分比。 
    validation_percentage：预留验证图像的整数百分比。 
  Returns： 
    一个字典包含进入每一个标签的子文件夹和分割到每个标签的训练，测试和验证集的图像。 
  """  
  if not gfile.Exists(image_dir):  
    print("Image directory '" + image_dir + "' not found.")  
    return None  
  result = {}  
  sub_dirs = [x[0] for x in gfile.Walk(image_dir)]  
  # 首先进入根目录，所以先跳过它。  
  is_root_dir = True  
  for sub_dir in sub_dirs:  
    if is_root_dir:  
      is_root_dir = False  
      continue  
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
    for extension in extensions:  
      # file_glob = os.path.join(image_dir, dir_name, '*.' + extension)  
      # file_list.extend(gfile.Glob(file_glob))
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
  """"返回给定索引中标签的图像路径。 
  Args： 
    image_lists：训练图像每个标签的词典。 
    label_name：我们想得到的一个图像的标签字符串。 
    index：我们想要图像的Int 偏移量。这将以标签的可用的图像数为模，因此它可以任意大。 
    image_dir：包含训练图像的子文件夹的根文件夹字符串。 
    category：从图像训练、测试或验证集提取的图像的字符串名称。 
  Returns： 
    将文件系统路径字符串映射到符合要求参数的图像。 
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
  """"返回给定索引中的标签的瓶颈文件的路径。 
  Args： 
    image_lists：训练图像每个标签的词典。 
    label_name：我们想得到的一个图像的标签字符串。 
    index：我们想要图像的Int 偏移量。这将以标签的可用的图像数为模，因此它可以任意大。 
    bottleneck_dir：文件夹字符串保持缓存文件的瓶颈值。 
    category：从图像训练、测试或验证集提取的图像的字符串名称。 
  Returns： 
    将文件系统路径字符串映射到符合要求参数的图像。 
  """  
  return get_image_path(image_lists, label_name, index, bottleneck_dir,  
                        category) + '.txt'  
  
""""从保存的GraphDef文件创建一个图像并返回一个图像对象。 
  Returns： 
    我们将操作的持有训练的Inception网络和各种张量的图像。 
"""    
def create_inception_graph():  
  
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
  
  """在图像上运行推理以提取“瓶颈”摘要层。 
  Args： 
    sess：当前活动的tensorflow会话。 
    image_data：原JPEG数据字符串。 
    image_data_tensor：图中的输入数据层。 
    bottleneck_tensor：最后一个softmax之前的层。 
  Returns： 
    NumPy数组的瓶颈值。 
  """ 
def run_bottleneck_on_image(sess, image_data, image_data_tensor, bottleneck_tensor):  
  bottleneck_values = sess.run(bottleneck_tensor, {image_data_tensor: image_data})  
  bottleneck_values = np.squeeze(bottleneck_values)  
  return bottleneck_values  
  
"""下载并提取模型的tar文件。 
    如果我们使用的pretrained模型已经不存在，这个函数会从tensorflow.org网站下载它并解压缩到一个目录。 
  """   
def maybe_download_and_extract():  
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
  
"""确保文件夹已经在磁盘上存在。 
  Args: 
    dir_name: 我们想创建的文件夹路径的字符串。 
  """  
def ensure_dir_exists(dir_name):  
  if not os.path.exists(dir_name):  
    os.makedirs(dir_name)  
  
  
def create_bottleneck_file(bottleneck_path, image_lists, label_name, index,  
                           image_dir, category, sess, jpeg_data_tensor, bottleneck_tensor):  
  print('Creating bottleneck at ' + bottleneck_path)  
  image_path = get_image_path(image_lists, label_name, index, image_dir, category)  
  if not gfile.Exists(image_path):  
    tf.logging.fatal('File does not exist %s', image_path)  
  image_data = gfile.FastGFile(image_path, 'rb').read()  
  bottleneck_values = run_bottleneck_on_image(sess, image_data, jpeg_data_tensor, bottleneck_tensor)  
  bottleneck_string = ','.join(str(x) for x in bottleneck_values)  
  with open(bottleneck_path, 'w') as bottleneck_file:  
    bottleneck_file.write(bottleneck_string)

    """检索或计算图像的瓶颈值。 
   如果磁盘上存在瓶颈数据的缓存版本，则返回，否则计算数据并将其保存到磁盘以备将来使用。 
 Args: 
   sess:当前活动的tensorflow会话。 
   image_lists：每个标签的训练图像的词典。 
   label_name：我们想得到一个图像的标签字符串。 
   index：我们想要的图像的整数偏移量。这将以标签图像的可用数为模，所以它可以任意大。 
   image_dir：包含训练图像的子文件夹的根文件夹字符串。 
   category：从图像训练、测试或验证集提取的图像的字符串名称。 
   bottleneck_dir：保存着缓存文件瓶颈值的文件夹字符串。 
   jpeg_data_tensor：满足加载的JPEG数据进入的张量。 
   bottleneck_tensor：瓶颈值的输出张量。 
 Returns: 
   通过图像的瓶颈层产生的NumPy数组值。 
  """  
  
def get_or_create_bottleneck(sess, image_lists, label_name, index, image_dir,category, bottleneck_dir, jpeg_data_tensor, bottleneck_tensor):
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
    #允许在这里传递异常，因为异常不应该发生在一个新的bottleneck创建之后。  
    bottleneck_values = [float(x) for x in bottleneck_string.split(',')]  
  return bottleneck_values  


"""确保所有的训练，测试和验证瓶颈被缓存。 
  因为我们可能会多次读取同一个图像（如果在训练中没有应用扭曲）。如果我们每个图像预处理期间的瓶颈层值只计算一次，在训练时只需反复读取这些缓存值，能大幅的加快速度。在这里，我们检测所有发现的图像，计算那些值，并保存。 
  Args： 
    sess：当前活动的tensorflow会话。 
    image_lists：每个标签的训练图像的词典。 
    image_dir：包含训练图像的子文件夹的根文件夹字符串。 
    bottleneck_dir：保存着缓存文件瓶颈值的文件夹字符串。 
    jpeg_data_tensor：从文件输入的JPEG数据的张量。 
    bottleneck_tensor：图中的倒数第二输出层。 
  Returns: 
   无。 
  """  
def cache_bottlenecks(sess, image_lists, image_dir, bottleneck_dir, jpeg_data_tensor, bottleneck_tensor): 
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
  
#设置我们写入TensorBoard摘要的目录。  
def main(_):  
  #设置预训练图像。  
  maybe_download_and_extract()  
  graph, bottleneck_tensor, jpeg_data_tensor, resized_image_tensor = (create_inception_graph())  
  
  #查看文件夹结构，创建所有图像的列表。  
  image_lists = create_image_lists(FLAGS.image_dir)  
  class_count = len(image_lists.keys())

  if class_count == 0:  
    print('No valid folders of images found at ' + FLAGS.image_dir)  
    return -1  
  if class_count == 1:  
    print('Only one valid folder of images found at ' + FLAGS.image_dir + ' - multiple classes are needed for classification.')  
    return -1  
  
  #看命令行标记是否意味着我们应用任何扭曲操作。  
  # do_distort_images = should_distort_images(FLAGS.flip_left_right, FLAGS.random_crop, FLAGS.random_scale,FLAGS.random_brightness)
  sess = tf.Session() 

  startTime = time.time() 
  # 我们将应用扭曲，因此设置我们需要的操作
  # if do_distort_images:  
  #   distorted_jpeg_data_tensor, distorted_image_tensor = add_input_distortions(FLAGS.flip_left_right, FLAGS.random_crop, FLAGS.random_scale, FLAGS.random_brightness)  
  # else:  
    #我们确定计算bottleneck图像总结并缓存在磁盘上。  
  cache_bottlenecks(sess, image_lists, FLAGS.image_dir, FLAGS.bottleneck_dir, jpeg_data_tensor, bottleneck_tensor)
  print("\nBottlenecks' Generation is done, located at ", FLAGS.bottleneck_dir)
  print("Time taken: %f" % (time.time() - startTime))
    # exit(0)
  
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