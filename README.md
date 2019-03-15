# inception V3 Retrain Fast Input With TFrecord QueueRunner

Transfer learning for inception V3 is popular [tutorial](https://www.tensorflow.org/hub/tutorials/image_retraining) from offcial Tensorflow, named as  [retrain.py](https://github.com/tensorflow/hub/blob/master/examples/image_retraining/retrain.py), which is an example script that shows how one can adapt a pretrained network for other classification problems. 

This repository improved with quick data read and process to make the training more efficient by utilizing [TFRecord](https://www.tensorflow.org/tutorials/load_data/tf_records).

## Compatibility
Tested under Tensorflow 1.6.0/1.9.0/1.12.0 with GPU support under Python 3.6.0 under Windows.

## Description of the Code
This repository splited the training process into three steps:
* Create the bottlenecks in .txt format (create_image_bottleneck.py)
* Convert bottlenecks into .tfrecord format (tfrec_data.py / tfrec_data_threads.py)
* Training and Evaluation Process nearly identical to [retrain.py](https://github.com/tensorflow/hub/blob/master/examples/image_retraining/retrain.py)

## Getting Started and Usage

### Requirements
* Tensorflow with Tensorboard, Numpy, six, Panda
* Followe the file named requirements.txt
```shell
pip3 install -r requirements.txt # For Python3, 'pip3' may vary to 'pip' depends on your machine
```

### Data Preparation
Follow http://download.tensorflow.org/example_images/flower_photos.tgz to download the flower images and extract.

### Create Bottlenecks
```shell
python create_image_bottlenecks.py --image_dir=<path to training images> --bottleneck_dir=<Path to cache bottleneck>
```
**Example**
```shell
python create_image_bottlenecks.py  # Run as Default
python create_image_bottlenecks.py --image_dir=./flowers_photo --bottleneck_dir=./bottleneck # Run with personalized settings
```

### Convert Bottlenecks to TFRecords Format
When creating the TFRecords, the dataset needs to be splited into training set, validation set and testing set as well. The same work as 'create_image_list' function in [retrain.py](https://github.com/tensorflow/hub/blob/master/examples/image_retraining/retrain.py). Thus, percentage of validation set and percentage of testing set needs to specified. Number of shards is a parameter to specify how many tfrecord file you would like to have to one set. For example, if num_shards = 5, it will divide training set into 5 tfrecord file evenly, so as validation set and testing set.

* *Option 1*: Covert Data Sequetially, for flower set is fast enough
```shell
python tfrec_data.py --bottleneck_dir=<Path to cache bottleneck> --tfrecord_dir=<Path to store tfrecord bottleneck> --validation_percentage=<validation %> --testing_percentage=<testing %> --num_shards=<number of tfrecords files>
```
**Example**
```shell
python tfrec_data.py  # Run as Default with validation 10% testing 10% and 5 shards
python tfrec_data.py --bottleneck_dir=./bottleneck --tfrecord_dir=./bottleneck_tf --validation_percentage=10 --10 --num_shards=5
```

* *Option 2*: Covert Data in multi-threads behavious, for larger dataset convertion
Note that number of shards mod number of threads needs to qual to zero (num_shards % num_threads == 0) to make it work.
```shell
python tfrec_data_thread.py --bottleneck_dir=<Path to cache bottleneck> --tfrecord_dir=<Path to store tfrecord bottleneck> --validation_percentage=<validation %> --testing_percentage=<testing %> --num_shards=<number of tfrecords files> --num_threads=<number of threads>
```
**Example**
```shell
python tfrec_data_thread.py  # Run as Default with validation 10% testing 10% , 5 shards and 5 threads
python tfrec_data_thread.py --bottleneck_dir=./bottleneck --tfrecord_dir=./bottleneck_tf --validation_percentage=10 --testing_percentage=10 --num_shards=5 --num_threads=5
```

### Start Training!
This training will utlize the TFRecord data only instead of .txt file as stated. As introcued in the official tutorial, only the last fully connected layer is traine. In order to train, learning rate, training steps, validation percentage, testing percentage and so on need to be specified.

```shell
python retrain_attempt_v3_tfrec.py --image_dir=<Path to original images> --tfrecord_dir=<Path to store tfrecord bottleneck> --output_graph=<Where to save the trained graph> --output_labels=<Where to save the trained graph's labels> --how_many_training_steps =<# of training steps> --learning_rate=<learning rate> --validation_percentage=<validation %> --testing_percentage=<testing %> --num_shards=<number of tfrecords files>
```
**Example**
```shell
python retrain_attempt_v3_tfrec.py --image_dir=./flower_photos  # Run as Default flower photos in the same directory
python retrain_attempt_v3_tfrec.py --image_dir=./flower_photos --validation_percentage=10 --testing_percentage=10 --num_shards=5 # Run as Default flower photos in the same directory with specified validation/testing percentage and number of shards
```

## Testing Results

**Environment:** Windows 8.1 / CPU Intel(R) Core(TM) i7-5500U / GPU GeForce 940M

**Settings:** 4000 Training Steps, 10% Validation, 10% Testing, Learning Rate = 0.001, periodical eval step size = 10
* Original [retrain.py](https://github.com/tensorflow/hub/blob/master/examples/image_retraining/retrain.py): 728 seconds
* This repository: &nbsp;&nbsp;&nbsp;&nbsp; **163 seconds**

## Author

* **Yanbing Jiang**

## License
This project is licensed under the MIT License - see the [LICENSE.md](https://github.com/YanbingJiang/inception-V3-Retrain-Fast-With-TFrecord/blob/master/LICENSE) file for details
