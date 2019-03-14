# inception V3 Retrain Fast Input With TFrecord QueueRunner

Transfer learning for inception V3 is popular tutorial from offcial Tensorflow, named as  [retrain.py](https://github.com/tensorflow/hub/blob/master/examples/image_retraining/retrain.py), which is an example script that shows how one can adapt a pretrained network for other classification problems. 

This repository improved with quick data read and process to make the training more efficient by utilizing [TFRecord](https://www.tensorflow.org/tutorials/load_data/tf_records).

## Compatibility
Tested under Tensorflow 1.6.0/1.9.0/1.12.0 with GPU support under Python 3.6.0.

## Description of the Code
This repository splited the training process into three steps:
* Create the bottlenecks in .txt format (create_image_bottleneck.py)
* Convert bottlenecks into .tfrecord format (tfrec_data.py / tfrec_data_threads.py)
* Training and Evaluation Process nearly identical to [retrain.py](https://github.com/tensorflow/hub/blob/master/examples/image_retraining/retrain.py)

## Requirements
* Tensorflow with Tensorboard, Numpy, six, Panda
* Followe the file named requirements.txt
```shell
pip3 install -r requirements.txt # For Python 3, 'pip3' may vary to 'pip' depends on your machine
```
