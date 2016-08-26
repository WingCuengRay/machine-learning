# Project 5: Deep Learning Capstone Project. Decoding a Sequence of Digits in Real World Photos.

This project is a tensorflow implementation of a Convolution Nueral Network to decode a sequence of digits in the realworld using 
trained on the SVHN dataset. This project is partial fulfillment of the Udacity Machine Learning nanodegree program. 

## Installation
```
git clone https://github.com/bdiesel/machine-learning.git
cd /machine-learning/projects/capstone
python svhn_data.py
```


## Libraries and Dependicies 
The following Python 2.7 libraries are required:
* h5py
* matplotlib
* numpy
* PIL
* tensorflow
* scipy
* six
* sklearn



##Step 1: Download Data
Download and preprocess the SVHN data using the svhn_data.py utility.
`python svhn_data.py`

This should generate a data folder data\svhn with two sub-directories cropped and full

The cropped cropped directory should contain 2 newly downloaded .mat files amd 6 numpy file for each dataset which wil be used for training.

The full directory should contain two sub directories test and full which contain png images of various sizes.


##Step 2: Train your own models
First train the classifer. The weights generated here will be resused in the training of the multi-digit reader.

`python train_classifier.py`

This should generate a tensorflow checkpoint file:

`classifier.ckpt`

Next train the multi-digit reader

`python train_regressor.py`

This should generate a tensorflow checkpoint file:

`regression.ckpt`


## Usage.

The single digit reader for an image file 1.png `python single_digit_reader.py 1.png`

The multi digit reader for an image file 123.png `python mulit_digit_reader.py 123.png`

