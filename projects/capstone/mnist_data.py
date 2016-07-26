import os
import struct
import sys
import numpy as np
import gzip
import glob

from scipy import ndimage
from sklearn.cross_validation import train_test_split
from six.moves.urllib.request import urlretrieve

'''for debugging'''
from pdb import set_trace as bp
#import matplotlib.pyplot as plt
#from random import randint

path = "data/minst_digit"
pixel_depth = 255
NUM_LABELS = 10
last_percent_reported = None
BASE_URL = "http://yann.lecun.com/exdb/mnist/"

def labels_to_one_hot(labels):
  labels = (np.arange(NUM_LABELS) == labels[:,None]).astype(np.float32)
  return labels

def get_dataset_by_name(data_set_name):
	if not os.path.exists(path):
		os.makedirs(path)
	if data_set_name == "training":	
		img_file = "train-images-idx3-ubyte"
		lbl_file = "train-labels-idx1-ubyte"
	elif data_set_name == "testing":
		img_file = "t10k-images-idx3-ubyte"
		lbl_file = "t10k-labels-idx1-ubyte"
	else:
		raise ValueError, "dataset must be 'testing' or 'training'"
	data_file_path = os.path.join(path, img_file)
	label_file_path = os.path.join(path, lbl_file)
	if os.path.exists(data_file_path):
		data_file = data_file_path
	else:
		data_file_gz = download_mnist_data(img_file)
		data_file = extract_gz(data_file_gz)
	if os.path.exists(label_file_path):
		label_file = label_file_path
	else:
		label_file_gz = download_mnist_data(lbl_file)
		label_file = extract_gz(label_file_gz)
	return(data_file, label_file)

def download_progress(count, block_size, total_size):
	global last_percent_reported
	percent = int(count * block_size * 100 / total_size)
	if last_percent_reported != percent:
		if percent % 5  == 0:
			sys.stdout.write("%s%%" % percent)
			sys.stdout.flush()
		else:
			sys.stdout.write(".")
			sys.stdout.flush()
		last_percent_reported = percent

def download_mnist_data(filename, force=False):
	file_name_with_extension = filename+".gz"
	print "Attempting to download", filename
	saved_file, _ = urlretrieve(BASE_URL + file_name_with_extension, os.path.join(path, file_name_with_extension), reporthook=download_progress)
	print("\nDownload Complete!")
	statinfo = os.stat(saved_file)
	if statinfo.st_size == get_expected_bytes(filename):
		print("Found and verified", saved_file)
		return saved_file
	else:
		raise Exception("Failed to verify " + filename)

def extract_gz(saved_file):
	print("Extracting:", saved_file)
	base = os.path.basename(saved_file)
	dest_name = os.path.join(path, base[:-3])
	with gzip.open(saved_file, 'rb') as infile:
		with open(dest_name, 'wb') as outfile:
			for line in infile:
				outfile.write(line)
	print("Success")
	return dest_name			

def get_expected_bytes(filename):
	if filename == "train-images-idx3-ubyte":
		byte_size = 9912422
	elif filename == "train-labels-idx1-ubyte":
		byte_size = 28881
	elif filename == "t10k-images-idx3-ubyte":
		byte_size = 1648877
	elif filename == "t10k-labels-idx1-ubyte":
		byte_size = 4542
	else:
		raise Exception("Invalid file name" + filename)
	return byte_size

def read_image_file_to_array(file_name):
	file = open(file_name, 'rb')
	images_array = process_img_file(file)
	file.close()
	return images_array

def process_img_file(file):
	magic, num, rows, cols = struct.unpack(">IIII", file.read(16))
	data_array = (np.fromfile(file, dtype=np.uint8).reshape(num, rows, cols).astype(np.float32) - pixel_depth / 2) / pixel_depth
	print("das1:", data_array.shape)
	data_array = data_array.reshape((-1, rows*cols*1)).astype(np.float32)
	print("das2:", data_array.shape)
	return data_array

def read_label_file_to_array(file_name):
	file = open(file_name, 'rb')
	magic, n = struct.unpack('>II', file.read(8))
	labels = np.fromfile(file, dtype=np.uint8)
	labels = labels_to_one_hot(labels)
	file.close()
	return labels

def creat_minst(data_set_name):
	""" Read MNIST from ubyte files.
    Parameters
    ----------
    data_set_name : str either training or testing
        path to the test or train MNIST ubyte file
    labels_path : str
        path to the test or train MNIST class labels file
    Returns
    --------
    images : [n_samples, n_pixels] numpy.array
        Pixel values of the images.
    labels : [n_samples] numpy array
        Target class labels
    """
	image_file, label_file = get_dataset_by_name(data_set_name)
	lbl_data = read_label_file_to_array(label_file)
	img_data = read_image_file_to_array(image_file)
	if len(lbl_data) != len(img_data):
		raise Exception('Label array length is not equal to Data Array length: %s s' % str(lbl_data.shape), str(img_data.shape))
	return img_data, lbl_data

def train_validation_spit(train_dataset, train_labels):
  train_dataset, validation_dataset, train_labels, validation_labels = train_test_split(train_dataset, train_labels, test_size=0.33, random_state = 42)
  return train_dataset, validation_dataset, train_labels, validation_labels

def write_npy_file(data_array, lbl_array, data_set_name):
	np.save(os.path.join(path, data_set_name+'_imgs.npy'), data_array)
	print 'Saving to %s_imgs.npy file done. Data size is: %s' %((data_set_name), data_array.shape)
	np.save(os.path.join(path, data_set_name+'_labels.npy'), lbl_array)
	print 'Saving to %s_labels.npy file done. Contains %d rows' %(data_set_name, len(lbl_array))

def load_mnist_data(data_type):
	if data_type == "training":
		imgs = np.load(os.path.join(path, 'training_imgs.npy'))
		labels = np.load(os.path.join(path, 'training_labels.npy'))
	elif data_type == "testing":
		imgs = np.load(os.path.join(path, 'testing_imgs.npy'))
		labels = np.load(os.path.join(path, 'testing_labels.npy'))
	elif(data_type == "validation"):
		imgs = np.load(os.path.join(path, 'validation_imgs.npy'))
		labels = np.load(os.path.join(path, 'validation_labels.npy'))
	else:
		raise Exception("Data Set not found!")
	return imgs, labels

def show_test(data_set_name):
	imgs_train, labels_train = load_minst_data("training")
	random_idx = randint(0, len(imgs_train))
	plt.imshow(imgs_train[random_idx], cmap="Greys")
	plt.show()

if __name__ == '__main__':
	train_data, train_labels = creat_minst("training")
	test_data, test_labels = creat_minst("testing")
	train_data, valid_data, train_labels, valid_labels = train_validation_spit(train_data, train_labels)
	write_npy_file(train_data, train_labels, "training")
	write_npy_file(valid_data, valid_labels, "validation")
	write_npy_file(test_data, test_labels, "testing")