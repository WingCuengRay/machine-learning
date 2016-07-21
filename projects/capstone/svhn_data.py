import os, struct
import numpy as np
import scipy
import h5py
from scipy.io import loadmat
# from scipy import ndimage
from sklearn.cross_validation import train_test_split
from itertools import product

# '''for debugging'''
from pdb import set_trace as bp
import matplotlib.pyplot as plt
from random import randint

data_path = "data"
cropped_data_path = data_path+"/cropped"
full = data_path+"/full"
FORMAT_2_FILES = ['{}_32x32.mat'.format(s) for s in ['train', 'test', 'extra']]
FORMAT_2_TRAIN_FILE, FORMAT_2_TEST_FILE, FORMAT_2_EXTRA_FILE = FORMAT_2_FILES
PIXEL_DEPTH = 255
NUM_LABELS = 10


def read_data_file(file_name):
	file = open(file_name, 'rb')
	data = process_data_file(file)
	file.close()
	return data

def convert_img_array_vec(img_array):
	rows = img_array.shape[0]
	cols = img_array.shape[1]
	chans = img_array.shape[2]
	num = img_array.shape[3]
	data_array = (img_array.reshape(num, rows, cols, chans).astype(np.float32) - PIXEL_DEPTH / 2) / PIXEL_DEPTH
	data_array = data_array.reshape((-1, rows, cols, chans)).astype(np.float32)
	#data_array = data_array.reshape((-1, rows*cols*chans)).astype(np.float32)
	return data_array

def convert_labels_to_one_hot(labels):
	labels = (np.arange(NUM_LABELS) == labels[:,None]).astype(np.float32)
	return labels

def sainty_check(imgs, labels):
	ri = randint(0,len(labels)-1)
	print(labels[ri])
	plt.imshow(imgs[:,:,:,ri])
	plt.show()

def process_data_file(file):
	data = loadmat(file)
	imgs = data['X']
	labels = data['y'].flatten()
	labels[labels==10] = 0 #Fix weird labeling in dataset
	labels_one_hot = convert_labels_to_one_hot(labels)
	#sainty_check(imgs, labels_one_hot)
	img_array = convert_img_array_vec(imgs)
	return img_array, labels_one_hot

def create_svhn(dataset):
	if dataset == "train":
		data_file = os.path.join(cropped_data_path, "train_32x32.mat")
	elif dataset == "test":
		data_file = os.path.join(cropped_data_path, "test_32x32.mat")
	elif dataset == "extra":
		data_file = os.path.join(cropped_data_path, "extra_32x32.mat")
	else:
		raise NotImplementedError('dataset must be either train or test')
	X, y = read_data_file(data_file)
	return X, y

def train_validation_spit(train_dataset, train_labels):
	train_dataset, validation_dataset, train_labels, validation_labels = train_test_split(train_dataset, train_labels, test_size=0.33, random_state = 42)
	return train_dataset, validation_dataset, train_labels, validation_labels

def write_npy_file(data_array, lbl_array, data_set_name):
	np.save(os.path.join(cropped_data_path,data_set_name+'_svhn_imgs.npy'), data_array)
	print 'Saving to %s_svhn_imgs.npy file done. Data size is: %s' %((data_set_name), data_array.shape)
	np.save(os.path.join(cropped_data_path,data_set_name+'_svhn_labels.npy'), lbl_array)
	print 'Saving to %s_svhn_labels.npy file done. Contains %d rows' %(data_set_name, len(lbl_array))

def load_svhn_data(data_type):
	if data_type == "training":
		imgs = np.load(os.path.join(cropped_data_path, 'train_svhn_imgs.npy'))
		labels = np.load(os.path.join(cropped_data_path, 'train_svhn_labels.npy'))
	elif data_type == "testing":
		imgs = np.load(os.path.join(cropped_data_path, 'test_svhn_imgs.npy'))
		labels = np.load(os.path.join(cropped_data_path, 'test_svhn_labels.npy'))
	elif(data_type == "validation"):
		imgs = np.load(os.path.join(cropped_data_path, 'valid_svhn_imgs.npy'))
		labels = np.load(os.path.join(cropped_data_path, 'valid_svhn_labels.npy'))
	else:
		raise Exception("Data Set not found!")
	return imgs, labels

if __name__ == '__main__':
	train_data, train_labels = create_svhn('train')
	test_data, test_labels = create_svhn('test')
	train_data, valid_data, train_labels, valid_labels = train_validation_spit(train_data, train_labels)	
	write_npy_file(train_data, train_labels, "train")
	write_npy_file(valid_data, valid_labels, "valid")
	write_npy_file(test_data, test_labels, "test")