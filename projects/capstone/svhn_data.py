import os
import struct
import numpy as np
import scipy
import sys
import tarfile


from digit_struct import DigitStruct

from scipy.io import loadmat
from sklearn.cross_validation import train_test_split
from itertools import product
from six.moves.urllib.request import urlretrieve

# '''for debugging'''
from pdb import set_trace as bp
#import matplotlib.pyplot as plt
#from random import randint

DATA_PATH = "data/svhn"
CROPPED_DATA_PATH = DATA_PATH+"/cropped"
FULL_DATA_PATH = DATA_PATH+"/full"
FORMAT_2_FILES = ['{}_32x32.mat'.format(s) for s in ['train', 'test', 'extra']]
PIXEL_DEPTH = 255
NUM_LABELS = 10
last_percent_reported = None

def read_data_file(file_name):
	if file_name.endswith("tar.gz"):
		extract_data_file(file_name)
		raise NotImplementedError('Need to do something after data is loaded!')
	else:
		file = open(file_name, 'rb')
		data = process_data_file(file)
		file.close()
	return data

def extract_data_file(file_name):
	tar = tarfile.open(file_name, "r:gz")
	tar.extractall(FULL_DATA_PATH)
	tar.close()

def read_digit_struct(dataset):
	path = FULL_DATA_PATH +"/"+dataset
	struct_file = os.path.join(path, "digitStruct.mat")
	return DigitStruct(struct_file)

def convert_img_array_vec(img_array):
	rows = img_array.shape[0]
	cols = img_array.shape[1]
	chans = img_array.shape[2]
	num_imgs = img_array.shape[3]
	scalar = 1 / PIXEL_DEPTH
	#not the most efficent way but can monitor what is happening
	new_array = np.empty(shape=(num_imgs, rows, cols, chans), dtype=np.float32)
	#new_array = np.empty(shape=(num_imgs, rows*cols*chans), dtype=np.float32)
	for x in range(0, num_imgs):
		temp = img_array[:,:,:,x]
		#vec = np.ndarray.flatten(temp)
		vec = temp
		#normalize pixels to 0 and 1. 0 is pure white, 1 is pure black.
  		norm_vec = (255-vec)*1.0/255.0  
		#bp()
		new_array[x] = norm_vec
	return new_array

def convert_labels_to_one_hot(labels):
	labels = (np.arange(NUM_LABELS) == labels[:,None]).astype(np.float32)
	return labels

def sainty_check(imgs, labels):
	ri = randint(0,len(labels)-1)
	print(labels[ri])
	plt.imshow(imgs[:,:,:,ri])
	plt.show()

def sainty_check(imgs_array):
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

def get_data_file_name(master_set, dataset):
	if master_set == "cropped":
		if dataset == "train":
			data_file_name = "train_32x32.mat"
		elif dataset == "test":
			data_file_name = "test_32x32.mat"
		elif dataset == "extra":
			data_file_name = "extra_32x32.mat"
		else:
			raise NotImplementedError('dataset must be either train, test or extra')
	elif master_set == "full":
		if dataset == "train":
			data_file_name = "train.tar.gz"
		elif dataset == "test":
			data_file_name = "test.tar.gz"
		elif dataset == "extra":
			data_file_name = "extra.tar.gz"
	else:
		raise NotImplementedError('Master data set must be full or cropped')
	return data_file_name;

def make_data_dirs(master_set):
	if master_set == "cropped":
		if not os.path.exists(CROPPED_DATA_PATH):
			os.makedirs(CROPPED_DATA_PATH)
	elif master_set == "full":
		if not os.path.exists(FULL_DATA_PATH):
			os.makedirs(FULL_DATA_PATH)
	else:
		raise NotImplementedError('Master data set must be full or cropped')

def create_svhn(master_set, dataset):
	#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	#MANUALLY CHANGE THIS FOR NOW!
	#path = FULL_DATA_PATH
	path = CROPPED_DATA_PATH
	data_file_name = get_data_file_name(master_set, dataset)
	data_file_pointer = os.path.join(path, data_file_name)
	if os.path.exists(data_file_pointer):
		return read_data_file(data_file_pointer)
	else:
		if not os.path.exists(path):
			make_data_dirs(master_set)
		downloaded_file = download_data_file(path, data_file_name)
		return read_data_file(downloaded_file)

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

def download_data_file(path, filename, force=False):
	base_url = "http://ufldl.stanford.edu/housenumbers/"
	print "Attempting to download", filename
	saved_file, _ = urlretrieve(base_url + filename, os.path.join(path, filename), reporthook=download_progress)
	print("\nDownload Complete!")
	statinfo = os.stat(saved_file)
	if statinfo.st_size == get_expected_bytes(filename):
		print("Found and verified", saved_file)
	else:
		raise Exception("Failed to verify " + filename)
	return saved_file

def get_expected_bytes(filename):
	if filename == "train_32x32.mat":
		byte_size = 182040794
	elif filename == "test_32x32.mat":
		byte_size = 64275384
	elif filename == "extra_32x32.mat":
		byte_size = 1329278602
	elif filename == "test.tar.gz":
		byte_size = 276555967
	elif filename == "train.tar.gz":
		byte_size = 404141560
	elif filename == "extra.tar.gz":
		byte_size = 1955489752
	else:
		raise Exception("Invalid file name " + filename)
	return byte_size

def train_validation_spit(train_dataset, train_labels):
	train_dataset, validation_dataset, train_labels, validation_labels = train_test_split(train_dataset, train_labels, test_size=0.33, random_state = 42)
	return train_dataset, validation_dataset, train_labels, validation_labels

def write_npy_file(data_array, lbl_array, data_set_name):
	np.save(os.path.join(CROPPED_DATA_PATH,data_set_name+'_svhn_imgs.npy'), data_array)
	print 'Saving to %s_svhn_imgs.npy file done. Data size is: %s' %((data_set_name), data_array.shape)
	np.save(os.path.join(CROPPED_DATA_PATH,data_set_name+'_svhn_labels.npy'), lbl_array)
	print 'Saving to %s_svhn_labels.npy file done. Contains %d rows' %(data_set_name, len(lbl_array))

def load_svhn_data(data_type):
	if data_type == "training":
		imgs = np.load(os.path.join(CROPPED_DATA_PATH, 'train_svhn_imgs.npy'))
		labels = np.load(os.path.join(CROPPED_DATA_PATH, 'train_svhn_labels.npy'))
	elif data_type == "testing":
		imgs = np.load(os.path.join(CROPPED_DATA_PATH, 'test_svhn_imgs.npy'))
		labels = np.load(os.path.join(CROPPED_DATA_PATH, 'test_svhn_labels.npy'))
	elif(data_type == "validation"):
		imgs = np.load(os.path.join(CROPPED_DATA_PATH, 'valid_svhn_imgs.npy'))
		labels = np.load(os.path.join(CROPPED_DATA_PATH, 'valid_svhn_labels.npy'))
	else:
		raise Exception("Data Set not found!")
	return imgs, labels

if __name__ == '__main__':
	train_data, train_labels = create_svhn('cropped', 'train')
	test_data, test_labels = create_svhn('cropped', 'test')
	train_data, valid_data, train_labels, valid_labels = train_validation_spit(train_data, train_labels)	
	write_npy_file(train_data, train_labels, "train")
	write_npy_file(valid_data, valid_labels, "valid")
	write_npy_file(test_data, test_labels, "test")
	
	#test_data, test_labels = create_svhn('full', 'test')
	#train_data, train_labels = create_svhn('full', 'train')
	#extra_data, extra_labels = create_svhn('full', 'extra')
	#train_full = read_digit_struct("train")
	#test_full = read_digit_struct("test")
	#print(train_full, test_full)
	pass