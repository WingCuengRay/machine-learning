import os, struct
import numpy as np
from scipy import ndimage
from sklearn.cross_validation import train_test_split

'''for debugging'''
#from pdb import set_trace as bp
#import matplotlib.pyplot as plt
#from random import randint

path = "data/minst_digit"
pixel_depth = 255
NUM_LABELS = 10

def labels_to_one_hot(labels):
  labels = (np.arange(NUM_LABELS) == labels[:,None]).astype(np.float32)
  return labels

def get_dataset_by_name(data_set_name):
	if data_set_name == "training":	
		img_file = os.path.join(path, 'train-images-idx3-ubyte') 
		lbl_file = os.path.join(path, 'train-labels-idx1-ubyte')
	elif data_set_name == "testing":
		img_file = os.path.join(path, 't10k-images-idx3-ubyte')
		lbl_file = os.path.join(path, 't10k-labels-idx1-ubyte')
	else:
		raise ValueError, "dataset must be 'testing' or 'training'"
	return(img_file, lbl_file)

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