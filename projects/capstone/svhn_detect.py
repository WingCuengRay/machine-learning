import sys
import os
import cv2
import numpy as np
import tensorflow as tf
import pylab
import matplotlib
from matplotlib import pyplot as plt

from pdb import set_trace as bp


def detect(img_path, saved_model_weights):
  #print("hola", img_path)
  img = cv2.imread(img_path, 3)
  img_data = matplotlib.pyplot.imshow(img)
  pylab.show(img_data)


if __name__ == "__main__":
  
  img_path = None
  if len(sys.argv) > 1:
    print("Reading Image file:", sys.argv[1])
    if os.path.isfile(sys.argv[1]):
      img_path = sys.argv[1]
    else:
      raise EnvironmentError("I'm sorry, I'm afraid I can't find that file.")
  else:
     raise EnvironmentError("You must pass an image file to process")
  
  saved_model_weights = "model.ckpt"

  detect(img_path, saved_model_weights)