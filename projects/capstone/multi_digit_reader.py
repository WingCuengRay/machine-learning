import sys
import os
import numpy as np
import tensorflow as tf
import pylab
import PIL.Image as Image
import matplotlib
from svhn_model import regression_head


from pdb import set_trace as bp

def detect(img_path, saved_model_weights):
  img = Image.open(img_path)
  img = img.resize((64,64), Image.ANTIALIAS)
  
  img_data = matplotlib.pyplot.imshow(img)
  pylab.show()
 
  
  img = np.array(img)
  batch_size = 1
  img_rows = img.shape[0]
  img_cols = img.shape[1]
  img_chans= img.shape[2]

  X_test = tf.placeholder(tf.float32, shape=(1, img_rows, img_cols, img_chans))
  #y_ = tf.placeholder(tf.int32, shape=(1, 11))
 
  test_prediction = tf.pack([tf.nn.softmax(regression_head(X_test)[0]),\
                    tf.nn.softmax(regression_head(X_test)[1]),\
                    tf.nn.softmax(regression_head(X_test)[2]),\
                    tf.nn.softmax(regression_head(X_test)[3]),\
                    tf.nn.softmax(regression_head(X_test)[4])])
  
  #create a loader
  saver = tf.train.Saver()
  with tf.Session() as sess:
	print("Loading Saved Checkpoints From:", saved_model_weights)
	#place the weights into the model.
	saver.restore(sess, saved_model_weights)
	print("Model restored.")

	norm_img = (255-img)*1.0/255.0
	exp = np.expand_dims(norm_img, axis=0)  

	#feed_dict = {X_train: img}

	test_prediction = sess.run(test_prediction, feed_dict={X_test : exp, })
	
	print("tp:", test_prediction)
	
	#predictions = sess.run(prediction, feed_dict=feed_dict)

	#print("Best Prediction is:", np.argmax(predictions))

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
  
  saved_model_weights = "regression.ckpt"
  detect(img_path, saved_model_weights)