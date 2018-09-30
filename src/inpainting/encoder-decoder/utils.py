import numpy as np
import math
import glob
import imageio

def psnr(image1, image2):
	mse = np.mean((image1-image2)**2)
	if mse == 0:
		return 100
	MAX = 255.0
	return 20*math.log10(MAX/math.sqrt(mse))

def fill_dark_part(input_image, predicted_image, height, width, h):
  h1 = height/2-h
  h2 = height/2+h
  w1 = width/2-h
  w2 = width/2+h
  for i in range(h1,h2):
    for j in range(w1,w2):
      input_image[i][j] = predicted_image[i][j]
  return input_image

def fill_half_dark_part(input_image, predicted_image, height, width):
  for i in range(height):
    for j in range(width/2):
      input_image[i][j] = predicted_image[i][j]
  return input_image

def load_dataset(dir_name):
  list_of_images = []
  i=0
  for image_path in glob.glob(dir_name+"*.png"):
    image = imageio.imread(image_path)
    ###add 2 rows and 2 columns with black pixels
    # temp1 = image.shape
    # temp1 = list(temp1)
    # temp1[1] = 2
    # temp1 = tuple(temp1)
    # blank_image = np.zeros(temp1,np.float32)
    # image = np.concatenate((image,blank_image),axis=1)
    # temp1 = image.shape
    # temp1 = list(temp1)
    # temp1[0] = 2
    # temp1 = tuple(temp1)
    # blank_image = np.zeros(temp1,np.float32)
    # image = np.concatenate((image,blank_image),axis=0)
    ###

    im = image.flatten()
    im = im.astype(np.float32)
    list_of_images.append(im)
  return np.array(list_of_images)

def tf_resize_images(X_img_file_paths):
  X_data = []
  tf.reset_default_graph()
  X = tf.placeholder(tf.float32, (None, None, 3))
  tf_img = tf.image.resize_images(X, (IMAGE_SIZE_X, IMAGE_SIZE_Y),tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
        
        # Each image is resized individually as different image may be of different size.
    for file_path in glob.glob(X_img_file_paths+"*.png"):
      img = mpimg.imread(file_path)[:, :, :3] # Do not read alpha channel.
      resized_img = sess.run(tf_img, feed_dict = {X: img})
      X_data.append(resized_img.flatten())

  X_data = np.array(X_data, dtype = np.float32) # Convert to numpy
  return X_data
