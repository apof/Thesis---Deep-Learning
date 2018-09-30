import tensorflow as tf
import os
import numpy as np
import imageio
import utils

#Define information about the problem
IMAGE_WIDTH = 28
IMAGE_HEIGHT = 28
CHANNELS = 1
IMAGE_SIZE = IMAGE_WIDTH*IMAGE_HEIGHT*CHANNELS

#Define Hyperparameters
batch_size = 10
batch_size2 = 100
FILTERS = 512

#Define path to Read Dataset and Write the results
DIR_TRAIN_LABELS = "../../../../Dataset/MNIST/train_labels/"
DIR_TRAIN_INPUTS = "../../../../Dataset/MNIST/denoising/train_inputs_sap/"
DIR_TEST_LABELS = "../../../../Dataset/MNIST/testing_labels/"
DIR_TEST_INPUTS = "../../../../Dataset/MNIST/denoising/test_inputs_sap/"
DIR_TO_WRITE_RESULTS = "../../../../Dataset/MNIST/denoising/encoder-decoder/predicted/"

def enc_dec(input_layer):

  # Input Tensor Shape: [batch_size, 28, 28, 1]
  # Output Tensor Shape: [batch_size, 28, 28, FILTERS]
	conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=FILTERS,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu)

  	# Input Tensor Shape: [batch_size, 28, 28, FILTERS]
  	# Output Tensor Shape: [batch_size, 14, 14, FILTERS]
  	pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

  	# Input Tensor Shape: [batch_size, 14, 14, FILTERS]
  	# Output Tensor Shape: [batch_size, 14, 14, FILTERS]
  	conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=FILTERS,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu)

  	# Input Tensor Shape: [batch_size, 14, 14, FILTERS]
  	# Output Tensor Shape: [batch_size, 7, 7, FILTERS]
  	pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    # Input Tensor Shape: [batch_size, 7, 7, FILTERS]
    # Output Tensor Shape: [batch_size, 7, 7, FILTERS]
  	conv3 = tf.layers.conv2d(
      inputs=pool2,
      filters=FILTERS,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu)

    # Input Tensor Shape: [batch_size, 7, 7, FILTERS]
    # Output Tensor Shape: [batch_size, 14, 14, FILTERS]
  	upsample1 = tf.layers.conv2d_transpose(
  	  inputs=conv3, 
  	  filters=FILTERS, 
  	  kernel_size=3, 
  	  padding="same", 
  	  strides=2)

    # Input Tensor Shape: [batch_size, 14, 14, FILTERS]
    # Output Tensor Shape: [batch_size, 28, 28, FILTERS]
  	upsample2 = tf.layers.conv2d_transpose(
  	  inputs=upsample1, 
  	  filters=FILTERS, 
  	  kernel_size=3, 
  	  padding="same", 
  	  strides=2)

    # Input Tensor Shape: [batch_size, 28, 28, FILTERS]
    # Output Tensor Shape: [batch_size, 28, 28, 1]
  	conv4 = tf.layers.conv2d(
      inputs=upsample2,
      filters=1,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu)

  	output_layer = tf.nn.relu(conv4)

  	return output_layer

def next_batch(data, num_of_batch):
  start_index = num_of_batch*batch_size
  end_index = num_of_batch*batch_size + batch_size
  if(end_index > len(data)):
    end_index = len(data)
  return data[start_index:end_index]

def next_batch2(data, num_of_batch):
  start_index = num_of_batch*batch_size2
  end_index = num_of_batch*batch_size2 + batch_size2
  if(end_index > len(data)):
    end_index = len(data)
  return data[start_index:end_index]

def main():
  print("Started")
  # Load training and test data
  train_labels = utils.load_dataset(DIR_TRAIN_LABELS)
  print("reading train_labels done")
  print(train_labels.shape)
  train_inputs = utils.load_dataset(DIR_TRAIN_INPUTS)
  print("reading train_inputs done")
  print(train_inputs.shape)
  test_labels = utils.load_dataset(DIR_TEST_LABELS)
  print("reading test_labels done")
  print(test_labels.shape)
  test_inputs = utils.load_dataset(DIR_TEST_INPUTS)
  print("reading test_inputs done")
  print(test_inputs.shape)

  #define the model
  inputs = tf.placeholder(tf.float32, (None, IMAGE_WIDTH, IMAGE_HEIGHT, CHANNELS))
  outputs = enc_dec(inputs)
  labels = tf.placeholder(tf.float32, (None, IMAGE_WIDTH, IMAGE_HEIGHT, CHANNELS))

  # Calculate Loss and Optimizer
  loss = tf.reduce_mean(tf.square(labels - outputs))

  optimizer = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss)

  init = tf.global_variables_initializer()

  if(len(train_labels) % batch_size == 0):
  	train_iters = len(train_labels) / batch_size
  else:
  	train_iters = len(train_labels) / batch_size + 1

  if(len(test_labels) % batch_size2 == 0):
    test_iters = len(test_labels) / batch_size2
  else:
    test_iters = len(test_labels) / batch_size2 + 1

  #Start a new Session
  with tf.Session() as sess:
    sess.run(init)
    #Train 
    for i in range(train_iters):
      next_inputs = next_batch(train_inputs, i)
      next_labels = next_batch(train_labels, i)
      next_inputs = next_inputs.reshape((-1, IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS))
      next_labels = next_labels.reshape((-1, IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS))
      sess.run([optimizer, loss], feed_dict = {inputs: next_inputs, labels: next_labels})

    if not os.path.exists(os.path.dirname(DIR_TO_WRITE_RESULTS)):
      os.makedirs(os.path.dirname(DIR_TO_WRITE_RESULTS))

    #Test
    count=0;
    total_psnr = 0
    for i in range(test_iters):
      next_test = next_batch2(test_inputs, i)
      next_test = next_test.reshape((-1, IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS))
      preds = sess.run([outputs], feed_dict = {inputs: next_test})

      for i in range(len(preds[0])):
        pred = preds[0][i]
        count+=1
        pred = pred.reshape((IMAGE_HEIGHT,IMAGE_WIDTH),order='C')
        for i in range(pred.shape[0]):
          for j in range(pred.shape[1]):
            if(pred[i][j]<0):
              pred[i][j] = 0
            else:
              pred[i][j] = int(pred[i][j])
              #pred[i][j] = (pred[i][j]-MIN)*255/(MAX-MIN)            
        total_psnr += utils.psnr(test_labels[count-1].reshape((IMAGE_HEIGHT,IMAGE_WIDTH),order='C'),pred)
        imageio.imwrite(DIR_TO_WRITE_RESULTS+str(count)+".png",pred)
    
    count=0;
    for inp in test_inputs:
      count+=1
      imageio.imwrite(DIR_TO_WRITE_RESULTS+"input"+str(count)+".png",inp.reshape((IMAGE_HEIGHT,IMAGE_WIDTH),order='C'))

    print("psnr = "+str(total_psnr/count))


if __name__ == "__main__":
	main()