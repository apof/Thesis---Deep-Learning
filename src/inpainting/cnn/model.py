import argparse
import sys
import os

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
WINDOW_SIZE = 5

#Define Hyperparameters
STEPS = 25000
DROPOUT_RATE = 0.4

#Define path to Read Dataset and Write the results
DIR_TRAIN_LABELS = "../../../../Dataset/MNIST/train_labels/"
DIR_TRAIN_INPUTS = "../../../../Dataset/MNIST/inpainting/train_inputs(5x5)/"
DIR_TEST_LABELS = "../../../../Dataset/MNIST/testing_labels/"
DIR_TEST_INPUTS = "../../../../Dataset/MNIST/inpainting/test_inputs(5x5)/"
DIR_TO_WRITE_RESULTS = "../../../../Dataset/MNIST/inpainting/cnn/predicted/"

FLAGS = None

def cnn_model_fn(features, labels, mode):

  # Reshape Input to 4-D tensor: [batch_size, IMAGE_WIDTH, IMAGE_HEIGHT, CHANNELS]
  input_layer = tf.reshape(features["x"], [-1, IMAGE_WIDTH, IMAGE_HEIGHT, CHANNELS])

  # Input Tensor Shape: [batch_size, 28, 28, 1]
  # Output Tensor Shape: [batch_size, 28, 28, 32]
  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=32,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

  # Input Tensor Shape: [batch_size, 28, 28, 32]
  # Output Tensor Shape: [batch_size, 14, 14, 32]
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

  # Input Tensor Shape: [batch_size, 14, 14, 32]
  # Output Tensor Shape: [batch_size, 14, 14, 64]
  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=64,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

  # Input Tensor Shape: [batch_size, 14, 14, 64]
  # Output Tensor Shape: [batch_size, 7, 7, 64]
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

  # Flatten tensor into a batch of vectors
  # Input Tensor Shape: [batch_size, 7, 7, 64]
  # Output Tensor Shape: [batch_size, 7 * 7 * 64]
  pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])

  # Input Tensor Shape: [batch_size, 7 * 7 * 64]
  # Output Tensor Shape: [batch_size, 1024]
  dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)

  # Add dropout operation; 0.6 probability that element will be kept
  dropout = tf.layers.dropout(
      inputs=dense, rate=DROPOUT_RATE, training=mode == tf.estimator.ModeKeys.TRAIN)

  # Input Tensor Shape: [batch_size, 1024]
  # Output Tensor Shape: [batch_size, IMAGE_SIZE]
  output_layer = tf.layers.dense(inputs=dropout, units=IMAGE_SIZE)

  if mode == tf.estimator.ModeKeys.PREDICT:
    print("Predicting")
    return tf.estimator.EstimatorSpec(mode=mode, predictions=output_layer)

  # Calculate Loss
  loss = tf.reduce_mean(tf.square(labels - output_layer))

  # Configure the Training
  if mode == tf.estimator.ModeKeys.TRAIN:
    print("Training")
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0001)
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
  
def main(unused_argv):

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
  

  # Create the Estimator
  mnist_estimator = tf.estimator.Estimator(
      model_fn=cnn_model_fn)

  # Train the model
  train_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": train_inputs},
      y=train_labels,
      batch_size=100,
      num_epochs=None,
      shuffle=True)
  mnist_estimator.train(
      input_fn=train_input_fn,
      steps=STEPS)

  test_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": test_inputs},
      shuffle=False)
  preds = mnist_estimator.predict(
    input_fn=test_input_fn)

  if not os.path.exists(os.path.dirname(DIR_TO_WRITE_RESULTS)):
    os.makedirs(os.path.dirname(DIR_TO_WRITE_RESULTS))

  count=0;
  for inp in test_inputs:
    count+=1
    imageio.imwrite(DIR_TO_WRITE_RESULTS+"input"+str(count)+".png",inp.reshape((IMAGE_HEIGHT,IMAGE_WIDTH),order='C'))

  count=0;
  total_psnr = 0
  for pred in preds:
    count+=1
    pred = pred.reshape((IMAGE_HEIGHT,IMAGE_WIDTH),order='C')
    for i in range(pred.shape[0]):
      for j in range(pred.shape[1]):
        if(pred[i][j]<0):
          pred[i][j] = 0
        #else:
          #pred[i][j] = (pred[i][j]-MIN)*255/(MAX-MIN)
          #pred[i][j] = int(pred[i][j])
    predicted_image = utils.fill_dark_part(test_inputs[count-1].reshape((IMAGE_HEIGHT,IMAGE_WIDTH),order='C'), pred, IMAGE_HEIGHT, IMAGE_HEIGHT, WINDOW_SIZE)
    #predicted_image = utils.fill_half_dark_part(test_inputs[count-1].reshape((IMAGE_HEIGHT, IMAGE_WIDTH),order='C'), pred, IMAGE_HEIGHT, IMAGE_WIDTH)
    
    total_psnr += utils.psnr(test_labels[count-1].reshape((IMAGE_HEIGHT,IMAGE_WIDTH),order='C'),predicted_image)
    imageio.imwrite(DIR_TO_WRITE_RESULTS+str(count)+".png",predicted_image)
    
  print("psnr = "+str(total_psnr/count))

if __name__ == "__main__":
  tf.app.run()