import imageio
import glob
import numpy as np
import math
import os
from skimage.measure import compare_ssim as ssim

TEST_NUMBER = 10000

def key_func(x):
	print(os.path.split(".")[0])
	return os.path.split(".")[0]

def load_dataset(dir_name):
	list_of_images = []
	list_image_parth = []
	i=1
	for image_path in sorted(glob.glob(dir_name+"*.png")):
		# if(i<10000):
		# 	print(image_path)
		# i+=1
		image = imageio.imread(image_path)
		im = image.flatten()
		im = im.astype(np.float32)
		list_of_images.append(im)
		list_image_parth.append(image_path)
	return np.array(list_of_images)

def psnrr(image1, image2):
	mse = np.mean((image1-image2)**2)
	if mse == 0:
		return 100
	MAX = 255.0
	return 20*math.log10(MAX/math.sqrt(mse))

inp = load_dataset("/home/jim/Desktop/Ptuxiaki/TensorFlow/Dataset/MNIST/inpainting/autoencoder/predicted(half)/")
labels = load_dataset("/home/jim/Desktop/Ptuxiaki/TensorFlow/Dataset/MNIST/testing_labels2/")
total_psnr = 0
total_ssim = 0

for i in range(TEST_NUMBER):
	image = labels[i].reshape((28,28),order='C')
	image_noisy = inp[i].reshape((28,28),order='C')
	total_psnr += psnrr(image, image_noisy)
	total_ssim += ssim(image,  image_noisy, data_range = image_noisy.max() - image_noisy.min())
	#imageio.imwrite("/home/jim/Desktop/Ptuxiaki/TensorFlow/Dataset/MNIST/testing_labels2/"+str(i+1)+".png",labels[i].reshape((28,28),order='C'))

print(total_psnr/TEST_NUMBER)
print(total_ssim/TEST_NUMBER)