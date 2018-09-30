import imageio
import glob
import sys
import numpy as np
import cv2 as cv

dir_name = "../../../../Dataset/MNIST/train_labels/"
dir_name_to_write = "../../../../Dataset/MNIST/inpainting/train_inputs(half)/"


def add_black_square_to_image(image, h):
	h = int(h)
	height = image.shape[0]
	width = image.shape[1]
	h1 = height/2-h
	h2 = height/2+h
	w1 = width/2-h
	w2 = width/2+h

	for i in range(h1,h2):
		for j in range(w1,w2):
			image[i][j] = 0
			#image[i][j][1] = 0
			#image[i][j][2] = 0
	return image

def add_black_to_half_image(image):
	height = image.shape[0]
	width = image.shape[1]

	for i in range(height):
		for j in range(width/2):
			image[i][j] = 0
			#image[i][j][1] = 0
			#image[i][j][2] = 0
	return image

def add_noise(image, noise_type):
	height = image.shape[0]
	width = image.shape[1]
	if noise_type == "gaussian":
		mean = 0
		var = 0.1
		sigma = 50
		gauss = np.random.normal(mean,sigma,(height,width))
		gauss = gauss.reshape(height,width)
		noisy_image = image+gauss
		print(noisy_image.shape)
		return noisy_image
	elif noise_type == "blur":
		return cv.GaussianBlur(image,ksize=(11,11),sigmaX=10.0,sigmaY=10.0)
	elif noise_type == "sp":
		s_vs_p = 0.5
		amount = 0.4
		out = np.copy(image)
		num_salt = np.ceil(amount*image.size*s_vs_p)
		coords = [np.random.randint(0,i-1,int(num_salt)) for i in image.shape]
		out[coords] = 1

		num_pepper = np.ceil(amount*image.size*(1. - s_vs_p))
		coords = [np.random.randint(0,i-1,int(num_pepper)) for i in image.shape]
		out[coords] = 0
		return out

def main():

	if(len(sys.argv) != 3):
		print("Give at least 2 arguments")
		return

	for image_path in glob.glob(dir_name+"*.png"):
		image = imageio.imread(image_path)
		image_path_splitted = image_path.split("/")
		if(sys.argv[1] == "addBlack"):
			image2 = add_black_square_to_image(image, sys.argv[2])
		elif(sys.argv[1] == "addHalfBlack"):
			image2 = add_black_to_half_image(image)
		elif(sys.argv[1] == "addNoise"):
			image2 = add_noise(image, sys.argv[2])
			#image2 = cv.GaussianBlur(image,(int(sys.argv[2]),int(sys.argv[2])),0)
		imageio.imwrite(dir_name_to_write+image_path_splitted[-1],image2[:,:])

if __name__ == '__main__':
	main()