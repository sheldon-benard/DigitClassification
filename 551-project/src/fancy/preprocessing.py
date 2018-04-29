import cv2
import numpy as np
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates

def threshold_normalize(data,transform):
	threshold = 254
	maxVal = 255

	ret, thresh = cv2.threshold(np.uint8(data), threshold, maxVal, cv2.THRESH_BINARY)

	if transform:
		copy = thresh.copy()

		copy = elastic_transform(copy)

		return thresh/255.0, copy/255.0

	return thresh/255.0

def elastic_transform(data):
	"""referenced from https://gist.github.com/fmder/e28813c1e8721830ff9c"""
	alpha = 15
	sigma = 15
	print("Elastic Transform")
	np.random.seed(1234)
	rand_state = np.random.RandomState()
	for i in range(len(data)):
		img_shape = data[i].shape
		dx = gaussian_filter((rand_state.rand(*img_shape) * 2 - 1), sigma, mode="constant") * alpha
		dy = gaussian_filter((rand_state.rand(*img_shape) * 2 - 1), sigma, mode="constant") * alpha
		x, y = np.meshgrid(np.arange(img_shape[0]), np.arange(img_shape[1]))
		indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1))
		data[i] = map_coordinates(data[i], indices, order=1).reshape(img_shape)
	return data

