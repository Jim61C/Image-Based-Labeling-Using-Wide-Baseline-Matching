import cv2
from cv2 import cv
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import entropy
import pyemd
import math
import sys
import drawMatches
import plotStatistics
import operator
from sklearn.preprocessing import normalize
import itertools
import random
import scipy.spatial.distance as DIST

class IntegralImage():
	"""
	Class for integral images, including HS integral image and HOG integral image
	"""
	def __init__(self, img):
		self.img = img

		self.integral_img_HS = None
		self.integral_img_HOG = None

	def computeHSIntegralImage(self, bin_len = 16):
		m = self.img.shape[0] # img height
		n = self.img.shape[1] # img width

		self.integral_img_HS = []

		for i in range(0, m):
			integral_img_HS_this_row = []
			for j in range(0, n):
				integral_img_HS_this_row.append(np.zeros(shape= (bin_len, bin_len))) # each entry is a 2D array of H,S (first dimension H, second dimension S)
			self.integral_img_HS.append(integral_img_HS_this_row)

		assert (len(self.integral_img_HS) == m), "integral image of features row length should be the same as image height"
		for i in range(0, len(self.integral_img_HS)):
			assert (len(self.integral_img_HS[i]) == n), "row " + i + \
			":integral image of features col length should be the same as image width"

		img_hsv = cv2.cvtColor(self.img.astype(np.float32), cv2.COLOR_BGR2HSV)

		for i in range(0, m):
			for j in range(0, n):
				this_hue_bin = int(img_hsv[i][j][0]/360.0 * bin_len)
				if (this_hue_bin == bin_len):
					this_hue_bin = bin_len - 1
				this_saturation_bin = int(img_hsv[i][j][1]/1.0 * bin_len)
				if (this_saturation_bin == bin_len):
					this_saturation_bin = bin_len - 1

				self.integral_img_HS[i][j][this_hue_bin][this_saturation_bin] += 1.0

				if (i == 0 and j == 0):
					self.integral_img_HS[i][j] = self.integral_img_HS[i][j]
				elif(i == 0):
					self.integral_img_HS[i][j] = self.integral_img_HS[i][j] + self.integral_img_HS[i][j-1]
				elif(j == 0):
					self.integral_img_HS[i][j] = self.integral_img_HS[i][j] + self.integral_img_HS[i-1][j]
				else:
					self.integral_img_HS[i][j] = self.integral_img_HS[i][j] + self.integral_img_HS[i-1][j] + self.integral_img_HS[i][j-1] - \
					self.integral_img_HS[i-1][j-1]
		return

	def computeHOGIntegralImage(self, bin_len = 16):
		return


def main():
	print "unit testing for integralImage"
	image_db = "images"
	folder_name = "testset_flower2"
	img_name = "test1.jpg"
	img = cv2.imread("{image_db}/{folder}/{name}".format(image_db = image_db, folder = folder_name, name = img_name), 1)
	cv2.imshow("test", img)
	cv2.waitKey(0)

	integral_img_obj = IntegralImage(img)
	integral_img_obj.computeHSIntegralImage()

	return

if __name__ == "__main__":
	main()
		
