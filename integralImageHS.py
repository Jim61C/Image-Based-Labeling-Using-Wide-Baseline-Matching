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
from integralImage import IntegralImage

class IntegralImageHS(IntegralImage):
	"""
	Class for integral images, including HS integral image and HOG integral image
	For roi: img[i:i+len, j:j+len], corresponding feature is found by 
	integral_img_feature[i+len-1][j+len-1] - integral_img_feature[i-1][j+len-1] - integral_img_feature[i+len-1][j-1] + integral_img_feature[i-1][j-1]
	"""
	def __init__(self, img):
		IntegralImage.__init__(self, img)
		self.img_hsv = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_BGR2HSV) # img_hsv: Hue: 0-360, Saturation: 0-1, Value: 0-255
		self.bin_len = 16 # H,S feature 16 bin

	def getEmptyFeature(self):
		"""
		Over written by subclasses
		"""
		# each entry is a 2D array of H,S (first dimension H, second dimension S)
		return np.zeros(shape = (self.bin_len, self.bin_len))

	def configureFeatureAtRowCol(self, i, j):
		"""
		Over written by subclasses, i is row index, j is col index, img is original img readin from cv2
		"""
		this_hue_bin = int(self.img_hsv[i][j][0]/360.0 * self.bin_len)
		if (this_hue_bin == self.bin_len):
			this_hue_bin = self.bin_len - 1
		this_saturation_bin = int(self.img_hsv[i][j][1]/1.0 * self.bin_len)
		if (this_saturation_bin == self.bin_len):
			this_saturation_bin = self.bin_len - 1

		self.integral_img_feature[i][j][this_hue_bin][this_saturation_bin] += 1.0
		return



def main():
	print "unit testing for integralImageHS"
	image_db = "images"
	folder_name = "testset_flower2"
	img_name = "test1.jpg"
	img = cv2.imread("{image_db}/{folder}/{name}".format(image_db = image_db, folder = folder_name, name = img_name), 1)
	cv2.imshow("test", img)
	cv2.waitKey(0)

	integral_img_obj = IntegralImageHS(img)
	integral_img_obj.computeIntegralImageFeature()

	"""test1, subpatch's HS 2d histogram should be exactly the same"""
	row_start = 200
	col_start = 200
	rows = 200
	cols = 200

	# hs_2d_hist_from_integral = integral_img_obj.integral_img_feature[row_start + rows - 1][col_start + cols - 1] - \
	# integral_img_obj.integral_img_feature[row_start - 1][col_start + cols - 1] - \
	# integral_img_obj.integral_img_feature[row_start + rows - 1][col_start - 1] + \
	# integral_img_obj.integral_img_feature[row_start - 1][col_start - 1]
	hs_2d_hist_from_integral = integral_img_obj.getIntegralImageFeature(\
		row_start = row_start, \
		row_end = row_start + rows, \
		col_start = col_start, \
		col_end = col_start + cols)

	roi = img[row_start:row_start + rows, col_start: col_start + cols]

	hsv_roi = cv2.cvtColor(roi.astype(np.float32),cv2.COLOR_BGR2HSV)
	hs_2d_from_roi_computation = cv2.calcHist([hsv_roi], [0, 1], None, \
		[integral_img_obj.bin_len, integral_img_obj.bin_len], [0, 360, 0, 1.0])

	print "hs_2d_hist_from_integral:\n", hs_2d_hist_from_integral
	print "hs_2d_from_roi_computation:\n", hs_2d_from_roi_computation

	print "\n\n, difference:\n", hs_2d_from_roi_computation - hs_2d_hist_from_integral

	print "integral_img_obj.integral_img_feature[row_start][col_start].shape: ", \
	integral_img_obj.integral_img_feature[row_start][col_start].shape

	return

if __name__ == "__main__":
	main()




