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
	For roi: img[i:i+len, j:j+len], corresponding feature is found by 
	integral_img_feature[i+len-1][j+len-1] - integral_img_feature[i-1][j+len-1] - integral_img_feature[i+len-1][j-1] + integral_img_feature[i-1][j-1]
	"""
	def __init__(self, img):
		self.img = img
		self.bin_len = None
		self.integral_img_feature = None
		self.integral_image_type = "GENERAL"

	def getIntegralImageFeature(self, row_start, row_end, col_start, col_end):
		"""
		get subpatch HS feature from self.integral_img_feature
		roi: img[row_start:row_end, col_start:col_end], ends are exclusive
		"""
		feature_HS_C = self.integral_img_feature[row_end - 1][col_end - 1]
		feature_HS_B = self.integral_img_feature[row_start - 1][col_end - 1] if (row_start > 0) else self.getEmptyFeature()
		feature_HS_D = self.integral_img_feature[row_end - 1][col_start - 1] if (col_start > 0) else self.getEmptyFeature()
		feature_HS_A = self.integral_img_feature[row_start -1][col_start -1] if (row_start > 0 and col_start > 0) else self.getEmptyFeature()

		# a copy will be created, will not affect self.integral_img_feature
		return feature_HS_C - feature_HS_B - feature_HS_D + feature_HS_A

	def computeIntegralImageFeature(self):
		m = self.img.shape[0] # img height
		n = self.img.shape[1] # img width

		self.integral_img_feature = []

		for i in range(0, m):
			integral_img_feature_this_row = []
			for j in range(0, n):
				integral_img_feature_this_row.append(self.getEmptyFeature())
			self.integral_img_feature.append(integral_img_feature_this_row)

		assert (len(self.integral_img_feature) == m), "integral image of features row length should be the same as image height"
		for i in range(0, len(self.integral_img_feature)):
			assert (len(self.integral_img_feature[i]) == n), "row " + i + \
			":integral image of features col length should be the same as image width"

		for i in range(0, m):
			for j in range(0, n):
				self.configureFeatureAtRowCol(i, j)

				if (i == 0 and j == 0):
					self.integral_img_feature[i][j] = self.integral_img_feature[i][j]
				elif(i == 0):
					self.integral_img_feature[i][j] = self.integral_img_feature[i][j] + self.integral_img_feature[i][j-1]
				elif(j == 0):
					self.integral_img_feature[i][j] = self.integral_img_feature[i][j] + self.integral_img_feature[i-1][j]
				else:
					self.integral_img_feature[i][j] = self.integral_img_feature[i][j] + \
					self.integral_img_feature[i-1][j] + \
					self.integral_img_feature[i][j-1] - \
					self.integral_img_feature[i-1][j-1]
		return

	def getEmptyFeature(self):
		"""
		Over written by subclasses
		"""
		return

	def configureFeatureAtRowCol(self, i, j):
		"""
		Over written by subclasses, i is row index, j is col index, img is original img readin from cv2
		"""
		return


		
