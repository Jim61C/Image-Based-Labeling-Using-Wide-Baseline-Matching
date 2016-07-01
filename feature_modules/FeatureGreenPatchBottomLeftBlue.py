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
import comparePatches
import scipy.spatial.distance as DIST
from sklearn.preprocessing import normalize
from feature_modules import Feature


class FeatureGreenPatchBottomLeftBlue(Feature):
	"""
	TODO: Add an HOG feature to compensate for this feature
	"""
	def __init__(self, patch, id):
		Feature.__init__(self, patch, id)
		self.HUE_START_INDEX = 8
		self.HUE_END_INDEX = 9
		self.SATURATION_START_INDEX = 7
		self.SATURATION_END_INDEX = 12

		self.OTHER_HUE_START_INDEX = 3
		self.OTHER_HUE_END_INDEX = 4
		self.OTHER_SATURATION_START_INDEX = 6
		self.OTHER_SATURATION_END_INDEX = 10
		
		# bottom left blue: hue model and saturation model normalized seperately:
		self.BLUE_FEATURE_MODEL = np.array([ 0.,          0.,          0.,          0.,          0.,          0. ,         0.,
											  0.,          1.,          0. ,         0.,          0.,          0.,          0.,
											  0.,          0. ,         \
											  0. ,         0.,          0.,          0.,          0.,
											  0.,          0.,   		0.2,  		 0.2,         0.2,         0.2,
											  0.2,          0.,          0.,          0.,          0.        ])
		# reinforce that the other sub patches must be green
		self.GREEN_FEATURE_MODEL = np.array([ 0.,          0.,          0.,          1.,          0.,          0.,          0.,
											  0.,          0.,          0.,          0.,          0.,          0.,          0.,
											  0.,          0.,         \
											  0.,          0.,          0.,          0.,          0.,          0.,          0.25,
											  0.25,         0.25,         0.25,         0.,          0.,          0.,          0.,
											  0.,          0.        ]) # with this, can get the target patch as 4th best

		# """try changing the sum of targeted saturation to be = 1"""
		# self.BLUE_FEATURE_MODEL = np.array([ 0.,          0.,          0.,          0.,          0.,          0. ,         0.,
		# 									  0.,          1.,          0. ,         0.,          0.,          0.,          0.,
		# 									  0.,          0. ,         \
		# 									  1.           ])
		# # reinforce that the other sub patches must be green
		# self.GREEN_FEATURE_MODEL = np.array([ 0.,          0.,          0.,          1.,          0.,          0.,          0.,
		# 									  0.,          0.,          0.,          0.,          0.,          0.,          0.,
		# 									  0.,          0.,         \
		# 									  1.           ])
		
		self.FEATURE_MODEL = np.concatenate(\
			(self.BLUE_FEATURE_MODEL, self.GREEN_FEATURE_MODEL, self.GREEN_FEATURE_MODEL, self.GREEN_FEATURE_MODEL), \
			axis = 1)
		self.FEATURE_MODEL = normalize(self.FEATURE_MODEL, norm='l1')[0] # normalize the FEATURE_MODEL using l1

	def computeFeatureIntegralImage(self, integral_img_obj):
		assert integral_img_obj.integral_image_type == "HS", "in FeatureGreenPatchBottomLeftBlue, integral_img_obj used should be HS"

		if (not len(self.patch.hs_2d_arr) == 5):
			self.computeHS2DArrFromIntegralImage(integral_img_obj, self.patch)

		if(not (len(self.patch.HueHistArr) == 5 and len(self.patch.SaturationHistArr) == 5)):
			self.patch.HueHistArr = []
			self.patch.SaturationHistArr = []
			self.patch.ValueHistArr = []
			"""derive from 2D instead of recompute"""
			for i in range(0, 5):
				self.patch.HueHistArr.append(self.derive1DHueFrom2D(self.patch.hs_2d_arr[i]))
				self.patch.SaturationHistArr.append(self.derive1DSaturationFrom2D(self.patch.hs_2d_arr[i]))

		self.hist = np.concatenate((self.patch.HueHistArr[self.BOTTOM_LEFT_INDEX], \
			self.patch.SaturationHistArr[self.BOTTOM_LEFT_INDEX]), axis = 1)

		for i in range(self.TOP_LEFT_INDEX, self.BOTTOM_RIGHT_INDEX + 1):
			if (i != self.BOTTOM_LEFT_INDEX):
				other_patch_hue = self.patch.HueHistArr[i]
				other_patch_saturation = self.patch.SaturationHistArr[i]
				other_patch_hist = np.concatenate((other_patch_hue, other_patch_saturation), axis = 1)
				self.hist = np.concatenate((self.hist, other_patch_hist), axis = 1)
			
		self.hist = normalize(self.hist, norm='l1')[0] # normalize the histogram using l1

	### override super class ###
	def computeFeature(self, img, useGaussianSmoothing = True):
		img_hsv = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_BGR2HSV)
		
		if (not len(self.patch.hs_2d_arr) == 5):
			gaussian_window = comparePatches.gauss_kernels(self.patch.size, sigma = self.patch.size/6.0)
			self.computeHS2DArr(img_hsv, self.patch, gaussian_window)

		if(not (len(self.patch.HueHistArr) == 5 and len(self.patch.SaturationHistArr) == 5)):
			self.patch.HueHistArr = []
			self.patch.SaturationHistArr = []
			self.patch.ValueHistArr = []
			"""derive from 2D instead of recompute"""
			for i in range(0, 5):
				self.patch.HueHistArr.append(self.derive1DHueFrom2D(self.patch.hs_2d_arr[i]))
				self.patch.SaturationHistArr.append(self.derive1DSaturationFrom2D(self.patch.hs_2d_arr[i]))

		self.hist = np.concatenate((self.patch.HueHistArr[self.BOTTOM_LEFT_INDEX], \
			self.patch.SaturationHistArr[self.BOTTOM_LEFT_INDEX]), axis = 1)
		# plotStatistics.plotOneGivenHist("", "BOTTOM_LEFT Hue", self.patch.HueHistArr[self.BOTTOM_LEFT_INDEX], save = False, show = True)
		# plotStatistics.plotOneGivenHist("", "BOTTOM_LEFT SaturationHistArr", self.patch.SaturationHistArr[self.BOTTOM_LEFT_INDEX], save = False, show = True)
		
		"""feature constructor at build phace, not run anymore after feature is built"""
		# model_constructor_hue = np.zeros(len(self.patch.HueHistArr[self.BOTTOM_LEFT_INDEX]))
		# model_constructor_saturation = np.zeros(len(self.patch.SaturationHistArr[self.BOTTOM_LEFT_INDEX]))
		
		# model_constructor_hue[self.HUE_START_INDEX:self.HUE_END_INDEX] = \
		# self.patch.HueHistArr[self.BOTTOM_LEFT_INDEX][self.HUE_START_INDEX:self.HUE_END_INDEX]

		# model_constructor_saturation[self.SATURATION_START_INDEX:self.SATURATION_END_INDEX] = \
		# self.patch.SaturationHistArr[self.BOTTOM_LEFT_INDEX][self.SATURATION_START_INDEX:self.SATURATION_END_INDEX]
		
		# model_hist = np.concatenate(\
		# 	(normalize(model_constructor_hue, norm = "l1")[0], normalize(model_constructor_saturation, norm = "l1")[0]), \
		# 	axis = 1) # no need to normalized the two models each as they will be normalized altogether later
		# print model_hist

		# green_model_hue = np.zeros(len(self.patch.HueHistArr[self.TOP_RIGHT_INDEX]))
		# green_model_saturation = np.zeros(len(self.patch.SaturationHistArr[self.TOP_RIGHT_INDEX]))
		# green_model_hue[3:4] = self.patch.HueHistArr[self.TOP_RIGHT_INDEX][3:4]
		# green_model_saturation[7:10] = self.patch.SaturationHistArr[self.TOP_RIGHT_INDEX][7:10]
		# green_model_hist = np.concatenate(\
		# 	(normalize(green_model_hue, norm = "l1")[0], normalize(green_model_saturation, norm = "l1")[0]), \
		# 	axis = 1)
		# print green_model_hist

		# plotStatistics.plotOneGivenHist("", "Bottom left hue", self.patch.HueHistArr[self.BOTTOM_LEFT_INDEX], save = False, show = True)
		# plotStatistics.plotOneGivenHist("", "Bottom left saturation", self.patch.SaturationHistArr[self.BOTTOM_LEFT_INDEX], save = False, show = True)
		# plotStatistics.plotOneGivenHist("", "model_hist", model_hist, save = False, show = True)

		# plotStatistics.plotOneGivenHist("", "green hue", self.patch.HueHistArr[self.TOP_RIGHT_INDEX], save = False, show = True)
		# plotStatistics.plotOneGivenHist("", "green saturation", self.patch.SaturationHistArr[self.TOP_RIGHT_INDEX], save = False, show = True)
		# plotStatistics.plotOneGivenHist("", "green_model_hist", green_model_hist, save = False, show = True)
		
		# plotStatistics.plotOneGivenHist("", "FEATURE_MODEL", self.FEATURE_MODEL, save = False, show = True)		

		for i in range(self.TOP_LEFT_INDEX, self.BOTTOM_RIGHT_INDEX + 1):
			if (i != self.BOTTOM_LEFT_INDEX):
				other_patch_hue = self.patch.HueHistArr[i]
				other_patch_saturation = self.patch.SaturationHistArr[i]

				# plotStatistics.plotOneGivenHist("", "subpatch {i} Hue".format(i = i), \
					# self.patch.HueHistArr[i], save = False, show = True)
				# plotStatistics.plotOneGivenHist("", "subpatch {i} Saturation".format(i = i), \
					# self.patch.SaturationHistArr[i], save = False, show = True)

				other_patch_hist = np.concatenate((other_patch_hue, other_patch_saturation), axis = 1)
				self.hist = np.concatenate((self.hist, other_patch_hist), axis = 1)
			
		self.hist = normalize(self.hist, norm='l1')[0] # normalize the histogram using l1
		
	def featureResponse(self):
		assert (not self.hist is None), "Error in FeatureGreenPatchBottomLeftBlue: calling computeScore before the feature hist is computed!"
		assert len(self.hist) == len(self.FEATURE_MODEL), "Error in FeatureGreenPatchBottomLeftBlue: feature length is not correctly computed!"
		dissimilarity = DIST.euclidean(self.hist, self.FEATURE_MODEL)
		# dissimilarity = comparePatches.Jensen_Shannon_Divergence(self.hist, self.FEATURE_MODEL)
		return 1.0 / (1.0 + dissimilarity)

	def computeScore(self):
		if(self.score is None):
			self.score = self.featureResponse()




