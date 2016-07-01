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


class FeatureBottomRightYellow(Feature):
	def __init__(self, patch, id):
		Feature.__init__(self, patch, id)
		self.HISTBINNUM = 16
		self.FEATURE_MODEL = np.array([  0.00000000e+00,   1.00000000e+00,   0.00000000e+00,   0.00000000e+00,
										   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
										   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
										   0.00000000e+00,  0.00000000e+00,   0.00000000e+00,  0.00000000e+00,
										   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
										   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
										   0.00000000e+00,   0.00000000e+00,   7.38737277e-04,   4.13136744e-01,
										   5.86124519e-01,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00])
		self.FEATURE_MODEL = np.concatenate((self.FEATURE_MODEL, np.zeros(3*len(range(1,2)))), axis = 1)
		self.FEATURE_MODEL = normalize(self.FEATURE_MODEL, norm='l1')[0] # normalize the FEATURE_MODEL using l1

	def computeFeatureIntegralImage(self, integral_img_obj):
		assert integral_img_obj.integral_image_type == "HS", "in FeatureBottomRightYellow, integral_img_obj used should be HS"

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

		self.hist = np.concatenate((self.patch.HueHistArr[self.BOTTOM_RIGHT_INDEX], \
			self.patch.SaturationHistArr[self.BOTTOM_RIGHT_INDEX]), axis = 1)
	
		self.HISTBINNUM = len(self.patch.HueHistArr[0])
		for i in range(self.TOP_LEFT_INDEX,self.BOTTOM_RIGHT_INDEX + 1):
			if( i != self.BOTTOM_RIGHT_INDEX):
				other_patch_hist = self.deriveSubPatchTargetHueFilteredBySaturationFrom2DArr(\
					sub_patch_index = i, \
					hs_2d_arr = self.patch.hs_2d_arr, \
					target_hue_bins = range(1,2), \
					target_saturation_bins = range(10,13))
				assert (len(other_patch_hist) == len(range(1,2))), \
				"In FeatureBottomRightYellow: other sub patch {i}'s hue response array needs to be of the same length as that of the patch of interest".format( i = i)
				self.hist = np.concatenate((self.hist, other_patch_hist), axis = 1)

		self.hist = normalize(self.hist, norm='l1')[0] # normalize the histogram using l1

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

		self.hist = np.concatenate((self.patch.HueHistArr[self.BOTTOM_RIGHT_INDEX], \
			self.patch.SaturationHistArr[self.BOTTOM_RIGHT_INDEX]), axis = 1)

		"""feature constructor at build phace, not run anymore after feature is built"""
		# model_constructor_hue = np.zeros(len(self.patch.HueHistArr[self.BOTTOM_RIGHT_INDEX]))
		# model_constructor_saturation = np.zeros(len(self.patch.SaturationHistArr[self.BOTTOM_RIGHT_INDEX]))
		# model_constructor_hue[1:2] = self.patch.HueHistArr[self.BOTTOM_RIGHT_INDEX][1:2]
		# model_constructor_saturation[10:13] = self.patch.SaturationHistArr[self.BOTTOM_RIGHT_INDEX][10:13]
		# model_hist = np.concatenate(\
		# 	(normalize(model_constructor_hue, norm='l1')[0], normalize(model_constructor_saturation, norm = "l1")[0]), \
		# 	axis = 1) # the models are normalized seperately each before being concatenated together
		# print "model_hist:\n", model_hist
		# plotStatistics.plotOneGivenHist("", "hue", self.patch.HueHistArr[self.BOTTOM_RIGHT_INDEX], save =False, show = True)
		# plotStatistics.plotOneGivenHist("", "saturation", self.patch.SaturationHistArr[self.BOTTOM_RIGHT_INDEX], save =False, show = True)
		# plotStatistics.plotOneGivenHist("", "self.hist", self.hist, save = False, show = True)
		# plotStatistics.plotOneGivenHist("", "model_hist", model_hist, save = False, show = True)

		self.HISTBINNUM = len(self.patch.HueHistArr[0])

		for i in range(self.TOP_LEFT_INDEX,self.BOTTOM_RIGHT_INDEX + 1):
			if( i != self.BOTTOM_RIGHT_INDEX):
				# other_patch_hue = self.patch.HueHistArr[i][1:2]
				# other_patch_saturation = self.patch.SaturationHistArr[i][10:13]
				# """
				# other patch hue / saturation, 
				# if one of them is not within the targeted range for the sub patch of interest, 
				# then mark as good (all zeros, same as FEATURE_MODEL)
				# """
				# if(np.sum(other_patch_hue) == 0 or np.sum(other_patch_saturation) == 0):
				# 	other_patch_hist = np.zeros(len(other_patch_hue) + len(other_patch_saturation))
				# else:
				# 	other_patch_hist = np.concatenate((other_patch_hue, other_patch_saturation), axis = 1)
				other_patch_hist = self.deriveSubPatchTargetHueFilteredBySaturationFrom2DArr(\
					sub_patch_index = i, \
					hs_2d_arr = self.patch.hs_2d_arr, \
					target_hue_bins = range(1,2), \
					target_saturation_bins = range(10,13))
				assert (len(other_patch_hist) == len(range(1,2))), \
				"In FeatureBottomRightYellow: other sub patch {i}'s hue response array needs to be of the same length as that of the patch of interest".format( i = i)
				self.hist = np.concatenate((self.hist, other_patch_hist), axis = 1)
		# print self.hist
		# plt.plot(self.FEATURE_MODEL)
		# plt.show()

		self.hist = normalize(self.hist, norm='l1')[0] # normalize the histogram using l1

	def featureResponse(self):
		assert (not self.hist is None), "Error in FeatureBottomRightYellow: calling computeScore before the feature hist is computed!"
		assert len(self.hist) == len(self.FEATURE_MODEL), "Error in FeatureBottomRightYellow: feature length is not correctly computed!"
		# high_response_thresh = 0.00
		# count = 0.0
		# for i in range(0, len(self.hist)):
		# 	if(self.hist[i] > high_response_thresh):
		# 		# count += 1
		# 		count += self.hist[i]
		# return count
		dissimilarity = comparePatches.Jensen_Shannon_Divergence(self.hist, self.FEATURE_MODEL) # if use Jensen_Shannon_Divergence, then need to force d to be > 0 in LDA Score
		# dissimilarity = DIST.euclidean(self.hist, self.FEATURE_MODEL)
		return 1.0 / (1.0 + dissimilarity)

	def computeScore(self):
		if(self.score is None):
			self.score = self.featureResponse()
		



