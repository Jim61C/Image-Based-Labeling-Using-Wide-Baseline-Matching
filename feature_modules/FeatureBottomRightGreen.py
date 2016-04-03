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


class FeatureBottomRightGreen(Feature):
	def __init__(self, patch, id):
		Feature.__init__(self, patch, id)
		# the distribution of hue and saturation needed for the subpatch of interest
		# self.FEATURE_MODEL = np.array([ 0.,          0.,          0.,          0.08106389,  0.11229446,  0.,          0.,
		# 								  0.,          0.,          0.,          0.,          0.,          0.,          0.,
		# 								  0.,          0.,          0.,          0.,          0.,          0.,          0.,
		# 								  0.,          0.,          0.02104336,  0.0664095,   0.02171191,  0.,          0.,
		# 								  0.,          0.,          0.,          0.        ]) 

		# hue model and saturation model normalized seperately:
		self.FEATURE_MODEL = np.array([ 0.,          0.,          0.,          0.15622393,  0.84377607,  0.,          0.,
										  0.,          0.,          0.,          0.,          0.,          0.,          0.,
										  0.,          0.,          1.0])
		# reinforce that the other sub patches must not have the hue/saturation for the green color
		self.FEATURE_MODEL = np.concatenate((self.FEATURE_MODEL, np.zeros(3 *(len(range(3,5))))), axis = 1)
		self.FEATURE_MODEL = normalize(self.FEATURE_MODEL, norm='l1')[0] # normalize the FEATURE_MODEL using l1

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
			
		# print "sum of HueHistArr[4]:", np.sum(self.patch.HueHistArr[4])
		# print "sum of SaturationHistArr[4]:", np.sum(self.patch.SaturationHistArr[4])
		self.hist = np.concatenate( \
			(self.patch.HueHistArr[4], \
			np.array([np.sum(self.patch.SaturationHistArr[4][7:10])]) \
			), axis = 1)
		"""Better to use sum(target_saturation_bins) = 1 instead of use the actual saturation hist"""
		
		"""feature constructor at build phace, not run anymore after feature is built"""
		# model_constructor_hue = np.zeros(len(self.patch.HueHistArr[4]))
		# model_constructor_saturation = np.zeros(len(self.patch.SaturationHistArr[4]))
		# model_constructor_hue[4:5] = self.patch.HueHistArr[4][4:5]
		# model_constructor_saturation[7:10] = self.patch.SaturationHistArr[4][7:10]
		# model_hist = np.concatenate(\
		# 	(normalize(model_constructor_hue, norm = "l1")[0], normalize(model_constructor_saturation, norm = "l1")[0]), \
		# 	axis = 1) # no need to normalized the two models each as they will be normalized altogether later
		# print model_hist
		# plotStatistics.plotOneGivenHist("", "FeatureBottomRightGreen", self.hist, save = False, show = True)
		# plotStatistics.plotOneGivenHist("", "self.FEATURE_MODEL", self.FEATURE_MODEL, save = False, show = True)
		
		self.HISTBINNUM = len(self.patch.HueHistArr[0])
		for i in range(1,4):
			# other_patch_hue = self.patch.HueHistArr[i][4:5]
			# other_patch_saturation = self.patch.SaturationHistArr[i][7:10]
			# """other patch hue / saturation at least one is not within green range"""
			# if(np.sum(other_patch_hue) == 0 or np.sum(other_patch_saturation) == 0):
			# 	other_patch_hist = np.zeros(len(other_patch_hue) + len(other_patch_saturation))
			# else:
			# 	other_patch_hist = np.concatenate((other_patch_hue, other_patch_saturation), axis = 1)
			
			# plotStatistics.plotOneGivenHist("", "other_patch_hue[{i}]".format(i = i), self.patch.HueHistArr[i], save = False, show = True)
			# plotStatistics.plotOneGivenHist("", "other_patch_saturation[{i}]".format(i = i), self.patch.SaturationHistArr[i], save = False, show = True)
			other_patch_hist = self.deriveSubPatchTargetHueFilteredBySaturationFrom2DArr(\
				img_hsv, \
				sub_patch_index = i, \
				hs_2d_arr = self.patch.hs_2d_arr, \
				target_hue_bins = range(3,5), \
				target_saturation_bins = range(7,10))
			# plotStatistics.plotOneGivenHist("", "other_patch_hist[{i}]".format(i = i), other_patch_hist, save = False, show = True)
			assert (len(other_patch_hist) == len(range(3,5))), \
			"other_patch_hist {i}".format(i = i) + " must be having same length as target hue bin range"
			self.hist = np.concatenate((self.hist, other_patch_hist), axis = 1)
			
		self.hist = normalize(self.hist, norm='l1')[0] # normalize the histogram using l1
		
	def featureResponse(self):
		assert (not self.hist is None), "Error in FeatureBottomRightGreen: calling computeScore before the feature hist is computed!"
		assert len(self.hist) == len(self.FEATURE_MODEL), "Error in FeatureBottomRightGreen: feature length is not correctly computed!"
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




