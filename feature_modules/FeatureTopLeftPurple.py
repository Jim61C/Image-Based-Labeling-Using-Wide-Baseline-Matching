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
from feature_modules import utils
from feature_modules import Feature

class FeatureTopLeftPurple(Feature):
	def __init__(self, patch, id):
		Feature.__init__(self, patch, id)
		self.FEATURE_MODEL = np.array([ 0.,          0.,          0.,          0.,          0.,          0.,          0.,
										  0.,          0.,          0.,          0.,          0.,          0.,          0.,
										  0.,          1.,          0.,          0.,          0.,          0.19443516,
										  0.23649953,  0.13200154,  0.23102906,  0.15323535,  0.05279936,  0. ,         0.,
										  0.,          0.,          0. ,         0. ,         0.        ])

		# reinforce that the other sub patches must not have the hue/saturation for the target color
		self.FEATURE_MODEL = np.concatenate((self.FEATURE_MODEL, np.zeros(3 *(len(range(15,16)) + len(range(3,9))))), axis = 1)
		self.FEATURE_MODEL = normalize(self.FEATURE_MODEL, norm='l1')[0] # normalize the FEATURE_MODEL using l1
	
	def computeFeature(self, img, useGaussianSmoothing = True):
		if(not (len(self.patch.HueHistArr) == 5 and len(self.patch.SaturationHistArr) == 5)):
			self.patch.HueHistArr = []
			self.patch.SaturationHistArr = []
			self.patch.ValueHistArr = []
			self.patch.computeSeperateHSVHistogram(img, useGaussianSmoothing)
		self.hist = np.concatenate((self.patch.HueHistArr[self.TOP_LEFT_INDEX], \
			self.patch.SaturationHistArr[self.TOP_LEFT_INDEX]), axis = 1)

		"""feature constructor at build phace, not run anymore after feature is built"""
		# model_constructor_hue = np.zeros(len(self.patch.HueHistArr[self.TOP_LEFT_INDEX]))
		# model_constructor_saturation = np.zeros(len(self.patch.SaturationHistArr[self.TOP_LEFT_INDEX]))
		# model_constructor_hue[15:16] = self.patch.HueHistArr[self.TOP_LEFT_INDEX][15:16]
		# model_constructor_saturation[3:9] = self.patch.SaturationHistArr[self.TOP_LEFT_INDEX][3:9]
		# model_hist = np.concatenate(\
		# 	(normalize(model_constructor_hue, norm='l1')[0], normalize(model_constructor_saturation, norm = "l1")[0]), \
		# 	axis = 1) # the models are normalized seperately each before being concatenated together
		# print "model_hist:\n", model_hist
		# plotStatistics.plotOneGivenHist("", "hue", self.patch.HueHistArr[self.TOP_LEFT_INDEX], save =False, show = True)
		# plotStatistics.plotOneGivenHist("", "saturation", self.patch.SaturationHistArr[self.TOP_LEFT_INDEX], save =False, show = True)
		# plotStatistics.plotOneGivenHist("", "self.hist", self.hist, save = False, show = True)
		# plotStatistics.plotOneGivenHist("", "model_hist", model_hist, save = False, show = True)
		
		for i in range(self.TOP_LEFT_INDEX, self.BOTTOM_RIGHT_INDEX + 1):
			if(i != self.TOP_LEFT_INDEX):
				other_patch_hue = self.patch.HueHistArr[i][15:16]
				other_patch_saturation = self.patch.SaturationHistArr[i][3:9]
				"""
				other patch hue / saturation, 
				if one of them is not within the targeted range for the sub patch of interest, 
				then mark as good (all zeros, same as FEATURE_MODEL)
				"""
				if(np.sum(other_patch_hue) == 0 or np.sum(other_patch_saturation) == 0):
					other_patch_hist = np.zeros(len(other_patch_hue) + len(other_patch_saturation))
				else:
					other_patch_hist = np.concatenate((other_patch_hue, other_patch_saturation), axis = 1)

				self.hist = np.concatenate((self.hist, other_patch_hist), axis = 1)

		self.hist = normalize(self.hist, norm='l1')[0] # normalize the histogram using l1

	def featureResponse(self):
		"""
		d_euclidean^2 = 2(1- cos(A, B)), if vector A, B are normalized: |A|, |B| = 1
		Thus, 0 <= d_euclidean <= sqrt(2) for normalized vector A, B, and further, A, B both >=0, cos(A,B) >=0

		0 <= Jensen_Shannon_Divergence (A, B) <= inf

		d_cosine = 1 - cos(A,B), since cos(A,B) >=0 
		Thus, 0 <= d_cosine <= 1, let cos_score = (1 - d_cosine), then 0 <= cos_score <= 1 
		"""
		assert (not self.hist is None), "Error in FeatureTopLeftPurple: calling computeScore before the feature hist is computed!"
		assert len(self.hist) == len(self.FEATURE_MODEL), "Error in FeatureTopLeftPurple: feature length is not correctly computed!"
		# high_response_thresh = 0.00
		# count = 0.0
		# for i in range(0, len(self.hist)):
		# 	if(self.hist[i] > high_response_thresh):
		# 		# count += 1
		# 		count += self.hist[i]
		# return count
		# dissimilarity = comparePatches.Jensen_Shannon_Divergence(self.hist, self.FEATURE_MODEL) # if use measure other than Eculidean distance, then need to force d to be > 0 in LDA Score
		dissimilarity = DIST.euclidean(self.hist, self.FEATURE_MODEL)
		return 1.0 / (1.0 + dissimilarity)

	def computeScore(self):
		"""
		score needs to be of scale [0, 1] as min, max value range
		"""
		if(self.score is None):
			self.score = utils.converScaleTo01(self.featureResponse(), min = utils.MIN_RAW_EUCLIDEAN_SCORE, max = utils.MAX_RAW_EUCLIDEAN_SCORE)



