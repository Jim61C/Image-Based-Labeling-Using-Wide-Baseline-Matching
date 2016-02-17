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
from feature_modules import utils


class FeatureTopRightYellow(Feature):
	def __init__(self, patch, id):
		Feature.__init__(self, patch, id)
		self.FEATURE_MODEL = np.array([ 0.,          1.,          0.,          0.,          0. ,         0. ,         0.,
										  0.,          0.,          0.,          0.,          0. ,         0.,          0.,
										  0.,          0.,          0.,          0.,          0.,          0.,          0.,
										  0. ,         0.,          0. ,         0.,          0.,          0.34,
										  0.33,        0.33,        0.,        0.,          0.        ])
		self.FEATURE_MODEL = np.concatenate((self.FEATURE_MODEL, np.zeros(3*len(range(1,2)))), axis = 1)
		self.FEATURE_MODEL = normalize(self.FEATURE_MODEL, norm='l1')[0] # normalize the FEATURE_MODEL using l1

	def computeFeature(self, img, useGaussianSmoothing = True):
		if(not (len(self.patch.HueHistArr) == 5 and len(self.patch.SaturationHistArr) == 5)):
			self.patch.HueHistArr = []
			self.patch.SaturationHistArr = []
			self.patch.ValueHistArr = []
			self.patch.computeSeperateHSVHistogram(img, useGaussianSmoothing)

		
		self.hist = np.concatenate((self.patch.HueHistArr[self.TOP_RIGHT_INDEX], \
			self.patch.SaturationHistArr[self.TOP_RIGHT_INDEX]), axis = 1)

		"""feature constructor at build phace, not run anymore after feature is built"""
		# model_constructor_hue = np.zeros(len(self.patch.HueHistArr[self.TOP_RIGHT_INDEX]))
		# model_constructor_saturation = np.zeros(len(self.patch.SaturationHistArr[self.TOP_RIGHT_INDEX]))
		# model_constructor_hue[1:2] = self.patch.HueHistArr[self.TOP_RIGHT_INDEX][1:2]
		# model_constructor_saturation[10:13] = self.patch.SaturationHistArr[self.TOP_RIGHT_INDEX][10:13]
		# model_hist = np.concatenate(\
		# 	(normalize(model_constructor_hue, norm='l1')[0], normalize(model_constructor_saturation, norm = "l1")[0]), \
		# 	axis = 1) # the models are normalized seperately each before being concatenated together
		# print "model_hist:\n", model_hist
		# plotStatistics.plotOneGivenHist("", "hue", self.patch.HueHistArr[self.TOP_RIGHT_INDEX], save =False, show = True)
		# plotStatistics.plotOneGivenHist("", "saturation", self.patch.SaturationHistArr[self.TOP_RIGHT_INDEX], save =False, show = True)
		# plotStatistics.plotOneGivenHist("", "self.hist", self.hist, save = False, show = True)
		# plotStatistics.plotOneGivenHist("", "self.FEATURE_MODEL", self.FEATURE_MODEL, save = False, show = True)

		img_hsv = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_BGR2HSV)
		self.HISTBINNUM = len(self.patch.HueHist)

		for i in range(self.TOP_LEFT_INDEX, self.BOTTOM_RIGHT_INDEX + 1):
			if( i != self.TOP_RIGHT_INDEX):
				other_patch_hist = self.getSubPatchTargetHueFilteredBySaturation(\
					img_hsv, i, range(1,2), range(10,13))
				assert (len(other_patch_hist) == len(range(1,2))), \
				"In FeatureTopRightYellow: other sub patch {i}'s hue response array needs to be of the same length as that of the patch of interest".format( i = i)
				self.hist = np.concatenate((self.hist, other_patch_hist), axis = 1)

		self.hist = normalize(self.hist, norm='l1')[0] # normalize the histogram using l1

	def featureResponse(self):
		assert (not self.hist is None), "Error in FeatureTopRightYellow: calling computeScore before the feature hist is computed!"
		assert len(self.hist) == len(self.FEATURE_MODEL), "Error in FeatureTopRightYellow: feature length is not correctly computed!"
		# high_response_thresh = 0.00
		# count = 0.0
		# for i in range(0, len(self.hist)):
		# 	if(self.hist[i] > high_response_thresh):
		# 		# count += 1
		# 		count += self.hist[i]
		# return count
		# dissimilarity = comparePatches.Jensen_Shannon_Divergence(self.hist, self.FEATURE_MODEL) # if use Jensen_Shannon_Divergence, then need to force d to be > 0 in LDA Score
		dissimilarity = DIST.euclidean(self.hist, self.FEATURE_MODEL)
		return 1.0 / (1.0 + dissimilarity)

	def computeScore(self):
		if(self.score is None):
			self.score = utils.converScaleTo01(self.featureResponse(), min = utils.MIN_RAW_EUCLIDEAN_SCORE, max = utils.MAX_RAW_EUCLIDEAN_SCORE)



