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
		self.FEATURE_MODEL = np.array([0.1923072,  0.03623447, 0.04407349, 0.02459949, 0.04305403, 0.02855658,
							  0.00983956, 0.01349747, 0.12322526, 0.07804763, 0.0785877,  0.10341255,
							  0.0172135,  0.10211276, 0.08209506, 0.02457068])

		self.FEATURE_MODEL = normalize(self.FEATURE_MODEL, norm='l1')[0] # normalize the FEATURE_MODEL using l1
	
	def computeFeature(self, img, useGaussianSmoothing = True):
		if(not (len(self.patch.HueHistArr) == 5 and len(self.patch.SaturationHistArr) == 5)):
			self.patch.HueHistArr = []
			self.patch.SaturationHistArr = []
			self.patch.ValueHistArr = []
			self.patch.computeSeperateHSVHistogram(img, useGaussianSmoothing)
		self.hist = np.concatenate((self.patch.HueHistArr[self.TOP_LEFT_INDEX][15:16], \
			self.patch.SaturationHistArr[self.TOP_LEFT_INDEX][3:9]), axis = 1)
		for i in range(self.TOP_LEFT_INDEX, self.BOTTOM_RIGHT_INDEX + 1):
			if(i != self.TOP_LEFT_INDEX):
				self.hist = np.concatenate((self.hist, self.patch.SaturationHistArr[i][0:3]), axis = 1)
		self.hist = normalize(self.hist, norm='l1')[0] # normalize the histogram using l1

	def featureResponse(self):
		"""
		d_euclidean^2 = 2(1- cos(A, B)), if vector A, B are normalized: |A|, |B| = 1
		Thus, 0 <= d_euclidean <= 2 for normalized vector A, B

		0 <= Jensen_Shannon_Divergence (A, B) <= inf

		d_cosine = 1 - cos(A,B)
		Thus, 0 <= d_cosine <= 2, let cos_score = (2 - d_cosine) / 2, then 0 <= cos_score <= 1 
		"""
		assert (not self.hist is None), "Error in FeatureTopLeftPurple: calling computeScore before the feature hist is computed!"
		# assert len(self.hist) == len(self.FEATURE_MODEL), "Error in FeatureTopLeftPurple: feature length is not correctly computed!"
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



