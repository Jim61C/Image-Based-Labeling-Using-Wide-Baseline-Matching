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
		self.FEATURE_MODEL = np.array([0.19163985,  0.03587214,  0.10104625,  0.07476147,  0.09146382,  0.03582894,
									  0.,          0.04768074,  0.10713325,  0.03140353,  0.0402167,   0.00868779])
		self.FEATURE_MODEL = normalize(self.FEATURE_MODEL, norm='l1')[0] # normalize the FEATURE_MODEL using l1

	def computeFeature(self, img, useGaussianSmoothing = True):
		if(not (len(self.patch.HueHistArr) == 5 and len(self.patch.SaturationHistArr) == 5)):
			self.patch.HueHistArr = []
			self.patch.SaturationHistArr = []
			self.patch.ValueHistArr = []
			self.patch.computeSeperateHSVHistogram(img, useGaussianSmoothing)
		self.hist = np.concatenate((self.patch.HueHistArr[self.BOTTOM_RIGHT_INDEX][1:2], \
			self.patch.SaturationHistArr[self.BOTTOM_RIGHT_INDEX][11:13]), axis = 1)

		for i in range(self.TOP_LEFT_INDEX,self.BOTTOM_RIGHT_INDEX + 1):
			if( i != self.BOTTOM_RIGHT_INDEX):
				self.hist = np.concatenate((self.hist, self.patch.SaturationHistArr[i][1:4]), axis = 1)
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
		# dissimilarity = comparePatches.Jensen_Shannon_Divergence(self.hist, self.FEATURE_MODEL) # if use Jensen_Shannon_Divergence, then need to force d to be > 0 in LDA Score
		dissimilarity = DIST.euclidean(self.hist, self.FEATURE_MODEL)
		return 1.0 / (1.0 + dissimilarity)

	def computeScore(self):
		if(self.score is None):
			self.score = self.featureResponse()
		



