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
import cornerResponse


class FeatureSharpHOG(Feature):
	def __init__(self, patch, id):
		Feature.__init__(self, patch, id)
		# other than the cutted bins of wanted HOG direction, all other HOG bins' values are zero
		self.FEATURE_MODEL = np.array([ 0.,          0.,          0.,          0. ,         0. ,         0. ,         0.,
										  0.  ,        0.,          0.    ,      0. ,         0.05131246 , 0.05356285,
										  0. ,         0.067983,    0.04214793,  0.,          0.  ,        0.  ,        0.,
										  0. ,         0.,          0.06094781,  0.,          0.  ,        0.  ,        0.,
										  0.,          0.,          0.1160895,   0. ,         0.  ,        0.   ,       0.,
										  0. ,         0.        ])
		self.FEATURE_MODEL = normalize(self.FEATURE_MODEL, norm='l1')[0] # normalize the histogram using l1
		self._checkRep()

	def computeFeature(self, img, useGaussianSmoothing = True):
		if(self.patch.HOG_Uncirculated is None):
			self.patch.computeHOG(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
		uncircualted = normalize(self.patch.HOG_Uncirculated, norm='l1')[0] 
		# plotStatistics.plotOneGivenHist("","uncircualted", uncircualted, save = False, show = True)

		# model_constructor = np.zeros(len(uncircualted))
		# model_constructor[11:13] = uncircualted[11:13]
		# model_constructor[14:16] = uncircualted[14:16]
		# model_constructor[22:23] = uncircualted[22:23]
		# model_constructor[29:30] = uncircualted[29:30]
		# print "model_constructor:", model_constructor

		# self.hist = np.concatenate((uncircualted[11:13], uncircualted[14:16], uncircualted[22:23], uncircualted[29:30]), axis = 1)
		# self.hist = normalize(self.hist, norm='l1')[0] # normalize the histogram using l1
		self.hist = uncircualted
		return


	def featureResponse(self):
		assert (not self.hist is None), "Error in FeatureSharpHOG: calling computeScore before the feature hist is computed!"
		assert (len(self.hist) == len(self.FEATURE_MODEL)), "Error in FeatureSharpHOG: hist length is not correct!"
		dissimilarity = comparePatches.Jensen_Shannon_Divergence(self.hist, self.FEATURE_MODEL)
		return 1.0 / (1.0 + dissimilarity)
		# return np.sum(self.hist) # if all HOG degree are at the cutted HOG bins, response will be 1.0

	def computeScore(self):
		"""
		set self.score
		"""
		if(self.score is None):
			self.score = self.featureResponse()

	def _checkRep(self):
		assert (self.id == utils.SHARP_HOG_FEATURE_ID), "Error in FeatureSharpHOG: id is not correctly set: {id}".format(id = self.id)



