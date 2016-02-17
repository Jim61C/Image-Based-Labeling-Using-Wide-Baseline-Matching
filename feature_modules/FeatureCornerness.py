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


class FeatureCornerness(Feature):
	def __init__(self, patch, id):
		Feature.__init__(self, patch, id)
		self.max_response = None
		self._checkRep()

	def computeFeature(self, img, max_response, useGaussianSmoothing = True):
		gx, gy = cornerResponse.sobelConvolutionOpencv(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.uint8))
		Ixx = np.multiply(gx,gx)
		Ixy = np.multiply(gx,gy)
		Iyy = np.multiply(gy,gy)
		self.hist = np.zeros(1)
		self.hist[0] = cornerResponse.getOnePatchHarrisCornerResponse(Ixx, Ixy, Iyy, self.patch, \
			comparePatches.gauss_kernels(self.patch.size, sigma = self.patch.size/6.0))
		self.max_response = max_response
		print "computeFeature for FeatureCornerness done"


	def featureResponse(self):
		assert (not self.hist is None), "Error in FeatureCornerness: calling computeScore before the feature hist is computed!"
		assert (not self.max_response is None), "Error in FeatureCornerness: max_response is needed to compute the score!"
		if (self.hist[0] > 0): # cornerness must have positive response
			return self.hist[0]/self.max_response
		else:
			return 0.0

	def computeScore(self):
		"""
		set self.score
		"""
		if(self.score is None):
			self.score = self.featureResponse()

	def _checkRep(self):
		assert (self.id == utils.CORNERNESS_FEATURE_ID), "Error in FeatureCornerness: id is not correctly set: {id}".format(id = self.id)



