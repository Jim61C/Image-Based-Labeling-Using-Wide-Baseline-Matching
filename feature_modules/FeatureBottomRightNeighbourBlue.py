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


class FeatureBottomRightNeighbourBlue(Feature):
	def __init__(self, patch, id):
		Feature.__init__(self, patch, id)
		self.HUE_MODEL = np.array([0.,         0.00091991,  0.01559402,  0.06014748,  0.20452348,  0.55954024,
						  0.15927488,  0.,         0.,          0.,          0.,          0.,         0.,
						  0.,          0.,          0.        ])
		self.HUE_MODEL = normalize(self.HUE_MODEL, norm='l1')[0]

	def withInImage(self, patch, img):
		if (patch.x - patch.size/2 >= 0 and \
			patch.y - patch.size/2 >= 0 and \
			patch.x + patch.size/2 < img.shape[0] and \
			patch.y + patch.size/2 < img.shape[1]):
			return True
		else:
			return False

	def computeFeature(self, img, useGaussianSmoothing = True):
		neighbour = comparePatches.Patch(self.patch.x + self.patch.size/2, self.patch.y + self.patch.size/2, \
			comparePatches.getGaussianScale(self.patch.size, 1.2, -3))
		if(self.withInImage(neighbour,img)):
			neighbour.computeHSVHistogram(img) # compute seperate H,S,V with gaussian smoothing
			self.hist = neighbour.HueHist
			self.hist = normalize(self.hist, norm='l1')[0]
			# plotStatistics.plotColorHistogram(neighbour, img, "", "neighbour", save = False, show = True)


	def computeScore(self):
		if(self.score is None):
			if(self.hist is None): # if the neighhour hist does not exist, should not consider this patch under this feature
				self.score = -sys.maxint
			else:
				# self.score = 1.0/(1.0 + DIST.euclidean(self.hist, self.HUE_MODEL))
				# use Jensen_Shannon_Divergence since the comparison is between two complete histograms
				self.score = 1.0/(1.0 + comparePatches.Jensen_Shannon_Divergence(self.hist, self.HUE_MODEL))

	def dissimilarityWith(self, hist):
		if(self.hist is None or hist is None):
			return sys.maxint
		else:
			self.assertHist(hist)
			return comparePatches.Jensen_Shannon_Divergence(self.hist, hist)


