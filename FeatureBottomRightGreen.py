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


class FeatureBottomRightGreen(object):
	def __init__(self, patch):
		self.patch = patch
		self.hist = None
		self.score = None

	### BOTTOM_RIGHT_GREEN ###
	def computeFeature(self, img, useGaussianSmoothing = True):
		if(not (len(self.patch.HueHistArr) == 5 and len(self.patch.SaturationHistArr) == 5)):
			self.patch.HueHistArr = []
			self.patch.SaturationHistArr = []
			self.patch.ValueHistArr = []
			self.patch.computeSeperateHSVHistogram(img, useGaussianSmoothing)
		self.hist = np.concatenate((self.patch.HueHistArr[4][4:5], self.patch.SaturationHistArr[4][8:10]), axis = 1)
		for i in range(1,4):
			self.hist = np.concatenate((self.hist, self.patch.SaturationHistArr[i][1:2]), axis = 1) 
		print "final length of self.BOTTOM_RIGHT_GREENHist:", len(self.hist)

	def featureResponse(self):
		assert (not self.hist is None), "Error in FeatureBottomRightGreen: calling computeScore before the feature hist is computed!"
		high_response_thresh = 0.05
		count = 0.0
		for i in range(0, len(self.hist)):
			if(self.hist[i] > high_response_thresh):
				# count += 1
				count += self.hist[i]
		return count

	def computeScore(self):
		self.score = self.featureResponse()
		
	def setScore(self):
		if(self.score is None):
			self.computeScore()
		self.patch.BOTTOM_RIGHT_GREENScore = self.score