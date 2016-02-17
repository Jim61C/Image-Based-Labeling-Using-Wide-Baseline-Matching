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

class FeatureBorderGreen(Feature):
	"""
	reinforce the inner rectangle to be non-green, thus, append the cutted inner hist (Hue Bin 4:7 indicates green) as well
	"""
	def __init__(self, patch, id):
		Feature.__init__(self, patch, id)
		self.HISTBINNUM = 16
		self.FEATURE_MODEL = np.array([ 0.,          0.,          0.,          0.,          0.04649839,  0.13987842,
										  0.09450414,  0.,          0.,          0.,          0.,          0.,          0.,
										  0.,          0.,          0.        ])
		self.FEATURE_MODEL = np.concatenate((self.FEATURE_MODEL, np.zeros(len(range(4,7)))), axis = 1)
		self.FEATURE_MODEL = normalize(self.FEATURE_MODEL, norm='l1')[0] # normalize the histogram using l1

	def computeHueHist(self,img_hsv, patch,gaussian_window):
		hist = np.zeros(self.HISTBINNUM)
		ref_x = patch.x - patch.size/2
		ref_y = patch.y - patch.size/2
		for i in range(patch.x - patch.size/2, patch.x + patch.size/2 + 1):
			for j in range(patch.y - patch.size/2, patch.y + patch.size/2 + 1):
				this_bin = int(img_hsv[i][j][0]/360.0 * self.HISTBINNUM)
				if (this_bin == self.HISTBINNUM):
					this_bin = self.HISTBINNUM - 1
				hist[this_bin] += 1 * gaussian_window[i - ref_x][j - ref_y]
				# hist[this_bin] += 1
		return hist

	def computeFeature(self, img, useGaussianSmoothing = True):
		img_hsv = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_BGR2HSV)
		# print "hue channel:", img_hsv[:,:,0]
		# print "saturation channel:", img_hsv[:,:,1]
		inner_patch_size = comparePatches.getGaussianScale(self.patch.size, 1.2, -3)
		inner_patch = comparePatches.Patch(self.patch.x, self.patch.y, inner_patch_size)

		gaussian_window = comparePatches.gauss_kernels(self.patch.size, sigma = self.patch.size/4.0)
		inner_gaussian_window = gaussian_window[ gaussian_window.shape[0]/2 - inner_patch.size/2: gaussian_window.shape[0]/2 + inner_patch.size/2 + 1 ,\
		 gaussian_window.shape[1]/2 - inner_patch.size/2: gaussian_window.shape[1]/2 + inner_patch.size/2 + 1]

		assert gaussian_window.shape == (self.patch.size, self.patch.size), "outer gaussian_window size not correct"
		assert inner_gaussian_window.shape == (inner_patch.size, inner_patch.size), "inner gaussian_window size not correct"

		outer_hist = self.computeHueHist(img_hsv, self.patch, gaussian_window)
		inner_hist = self.computeHueHist(img_hsv, inner_patch, inner_gaussian_window)
		border_hist = outer_hist - inner_hist
		# model_constructor = np.zeros(len(border_hist))
		# model_constructor[4:7] = border_hist[4:7]
		# print model_constructor
		self.hist = np.concatenate((border_hist, inner_hist[4:7]), axis = 1)
		self.hist = normalize(self.hist, norm='l1')[0] # normalize the histogram using l1
		
		# comparePatches.drawPatchesOnImg(np.copy(img),[self.patch, inner_patch], True)
		# plotStatistics.plotOneGivenHist("","inner_hist", inner_hist, save = False, show = True)
		# plotStatistics.plotOneGivenHist("","border_hist", border_hist, save = False, show = True)
		# plotStatistics.plotOneGivenHist("","model constructed", model_constructor, save = False, show = True)
		

	def featureResponse(self):
		assert (not self.hist is None), "Error in FeatureBorderGreen: calling computeScore before the feature hist is computed!"
		assert (len(self.hist) == len(self.FEATURE_MODEL)), "Error in FeatureBorderGreen: hist length is not correct!"
		# return np.sum(self.hist)
		return 1.0 / (1.0 + DIST.euclidean(self.hist, self.FEATURE_MODEL))

	def computeScore(self):
		if(self.score is None):
			self.score = self.featureResponse()
	