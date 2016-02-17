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

class FeatureCentreYellow(Feature):
	"""
	reinforce the outer border to be non-targeted yellow hue, i.e., hue does not contain 0.13-0.18, (bin 2 out of 16)
	TODO: add inner patch's saturation hist (bin 7:9) as well, not just Hue 
	"""
	def __init__(self, patch, id):
		Feature.__init__(self, patch, id)
		self.HISTBINNUM = 16
		self.FEATURE_MODEL_HUE = np.array([ 0.,          0.,          0.44326144,  0.,          0.,          0.,          0.,
										  0.,          0.,          0.,          0.,          0. ,         0.,          0.,
										  0.,          0.        ])
		self.FEATURE_MODEL_SATURATION = np.array([ 0.,          0.,          0.,          0.,          0. ,         0.,          0.,
										  0.21110128,  0.21070874,  0.,          0.,          0. ,         0.,          0.,
										  0.,          0.        ])
		self.FEATURE_MODEL = np.concatenate(( \
			self.FEATURE_MODEL_HUE, \
			self.FEATURE_MODEL_SATURATION, \
			np.zeros(len(range(2,3)) + len(range(7,9)))), axis = 1) # append the expected border response
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

	def computeSaturationHist(self, img_hsv, patch, gaussian_window):
		"""
		img_hsv: Hue: 0-360, Saturation: 0-1, Value: 0-1
		"""
		hist = np.zeros(self.HISTBINNUM)
		ref_x = patch.x - patch.size/2
		ref_y = patch.y - patch.size/2
		for i in range(patch.x - patch.size/2, patch.x + patch.size/2 + 1):
			for j in range(patch.y - patch.size/2, patch.y + patch.size/2 + 1):
				this_bin = int(img_hsv[i][j][1]/1.0 * self.HISTBINNUM)
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

		outer_hist_hue = self.computeHueHist(img_hsv, self.patch, gaussian_window)
		outer_hist_saturation = self.computeSaturationHist(img_hsv, self.patch, gaussian_window)

		inner_hist_hue = self.computeHueHist(img_hsv, inner_patch, inner_gaussian_window)
		inner_hist_saturation = self.computeSaturationHist(img_hsv, inner_patch, inner_gaussian_window)

		border_hist_hue = outer_hist_hue - inner_hist_hue
		border_hist_saturation = outer_hist_saturation - inner_hist_saturation

		# model_constructor_inner_saturation = np.zeros(len(inner_hist_saturation))
		# model_constructor_inner_saturation[7:9] = inner_hist_saturation[7:9]
		# print "model_constructor_inner_saturation:\n", model_constructor_inner_saturation

		# model_constructor = np.zeros(len(border_hist_hue))
		# model_constructor[2:3] = inner_hist_hue[2:3]
		# print model_constructor

		if(np.sum(border_hist_hue[2:3]) == 0 or np.sum(border_hist_saturation[7:9]) == 0):
			border_hist = np.zeros(len(range(2,3)) + len(range(7,9)))
		else:
			border_hist = np.concatenate((border_hist_hue[2:3], border_hist_saturation[7:9]), axis = 1)
		self.hist = np.concatenate((inner_hist_hue, inner_hist_saturation, border_hist), axis = 1)
		self.hist = normalize(self.hist, norm='l1')[0] # normalize the histogram using l1
		
		# comparePatches.drawPatchesOnImg(np.copy(img),[self.patch, inner_patch], True)
		# plotStatistics.plotOneGivenHist("","inner_hist", inner_hist_hue, save = False, show = True)
		# plotStatistics.plotOneGivenHist("","border_hist", border_hist_hue, save = False, show = True)
		# plotStatistics.plotOneGivenHist("","model constructed", model_constructor, save = False, show = True)
		# plotStatistics.plotOneGivenHist("", "inner_hist_saturation", inner_hist_saturation, save = False, show = True)
		# plotStatistics.plotOneGivenHist("", "border_hist_saturation", border_hist_saturation, save = False, show = True)
		# plotStatistics.plotOneGivenHist("", "self.hist", self.hist, save = False, show = True)
		

	def featureResponse(self):
		assert (not self.hist is None), "Error in FeatureCentreYellow: calling computeScore before the feature hist is computed!"
		assert (len(self.hist) == len(self.FEATURE_MODEL)), "Error in FeatureCentreYellow: hist length is not correct!"
		# return np.sum(self.hist)
		return 1.0 / (1.0 + DIST.euclidean(self.hist, self.FEATURE_MODEL))

	def computeScore(self):
		if(self.score is None):
			# self.score = utils.converScaleTo01(self.featureResponse(), utils.MIN_RAW_EUCLIDEAN_SCORE, utils.MAX_RAW_EUCLIDEAN_SCORE)
			self.score = self.featureResponse()
	