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

class FeatureCentreBlue(Feature):
	"""
	reinforce the outer border to be non-targeted blue hue, i.e.,
	"""
	def __init__(self, patch, id):
		Feature.__init__(self, patch, id)
		self.HISTBINNUM = 16
		
		self.HUE_START_INDEX = 9
		self.HUE_END_INDEX = 10
		self.SATURATION_START_INDEX = 8
		self.SATURATION_END_INDEX = 10
		
		self.FEATURE_MODEL_HUE = np.zeros(self.HISTBINNUM)
		self.FEATURE_MODEL_HUE[self.HUE_START_INDEX] = 1.0 # all in bin 9
		
		self.FEATURE_MODEL_SATURATION = np.zeros(self.HISTBINNUM)
		self.FEATURE_MODEL_SATURATION[self.SATURATION_START_INDEX] = 0.5 # all in bin 8 or bin 9
		self.FEATURE_MODEL_SATURATION[self.SATURATION_START_INDEX + 1] = 0.5 # all in bin 8 or bin 9

		self.FEATURE_MODEL = np.concatenate(( \
			self.FEATURE_MODEL_HUE, \
			self.FEATURE_MODEL_SATURATION, \
			np.zeros(len(range(self.HUE_START_INDEX,self.HUE_END_INDEX)) + \
			len(range(self.SATURATION_START_INDEX,self.SATURATION_END_INDEX)))), axis = 1) # append the expected border response
		self.FEATURE_MODEL = normalize(self.FEATURE_MODEL, norm='l1')[0] # normalize the histogram using l1

	def computeFeature(self, img, useGaussianSmoothing = True):
		img_hsv = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_BGR2HSV)
		# print "hue channel:", img_hsv[:,:,0]
		# print "saturation channel:", img_hsv[:,:,1]
		inner_patch_size = comparePatches.getGaussianScale(self.patch.size, self.GAUSSIAN_SCALE_FACTOR, -3)
		inner_patch = comparePatches.Patch(self.patch.x, self.patch.y, inner_patch_size)

		gaussian_window = comparePatches.gauss_kernels(self.patch.size, sigma = self.patch.size/4.0)
		inner_gaussian_window = gaussian_window[ \
		gaussian_window.shape[0]/2 - inner_patch.size/2: gaussian_window.shape[0]/2 + inner_patch.size/2 + 1 ,\
		gaussian_window.shape[1]/2 - inner_patch.size/2: gaussian_window.shape[1]/2 + inner_patch.size/2 + 1]

		assert gaussian_window.shape == (self.patch.size, self.patch.size), "outer gaussian_window size not correct"
		assert inner_gaussian_window.shape == (inner_patch.size, inner_patch.size), "inner gaussian_window size not correct"

		"""Still need to compute the overall Hue, Saturation, since sigma is different now 4.0 instead of 6.0"""
		if (self.patch.outer_hs_2d_gaus_4 is None):
			self.patch.outer_hs_2d_gaus_4 = self.computeHS2DWithGaussianWindow(img_hsv, self.patch, gaussian_window)

		key = "{gaus}_{scale}".format(gaus = 4, scale = 3)
		if (not key in self.patch.gaus_scale_to_inner_hs_2d_dict):
		 self.patch.gaus_scale_to_inner_hs_2d_dict[key] = self.computeHS2DWithGaussianWindow(\
		 	img_hsv, inner_patch, inner_gaussian_window)
		
		inner_hist_hue = self.derive1DHueFrom2D(self.patch.gaus_scale_to_inner_hs_2d_dict[key])
		inner_hist_saturation = self.derive1DSaturationFrom2D(self.patch.gaus_scale_to_inner_hs_2d_dict[key])

		outer_hist_hue = self.derive1DHueFrom2D(self.patch.outer_hs_2d_gaus_4)
		outer_hist_saturation = self.derive1DSaturationFrom2D(self.patch.outer_hs_2d_gaus_4)

		border_hist_hue = outer_hist_hue - inner_hist_hue
		border_hist_saturation = outer_hist_saturation - inner_hist_saturation

		# model_constructor_inner_saturation = np.zeros(len(inner_hist_saturation))
		# model_constructor_inner_saturation[self.SATURATION_START_INDEX:self.SATURATION_END_INDEX] = \
		# inner_hist_saturation[self.SATURATION_START_INDEX:self.SATURATION_END_INDEX]
		# print "model_constructor_inner_saturation:\n", model_constructor_inner_saturation

		# model_constructor_inner_hue = np.zeros(len(inner_hist_hue))
		# model_constructor_inner_hue[self.HUE_START_INDEX:self.HUE_END_INDEX] = \
		# inner_hist_hue[self.HUE_START_INDEX:self.HUE_END_INDEX]
		# print model_constructor_hue
		# print "model_constructor_inner_hue:\n", model_constructor_inner_saturation

		if(np.sum(border_hist_hue[self.HUE_START_INDEX:self.HUE_END_INDEX]) == 0 or \
			np.sum(border_hist_saturation[self.SATURATION_START_INDEX:self.SATURATION_END_INDEX]) == 0):
			border_hist = np.zeros(len(range(self.HUE_START_INDEX,self.HUE_END_INDEX)) + \
				len(range(self.SATURATION_START_INDEX,self.SATURATION_END_INDEX)))
		else:
			border_hist = np.concatenate((border_hist_hue[self.HUE_START_INDEX:self.HUE_END_INDEX], \
				border_hist_saturation[self.SATURATION_START_INDEX:self.SATURATION_END_INDEX]), axis = 1)
		self.hist = np.concatenate((inner_hist_hue, inner_hist_saturation, border_hist), axis = 1)
		self.hist = normalize(self.hist, norm='l1')[0] # normalize the histogram using l1
		
		# comparePatches.drawPatchesOnImg(np.copy(img),[self.patch, inner_patch], True)
		# plotStatistics.plotOneGivenHist("","inner_hist_hue", inner_hist_hue, save = False, show = True)
		# plotStatistics.plotOneGivenHist("","border_hist_hue", border_hist_hue, save = False, show = True)
		# plotStatistics.plotOneGivenHist("", "inner_hist_saturation", inner_hist_saturation, save = False, show = True)
		# plotStatistics.plotOneGivenHist("", "border_hist_saturation", border_hist_saturation, save = False, show = True)
		# plotStatistics.plotOneGivenHist("", "self.hist", self.hist, save = False, show = True)
		# plotStatistics.plotOneGivenHist("", "FEATURE_MODEL", self.FEATURE_MODEL, save = False, show = True)
		

	def featureResponse(self):
		assert (not self.hist is None), "Error in FeatureCentreBlue: calling computeScore before the feature hist is computed!"
		assert (len(self.hist) == len(self.FEATURE_MODEL)), "Error in FeatureCentreBlue: hist length is not correct!" + \
		"len(self.hist): {self_his_len}, len(self.FEATURE_MODEL): {feature_model_len}".format(self_his_len = len(self.hist), feature_model_len = len(self.FEATURE_MODEL))
		# return np.sum(self.hist)
		return 1.0 / (1.0 + DIST.euclidean(self.hist, self.FEATURE_MODEL))

	def computeScore(self):
		if(self.score is None):
			self.score = self.featureResponse()
	