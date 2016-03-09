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

class FeatureCentreParadigm(Feature):
	"""
	reinforce the outer border to be non-targeted blue hue, i.e.,
	"""
	def __init__(self, patch, id):
		Feature.__init__(self, patch, id)
		self.HISTBINNUM = 36
		self.GAUSSIAN_SCALE = 3
		
		self.HUE_START_INDEX = None
		self.HUE_END_INDEX = None
		self.SATURATION_START_INDEX = None
		self.SATURATION_END_INDEX = None
		
		self.FEATURE_MODEL_HUE = np.zeros(self.HISTBINNUM)
		
		self.FEATURE_MODEL_SATURATION = np.zeros(self.HISTBINNUM)

		self.FEATURE_MODEL = np.concatenate(( \
			self.FEATURE_MODEL_HUE, \
			self.FEATURE_MODEL_SATURATION, \
			np.zeros(len(range(self.HUE_START_INDEX,self.HUE_END_INDEX)) + \
			len(range(self.SATURATION_START_INDEX,self.SATURATION_END_INDEX)))), axis = 1) # append the expected border response
		self.FEATURE_MODEL = normalize(self.FEATURE_MODEL, norm='l1')[0] # normalize the histogram using l1

	def computeFeature(self, img, useGaussianSmoothing = True):
		self.HISTBINNUM = 16
		img_hsv = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_BGR2HSV)
		# print "hue channel:", img_hsv[:,:,0]
		# print "saturation channel:", img_hsv[:,:,1]
		inner_patch_size = comparePatches.getGaussianScale(self.patch.size, self.GAUSSIAN_SCALE_FACTOR, -self.GAUSSIAN_SCALE)
		inner_patch = comparePatches.Patch(self.patch.x, self.patch.y, inner_patch_size)

		gaussian_window = comparePatches.gauss_kernels(self.patch.size, sigma = self.patch.size/4.0)
		inner_gaussian_window = gaussian_window[ \
		gaussian_window.shape[0]/2 - inner_patch.size/2: gaussian_window.shape[0]/2 + inner_patch.size/2 + 1 ,\
		gaussian_window.shape[1]/2 - inner_patch.size/2: gaussian_window.shape[1]/2 + inner_patch.size/2 + 1]

		assert gaussian_window.shape == (self.patch.size, self.patch.size), "outer gaussian_window size not correct"
		assert inner_gaussian_window.shape == (inner_patch.size, inner_patch.size), "inner gaussian_window size not correct"

		"""Still need to compute the overall Hue, Saturation, since sigma is different now 4.0 instead of 6.0"""
		if (self.patch.outer_hue_hist_scale_3_gaus_4 is None):
			self.patch.outer_hue_hist_scale_3_gaus_4 = self.computeHueHist(img_hsv, self.patch, gaussian_window)
		if (self.patch.outer_saturation_hist_scale_3_gaus_4 is None):
			self.patch.outer_saturation_hist_scale_3_gaus_4 = self.computeSaturationHist(img_hsv, self.patch, gaussian_window)
		if (self.patch.inner_hue_hist_scale_3_gaus_4 is None):
			self.patch.inner_hue_hist_scale_3_gaus_4 = self.computeHueHist(img_hsv, inner_patch, inner_gaussian_window)
		if (self.patch.inner_saturation_hist_scale_3_gaus_4 is None):
			self.patch.inner_saturation_hist_scale_3_gaus_4 = self.computeSaturationHist(img_hsv, inner_patch, inner_gaussian_window)

		outer_hist_hue = self.patch.outer_hue_hist_scale_3_gaus_4
		outer_hist_saturation = self.patch.outer_saturation_hist_scale_3_gaus_4

		inner_hist_hue = self.patch.inner_hue_hist_scale_3_gaus_4
		inner_hist_saturation = self.patch.inner_saturation_hist_scale_3_gaus_4

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
		assert (not self.hist is None), "Error in FeatureCentreParadigm: calling computeScore before the feature hist is computed!"
		assert (len(self.hist) == len(self.FEATURE_MODEL)), "Error in FeatureCentreParadigm: hist length is not correct!" + \
		"len(self.hist): {self_his_len}, len(self.FEATURE_MODEL): {feature_model_len}".format(\
			self_his_len = len(self.hist), feature_model_len = len(self.FEATURE_MODEL))
		return 1.0 / (1.0 + DIST.euclidean(self.hist, self.FEATURE_MODEL))

	def computeScore(self):
		if(self.score is None):
			self.score = self.featureResponse()


	def fitParadigm(self, img):
		"""
		1. checks the inner hue, saturation hist, finds the mode color and see whether that color constitutes the majority of the inner patch
		2. checks the border hue, saturation hist, make sure that the border does not have the found inner hue color
		3. check for different inner GAUSSIAN_SCALE, default is 3
		returns true or false
		"""
		self.HUEFRACTION = 0.7
		self.SATURATIONFRACTION_INVERSE = 0.1
		self.SHRINK_HUE_BIN_FRACTION = 0.9

		self.HISTBINNUM = 36

		img_hsv = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_BGR2HSV)
		gaussian_window = comparePatches.gauss_kernels(self.patch.size, sigma = self.patch.size/4.0)

		inner_patch_size = comparePatches.getGaussianScale(self.patch.size, self.GAUSSIAN_SCALE_FACTOR, -self.GAUSSIAN_SCALE)
		inner_patch = comparePatches.Patch(self.patch.x, self.patch.y, inner_patch_size)
		inner_gaussian_window = gaussian_window[ \
		gaussian_window.shape[0]/2 - inner_patch.size/2: gaussian_window.shape[0]/2 + inner_patch.size/2 + 1 ,\
		gaussian_window.shape[1]/2 - inner_patch.size/2: gaussian_window.shape[1]/2 + inner_patch.size/2 + 1]

		outer_hue = self.computeHueHist(img_hsv, self.patch, gaussian_window)
		outer_saturation = self.computeSaturationHist(img_hsv, self.patch, gaussian_window)

		inner_hue = self.computeHueHist(img_hsv, inner_patch, inner_gaussian_window)
		inner_saturation = self.computeSaturationHist(img_hsv, inner_patch, inner_gaussian_window)

		inner_hue_density = np.zeros(self.HISTBINNUM)
		for i in range(0, self.HISTBINNUM):
			inner_hue_density[i] = inner_hue[i] + inner_hue[((i+1)%self.HISTBINNUM)]
		max_hue_bin = np.argmax(inner_hue_density)
		if(inner_hue_density[max_hue_bin] < self.HUEFRACTION * np.sum(inner_hue)):
			return False

		"""shrink hue bin, assign HUE_START_INDEX, HUE_END_INDEX, note that these two indexes need to mod self.HISTBINNUM before use"""
		if (inner_hue[max_hue_bin]/inner_hue_density[max_hue_bin] > self.SHRINK_HUE_BIN_FRACTION):
			# max_hue_bin itself is prominent
			self.HUE_START_INDEX = max_hue_bin
			self.HUE_END_INDEX = max_hue_bin + 1
		elif (inner_hue[((max_hue_bin + 1) % self.HISTBINNUM)]/inner_hue_density[max_hue_bin] > self.SHRINK_HUE_BIN_FRACTION):
			# max_hue_bin + 1 itself is prominent
			self.HUE_START_INDEX = (max_hue_bin + 1)
			self.HUE_END_INDEX = (max_hue_bin + 1 + 1)
		else:
			# need two bins
			self.HUE_START_INDEX = max_hue_bin
			self.HUE_END_INDEX = (max_hue_bin + 1 + 1)

		"""Acquire Saturation Bin"""
		target_hue_bins = []
		for i in range(self.HUE_START_INDEX, self.HUE_END_INDEX):
			target_hue_bins.append(i % self.HISTBINNUM)
		self.SATURATION_START_INDEX, self.SATURATION_END_INDEX = self.findSaturationRangeForTargetHueBin(img_hsv, inner_patch, target_hue_bins, inner_gaussian_window)
		plotStatistics.plotOneGivenHist("", "inner_saturation", inner_saturation, save = False, show = True)
		
		"""Check border hist, should not contain targeted hue"""
		target_saturation_bins = range(self.SATURATION_START_INDEX, self.SATURATION_END_INDEX)
		filtered_border_hue = self.borderTargetHueFilteredBySaturation(img_hsv, self.patch, inner_patch, gaussian_window, target_hue_bins, target_saturation_bins)
		border_hue = outer_hue - inner_hue

		plotStatistics.plotOneGivenHist("", "filtered_border_hue", filtered_border_hue, save = False, show = True)
		plotStatistics.plotOneGivenHist("", "border_hue", border_hue, save = False, show = True)

		if (np.sum(filtered_border_hue)/ np.sum(border_hue) > self.SATURATIONFRACTION_INVERSE):
			return False

		print "successfully constructed feature centre_paradigm, self.HUE_START_INDEX:", self.HUE_START_INDEX, \
		"self.HUE_END_INDEX:", self.HUE_END_INDEX, \
		"self.SATURATION_START_INDEX:", self.SATURATION_START_INDEX, \
		"self.SATURATION_END_INDEX:", self.SATURATION_END_INDEX

		return True

	def findSaturationRangeForTargetHueBin(self, img_hsv, inner_patch, target_hue_bins, inner_gaussian_window):
		SATURATION_ACQUIRE_FRACTION = 0.5
		hist = np.zeros(self.HISTBINNUM)
		ref_x = inner_patch.x - inner_patch.size/2
		ref_y = inner_patch.y - inner_patch.size/2
		for i in range(inner_patch.x - inner_patch.size/2, inner_patch.x + inner_patch.size/2 + 1):
			for j in range(inner_patch.y - inner_patch.size/2, inner_patch.y + inner_patch.size/2 + 1):
				this_hue_bin = int(img_hsv[i][j][0]/360.0 * self.HISTBINNUM)
				if (this_hue_bin == self.HISTBINNUM):
					this_hue_bin = self.HISTBINNUM - 1
				this_saturation_bin = int(img_hsv[i][j][1]/1.0 * self.HISTBINNUM)
				if (this_saturation_bin == self.HISTBINNUM):
					this_saturation_bin = self.HISTBINNUM - 1

				if (this_hue_bin in target_hue_bins):
					print "for ", this_hue_bin, " saturation bin is:", this_saturation_bin
					hist[this_saturation_bin] += 1 * inner_gaussian_window[i - ref_x][j - ref_y]

		non_zeros_saturation_bins = np.where(hist != 0)[0]
		max_saturation_bin = np.argmax(hist)
		acquired_saturation_bins = []
		for bin in non_zeros_saturation_bins:
			if (hist[bin] > SATURATION_ACQUIRE_FRACTION * hist[max_saturation_bin]):
				acquired_saturation_bins.append(bin)
		print "acquired saturation bins:", 
		return np.min(acquired_saturation_bins),  np.max(acquired_saturation_bins) + 1


	def withinPatch(self, patch, i, j):
		if (i < patch.x + patch.size/2 + 1 and \
			j < patch.y + patch.size/2 + 1 and \
			i >= patch.x - patch.size/2 and \
			j >= patch.y - patch.size/2):
			return True
		else:
			return False

	def borderTargetHueFilteredBySaturation(self, img_hsv, patch, inner_patch, gaussian_window, target_hue_bins, target_saturation_bins):
		hist = np.zeros(self.HISTBINNUM)
		ref_x = patch.x - patch.size/2
		ref_y = patch.y - patch.size/2
		for i in range(patch.x - patch.size/2, patch.x + patch.size/2 + 1):
			for j in range(patch.y - patch.size/2, patch.y + patch.size/2 + 1):
				if (not self.withinPatch(inner_patch, i, j)): # only check the border pixels
					this_hue_bin = int(img_hsv[i][j][0]/360.0 * self.HISTBINNUM)
					if (this_hue_bin == self.HISTBINNUM):
						this_hue_bin = self.HISTBINNUM - 1
					this_saturation_bin = int(img_hsv[i][j][1]/1.0 * self.HISTBINNUM)
					if (this_saturation_bin == self.HISTBINNUM):
						this_saturation_bin = self.HISTBINNUM - 1

					if (this_hue_bin in target_hue_bins and this_saturation_bin in target_saturation_bins):
						hist[this_hue_bin] += 1 * gaussian_window[i - ref_x][j - ref_y]
		return hist




	