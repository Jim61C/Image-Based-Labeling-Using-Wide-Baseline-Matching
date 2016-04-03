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

class FeatureSubSquareParadigm(Feature):
	"""
	reinforce the outer border to be non-targeted blue hue, i.e.,
	"""
	def __init__(self, patch, id):
		Feature.__init__(self, patch, id)
		self.HISTBINNUM = 16
		self.GAUSSIAN_SCALE = 3
		self.GAUSSIAN_WINDOW_LENGTH_SIGMA = 6.0 # window length = 6 sigma
		
		self.HUE_START_INDEX = None
		self.HUE_END_INDEX = None
		self.SATURATION_START_INDEX = None
		self.SATURATION_END_INDEX = None

		"""TODO: try indexes used to filter hue bins that are wrongly categorized within target bins"""
		self.SATURATION_FILTER_START_INDEX = None
		self.SATURATION_FILTER_END_INDEX = None

		### The patch of interest, one of TOP_LEFT, TOP_RIGHT, BOTTOM_LEFT, BOTTOM_RIGHT ###
		self.SUBPATCH_OF_INTEREST_INDEX = None
		
		self.FEATURE_MODEL_HUE = np.zeros(self.HISTBINNUM)
		
		self.FEATURE_MODEL_SATURATION = np.zeros(self.HISTBINNUM)

		self.FEATURE_MODEL = None

		# self.FEATURE_MODEL = np.concatenate(( \
		# 	self.FEATURE_MODEL_HUE, \
		# 	self.FEATURE_MODEL_SATURATION, \
		# 	np.zeros(len(range(self.HUE_START_INDEX,self.HUE_END_INDEX)) + \
		# 	len(range(self.SATURATION_START_INDEX,self.SATURATION_END_INDEX)))), axis = 1) # append the expected border response
		# self.FEATURE_MODEL = normalize(self.FEATURE_MODEL, norm='l1')[0] # normalize the histogram using l1

	def computeFeature(self, img, useGaussianSmoothing = True):
		img_hsv = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_BGR2HSV)
		sub_patch, sub_gaussian_window , full_patch_gaussian_window = \
		self.getSubPatchAndSubPatchGaussianFromSubPatchIndex(self.SUBPATCH_OF_INTEREST_INDEX)
		target_hue_bins, target_saturation_bins = self.getTargetHueAndSaturationBins()

		if (not len(self.patch.hs_2d_arr) == 5):
			self.computeHS2DArr(img_hsv, self.patch, full_patch_gaussian_window)

		if(not (len(self.patch.HueHistArr) == 5 and len(self.patch.SaturationHistArr) == 5)):
			self.patch.HueHistArr = []
			self.patch.SaturationHistArr = []
			self.patch.ValueHistArr = []
			"""derive from 2D instead of recompute"""
			for i in range(0, 5):
				self.patch.HueHistArr.append(self.derive1DHueFrom2D(self.patch.hs_2d_arr[i]))
				self.patch.SaturationHistArr.append(self.derive1DSaturationFrom2D(self.patch.hs_2d_arr[i]))

		"""broder hue range used for filtering hue than filtering saturation"""
		filtered_hue_of_interest = self.deriveHueHistFilterOffHueWithWrongSaturationFrom2D(\
			self.patch.hs_2d_arr[self.SUBPATCH_OF_INTEREST_INDEX], \
			range(self.SATURATION_FILTER_START_INDEX, self.SATURATION_FILTER_END_INDEX), target_hue_bins)

		filtered_saturation_of_interest = self.deriveSaturationHistFilterOffSaturationWithWrongHueFrom2D(\
			self.patch.hs_2d_arr[self.SUBPATCH_OF_INTEREST_INDEX], target_saturation_bins, target_hue_bins)
		
		self.hist = np.concatenate((filtered_hue_of_interest * np.sum(full_patch_gaussian_window) / np.sum(sub_gaussian_window), \
			filtered_saturation_of_interest * np.sum(full_patch_gaussian_window) / np.sum(sub_gaussian_window)), axis = 1)

		for i in range(self.TOP_LEFT_INDEX, self.BOTTOM_RIGHT_INDEX + 1):
			if(i != self.SUBPATCH_OF_INTEREST_INDEX):
				other_patch_hue = np.array([ self.patch.HueHistArr[i][j % self.HISTBINNUM] \
					for j in range(self.HUE_START_INDEX, self.HUE_END_INDEX)])
				other_patch_saturation = self.patch.SaturationHistArr[i][self.SATURATION_FILTER_START_INDEX:self.SATURATION_FILTER_END_INDEX]
				"""
				other patch hue / saturation, 
				if one of them is not within the targeted range for the sub patch of interest, 
				then mark as good (all zeros, same as FEATURE_MODEL)
				"""
				if(np.sum(other_patch_hue) == 0 or np.sum(other_patch_saturation) == 0):
					other_patch_hist = np.zeros(len(other_patch_hue) + len(other_patch_saturation))
				else:
					other_patch_hist = np.concatenate((other_patch_hue, other_patch_saturation), axis = 1)

				self.hist = np.concatenate((self.hist, other_patch_hist), axis = 1)

	def featureResponse(self, metric_func = DIST.euclidean):
		assert (not self.hist is None), "Error in FeatureSubSquareParadigm: calling computeScore before the feature hist is computed!"
		assert (len(self.hist) == len(self.FEATURE_MODEL)), "Error in FeatureSubSquareParadigm: hist length is not correct!" + \
		"len(self.hist): {self_his_len}, len(self.FEATURE_MODEL): {feature_model_len}".format(\
			self_his_len = len(self.hist), feature_model_len = len(self.FEATURE_MODEL))
		return 1.0 / (1.0 + metric_func(self.hist, self.FEATURE_MODEL))

	def computeScore(self):
		if(self.score is None):
			self.score = self.featureResponse(comparePatches.Jensen_Shannon_Divergence_Hat)
			# self.score = self.featureResponse()

	def fitParadigm(self, img):
		"""
		For each of the 4 square sub patch:
		1. checks the hue, saturation hist, finds the mode color and see whether that color constitutes the majority of that sub patch
		2. checks the other 3 sub patches' hue, saturation hist, make sure that they do not have the found hue color of the SUBPATCH_OF_INTEREST_INDEX
		returns true or false
		"""
		self.HUEFRACTION = 0.7
		self.SATURATIONFRACTION_INVERSE = 0.1 # maximum noise in other hists
		self.SHRINK_HUE_BIN_FRACTION = 0.9
		self.HISTBINNUM = 16
		MINIMUM_VALUE_CHANNEL_BIN = 4
		MAX_FIRST_BIN_SATURATION_PERCENT = 0.5

		img_hsv = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_BGR2HSV)
		gaussian_window = comparePatches.gauss_kernels(self.patch.size, sigma = self.patch.size/self.GAUSSIAN_WINDOW_LENGTH_SIGMA)
		
		sub_gaussian_windows = []
		sub_patches = []

		newLen = (self.patch.size+1)/2
		# newSize is the size of the sub patch
		if(newLen % 2 == 0):
			newSize = newLen -1
		else:
			newSize = newLen
		
		"""TOP_LEFT subpatch"""
		sub_gaussian_windows.append(gaussian_window[0:newSize,0:newSize])
		sub_patches.append(comparePatches.Patch(self.patch.x - newLen/2, self.patch.y - newLen/2, newSize, initialize_features = False))
		"""TOP_RIGHT subpatch"""
		sub_gaussian_windows.append(gaussian_window[0:newSize, gaussian_window.shape[1] - newSize:gaussian_window.shape[1]])
		sub_patches.append(comparePatches.Patch(self.patch.x - newLen/2, self.patch.y + newLen/2, newSize, initialize_features = False))
		"""BOTTOM_LEFT subpatch"""
		sub_gaussian_windows.append(gaussian_window[gaussian_window.shape[0] - newSize:gaussian_window.shape[0], 0:newSize])
		sub_patches.append(comparePatches.Patch(self.patch.x + newLen/2, self.patch.y - newLen/2, newSize, initialize_features = False))
		"""BOTTOM_RIGHT subpatch"""
		sub_gaussian_windows.append(gaussian_window[gaussian_window.shape[0] - newSize:gaussian_window.shape[0], gaussian_window.shape[1] - newSize: gaussian_window.shape[1]])
		sub_patches.append(comparePatches.Patch(self.patch.x + newLen/2, self.patch.y + newLen/2, newSize, initialize_features = False))

		found_feature_patch_flag = False
		for i in range(self.TOP_LEFT_INDEX, self.BOTTOM_RIGHT_INDEX + 1):
			idx = i - 1 # true idx for the lists
			# print "checking subpatch index:", idx
			this_patch = sub_patches[idx]
			this_sub_gaussian_window = sub_gaussian_windows[idx]

			this_hue = self.computeHueHist(img_hsv, this_patch , this_sub_gaussian_window)
			this_saturation = self.computeSaturationHist(img_hsv, this_patch, this_sub_gaussian_window)

			this_hue_density = np.zeros(self.HISTBINNUM)
			for j in range(0, self.HISTBINNUM):
				this_hue_density[j] = this_hue[j] + this_hue[((j+1)%self.HISTBINNUM)]
			max_hue_bin = np.argmax(this_hue_density)

			# print "np.sum(this_hue):", np.sum(this_hue)
			# print "max response:", this_hue_density[max_hue_bin]

			# plotStatistics.plotOneGivenHist("", "potential_target_hue", this_hue, save = False, show = True)
			# plotStatistics.plotOneGivenHist("", "potential_target_saturation", this_saturation, save = False, show = True)

			if(this_hue_density[max_hue_bin] >= self.HUEFRACTION * np.sum(this_hue)):

				"""shrink hue bin, assign HUE_START_INDEX, HUE_END_INDEX, note that these two indexes need to mod self.HISTBINNUM before use"""
				if (this_hue[max_hue_bin]/this_hue_density[max_hue_bin] > self.SHRINK_HUE_BIN_FRACTION):
					# max_hue_bin itself is prominent
					self.HUE_START_INDEX = max_hue_bin
					self.HUE_END_INDEX = max_hue_bin + 1
				elif (this_hue[((max_hue_bin + 1) % self.HISTBINNUM)]/this_hue_density[max_hue_bin] > self.SHRINK_HUE_BIN_FRACTION):
					# max_hue_bin + 1 itself is prominent
					self.HUE_START_INDEX = (max_hue_bin + 1)
					self.HUE_END_INDEX = (max_hue_bin + 1 + 1)
				else:
					# need two bins
					self.HUE_START_INDEX = max_hue_bin
					self.HUE_END_INDEX = (max_hue_bin + 1 + 1)

				"""Acquire Saturation Bin"""
				target_hue_bins = []
				for j in range(self.HUE_START_INDEX, self.HUE_END_INDEX):
					target_hue_bins.append(j % self.HISTBINNUM)


				value_for_target_hue_bins = self.findValueHistForTargetHueBin(\
				img_hsv, this_patch, this_sub_gaussian_window, target_hue_bins)

				saturation_for_target_hue_bins = self.findSaturationHistForTargetHueBin(\
				img_hsv, this_patch, this_sub_gaussian_window, target_hue_bins)

				"""Value Channel of the interest sub patch must have minimum thresh"""
				if (np.argmax(value_for_target_hue_bins) > MINIMUM_VALUE_CHANNEL_BIN and \
					saturation_for_target_hue_bins[0]/ np.sum(saturation_for_target_hue_bins) < MAX_FIRST_BIN_SATURATION_PERCENT):

					"""SATURATION_START_INDEX, SATURATION_END_INDEX does not need to be Mod before use"""
					self.SATURATION_START_INDEX, self.SATURATION_END_INDEX, \
					self.SATURATION_FILTER_START_INDEX, self.SATURATION_FILTER_END_INDEX = \
					self.findSaturationRangeForTargetHueBin( \
						img_hsv, this_patch, target_hue_bins, this_sub_gaussian_window)
					
					"""Check other three hists, should not contain targeted hue"""
					target_saturation_bins = range(self.SATURATION_START_INDEX, self.SATURATION_END_INDEX)
					# filtered_other_hue have the same size of target_hue_bins, which only contain bins of interest
					filtered_other_hue = np.zeros(len(target_hue_bins))
					other_hue = np.zeros(self.HISTBINNUM)
					for j in range(self.TOP_LEFT_INDEX, self.BOTTOM_RIGHT_INDEX + 1):
						if (j != i):
							one_other_hue = self.computeHueHist(img_hsv, sub_patches[j - 1] , sub_gaussian_windows[j - 1])
							other_hue = other_hue + one_other_hue
							
							one_filtered_other_hue = self.targetHueFilteredBySaturation( \
								img_hsv, sub_patches[j - 1], sub_gaussian_windows[j - 1], target_hue_bins, target_saturation_bins)
							filtered_other_hue = filtered_other_hue + one_filtered_other_hue

					# print "filtered_other_hue:", filtered_other_hue
					# print "np.sum(filtered_other_hue):", np.sum(filtered_other_hue)
					# print "np.sum(other_hue):", np.sum(other_hue)

					# plotStatistics.plotOneGivenHist("", "filtered_other_hue", filtered_other_hue, save = False, show = True)
					# plotStatistics.plotOneGivenHist("", "other_hue", other_hue, save = False, show = True)

					if (np.sum(filtered_other_hue)/ np.sum(other_hue) <= self.SATURATIONFRACTION_INVERSE):
						"""Found feature patch, break and return True"""
						found_feature_patch_flag = True
						self.SUBPATCH_OF_INTEREST_INDEX = i
						break

		if (found_feature_patch_flag):
			print "successfully constructed FeatureSubSquareParadigm, self.HUE_START_INDEX:", self.HUE_START_INDEX, \
			"self.HUE_END_INDEX:", self.HUE_END_INDEX, \
			"self.SATURATION_START_INDEX:", self.SATURATION_START_INDEX, \
			"self.SATURATION_END_INDEX:", self.SATURATION_END_INDEX, \
			"self.SATURATION_FILTER_START_INDEX:", self.SATURATION_FILTER_START_INDEX, \
			"self.SATURATION_FILTER_END_INDEX:", self.SATURATION_FILTER_END_INDEX, \
			"self.SUBPATCH_OF_INTEREST_INDEX:", self.SUBPATCH_OF_INTEREST_INDEX

			self.computeFeatureModel(this_hue, this_saturation)

		return found_feature_patch_flag

	def computeFeatureModel(self, hue, saturation):
		assert (not self.HUE_START_INDEX is None), "in computeFeatureModel: HUE_START_INDEX must not be None"
		assert (not self.HUE_END_INDEX is None), "in computeFeatureModel: HUE_END_INDEX must not be None"
		assert (not self.SATURATION_START_INDEX is None), "in computeFeatureModel: SATURATION_START_INDEX must not be None"
		assert (not self.SATURATION_END_INDEX is None), "in computeFeatureModel: SATURATION_END_INDEX must not be None"
		assert (not self.SATURATION_FILTER_START_INDEX is None), "in computeFeatureModel: SATURATION_FILTER_START_INDEX must not be None"
		assert (not self.SATURATION_FILTER_END_INDEX is None), "in computeFeatureModel: SATURATION_FILTER_END_INDEX must not be None"
		assert (not self.SUBPATCH_OF_INTEREST_INDEX is None), "in computeFeatureModel: SUBPATCH_OF_INTEREST_INDEX must not be None"
		assert (len(hue) == self.HISTBINNUM), "in computeFeatureModel:, hue length must be the same as HISTBINNUM"
		assert (len(saturation) == self.HISTBINNUM), "in computeFeatureModel:, saturation length must be the same as HISTBINNUM"
		
		for i in range(self.HUE_START_INDEX, self.HUE_END_INDEX):
			# self.FEATURE_MODEL_HUE[i % self.HISTBINNUM] = 1.0/(self.HUE_END_INDEX - self.HUE_START_INDEX) # hue indexes need mod before use
			self.FEATURE_MODEL_HUE[i % self.HISTBINNUM] = hue[i % self.HISTBINNUM] # hue indexes need mod before use
		for i in range(self.SATURATION_FILTER_START_INDEX, self.SATURATION_FILTER_END_INDEX):
			# self.FEATURE_MODEL_SATURATION[i] = 1.0/(self.SATURATION_FILTER_END_INDEX - self.SATURATION_FILTER_START_INDEX)
			self.FEATURE_MODEL_SATURATION[i] = saturation[i]
		
		self.FEATURE_MODEL = np.concatenate(( \
			normalize(self.FEATURE_MODEL_HUE, norm = 'l1')[0], \
			normalize(self.FEATURE_MODEL_SATURATION, norm = 'l1')[0], \
			np.zeros( 3 *(len(range(self.HUE_START_INDEX,self.HUE_END_INDEX)) + \
			len(range(self.SATURATION_FILTER_START_INDEX,self.SATURATION_FILTER_END_INDEX))) )), axis = 1) # append the expected border response

		plotStatistics.plotOneGivenHist("", "Sub Square Paradigm FEATURE_MODEL", self.FEATURE_MODEL, save = False, show = True)







