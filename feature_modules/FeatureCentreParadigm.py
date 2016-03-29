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
from feature_modules import Feature
from feature_modules import utils

class FeatureCentreParadigm(Feature):
	"""
	reinforce the outer border to be non-targeted blue hue, i.e.,
	"""
	def __init__(self, patch, id):
		Feature.__init__(self, patch, id)
		self.HISTBINNUM = 36 # adjustable number of bins
		# for 2D histograms
		self.HISTBINNUM_HUE = 36
		self.HISTBINNUM_SATURATION = 36

		self.GAUSSIAN_SCALE = 3
		self.GAUSSIAN_WINDOW_LENGTH_SIGMA = 4.0 # used for border feature case, window length = 4 sigma
		
		self.HUE_START_INDEX = None
		self.HUE_END_INDEX = None
		self.SATURATION_START_INDEX = None
		self.SATURATION_END_INDEX = None

		"""
		TODO 0: try aggragate Hue bins
			 1: try seperate indexes used to filter hue bins that are wrongly categorized within target bins (now saturation range is very wide, may introduce noise)
		     2: try change HISTBINNUM to 16
		     3: If border saturation is concentrated, then add that as well.
		     4: !try use Saturation * Hue as FEATURE_MODEL
		     5: try 2D histogram -> leads to the problem of dissimilarity metric
		     6: !try switch back to 36 bins, but for hue, widen up for border error detection
		     7: !! if current one still wrongly match, try adjust border_hist weight and increase border_hist saturation range!!!!
		"""
		self.SATURATION_FILTER_START_INDEX = None
		self.SATURATION_FILTER_END_INDEX = None

		self.SATURATION_BORDER_FILTER_START_INDEX = None
		self.SATURATION_BORDER_FILTER_END_INDEX = None
		
		self.FEATURE_MODEL_HUE = np.zeros(self.HISTBINNUM)
		
		self.FEATURE_MODEL_SATURATION = np.zeros(self.HISTBINNUM)

		self.FEATURE_MODEL = None
		# for 2D histograms
		self.FEATURE_MODEL_BORDER = None

		self.border_hist_hue = None
		self.border_hist_saturation = None
		
		self.inner_hist_hue = None
		self.inner_hist_saturation = None

		# cutted and mannually crafted hist
		self.border_hist = None

		# self.FEATURE_MODEL = np.concatenate(( \
		# 	self.FEATURE_MODEL_HUE, \
		# 	self.FEATURE_MODEL_SATURATION, \
		# 	np.zeros(len(range(self.HUE_START_INDEX,self.HUE_END_INDEX)) + \
		# 	len(range(self.SATURATION_START_INDEX,self.SATURATION_END_INDEX)))), axis = 1) # append the expected border response
		# self.FEATURE_MODEL = normalize(self.FEATURE_MODEL, norm='l1')[0] # normalize the histogram using l1

	def computeFeature(self, img, useGaussianSmoothing = True):
		img_hsv = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_BGR2HSV)
		# print "hue channel:", img_hsv[:,:,0]
		# print "saturation channel:", img_hsv[:,:,1]
		inner_patch_size = comparePatches.getGaussianScale(self.patch.size, self.GAUSSIAN_SCALE_FACTOR, -self.GAUSSIAN_SCALE)
		inner_patch = comparePatches.Patch(self.patch.x, self.patch.y, inner_patch_size)

		gaussian_window = comparePatches.gauss_kernels(self.patch.size, sigma = self.patch.size/self.GAUSSIAN_WINDOW_LENGTH_SIGMA)
		inner_gaussian_window = gaussian_window[ \
		gaussian_window.shape[0]/2 - inner_patch.size/2: gaussian_window.shape[0]/2 + inner_patch.size/2 + 1 ,\
		gaussian_window.shape[1]/2 - inner_patch.size/2: gaussian_window.shape[1]/2 + inner_patch.size/2 + 1]

		assert gaussian_window.shape == (self.patch.size, self.patch.size), "outer gaussian_window size not correct"
		assert inner_gaussian_window.shape == (inner_patch.size, inner_patch.size), "inner gaussian_window size not correct"

		"""Still need to compute the overall Hue, Saturation, since sigma is different now 4.0 instead of 6.0"""
		if (self.patch.inner_hist_scale_3_gaus_4_centre_paradigm_2d is None):
			self.patch.inner_hist_scale_3_gaus_4_centre_paradigm_2d = cv2.calcHist(\
			[img_hsv[inner_patch.x - inner_patch.size/2 :inner_patch.x + inner_patch.size/2 + 1, \
			inner_patch.y - inner_patch.size/2: inner_patch.y + inner_patch.size/2 + 1]], \
			[0, 1], None, [self.HISTBINNUM_HUE, self.HISTBINNUM_SATURATION], [0, 360.0, 0, 1.0]) # for 2D HS case
		
		if (self.patch.outer_hist_scale_3_gaus_4_centre_paradigm_2d is None):
			self.patch.outer_hist_scale_3_gaus_4_centre_paradigm_2d = cv2.calcHist(\
			[img_hsv[self.patch.x - self.patch.size/2 :self.patch.x + self.patch.size/2 + 1, \
			self.patch.y - self.patch.size/2: self.patch.y + self.patch.size/2 + 1]], \
			[0, 1], None, [self.HISTBINNUM_HUE, self.HISTBINNUM_SATURATION], [0, 360.0, 0, 1.0]) # for 2D HS case
		
		inner_hue_saturation = self.patch.inner_hist_scale_3_gaus_4_centre_paradigm_2d
		outer_hue_saturation = self.patch.outer_hist_scale_3_gaus_4_centre_paradigm_2d
		border_hue_saturation = outer_hue_saturation - inner_hue_saturation
		border_hue_saturation = cv2.normalize(border_hue_saturation, norm_type = cv2.NORM_L1)

		target_hue_bins = []
		aggregated_sum_row = np.zeros(self.HISTBINNUM_SATURATION, dtype = inner_hue_saturation.dtype)
		for i in range(self.HUE_START_INDEX, self.HUE_END_INDEX):
			target_hue_bins.append(i % self.HISTBINNUM_HUE)
			# record the aggreagetd sum row
			aggregated_sum_row = aggregated_sum_row + inner_hue_saturation[i%self.HISTBINNUM_HUE,]
		
		"""compute border_hist"""
		self.border_hist = None
		for bin in target_hue_bins:
			if (self.border_hist is None):
				self.border_hist = border_hue_saturation[bin:bin+1,\
				self.SATURATION_FILTER_START_INDEX:self.SATURATION_FILTER_END_INDEX] # make sure is 2d shape
			else:
				self.border_hist = np.vstack((self.border_hist, \
					border_hue_saturation[bin,self.SATURATION_FILTER_START_INDEX:self.SATURATION_FILTER_END_INDEX]))

		"""aggregate Hue bins"""
		if (self.HUE_END_INDEX > self.HISTBINNUM_HUE):
			self.hist = np.concatenate((\
				inner_hue_saturation[self.HUE_END_INDEX % self.HISTBINNUM:self.HUE_START_INDEX, ], \
				aggregated_sum_row.reshape(1, len(aggregated_sum_row)),\
				inner_hue_saturation[self.HUE_END_INDEX:,]), axis = 0)
		else:
			self.hist = np.concatenate((\
				inner_hue_saturation[:self.HUE_START_INDEX,], \
				aggregated_sum_row.reshape(1, len(aggregated_sum_row)),\
				inner_hue_saturation[self.HUE_END_INDEX:,]), axis = 0)
		self.hist = cv2.normalize(self.hist, norm_type = cv2.NORM_L1)
		
		# comparePatches.drawPatchesOnImg(np.copy(img),[self.patch, inner_patch], True)
		

	def featureResponse(self, metric_func):
		assert (not self.hist is None), "Error in FeatureCentreParadigm: calling computeScore before the feature hist is computed!"
		assert (self.hist.shape == self.FEATURE_MODEL.shape), "Error in FeatureCentreParadigm: hist shape is not correct!" + \
		"self.hist.shape: {self_his_shape}, self.FEATURE_MODEL.shape: {feature_model_shape}".format(\
			self_his_shape = self.hist.shape, feature_model_shape = self.FEATURE_MODEL.shape)
		
		# if distanceBHATTACHARYYA, each of h_s_distance, border_distance <= 1
		"""Try: change to return 1 - sqrt(0.5 * h_s_distance**2, 0.5 * border_distance**2)"""
		h_s_distance = metric_func(self.hist, self.FEATURE_MODEL)
		border_distance = metric_func(self.border_hist, self.FEATURE_MODEL_BORDER)
		
		return 1.0/(1.0 + np.linalg.norm([h_s_distance, border_distance], 2))

	def computeScore(self):
		if(self.score is None):
			self.score = self.featureResponse(comparePatches.distanceBHATTACHARYYA)


	def computeFeatureModel(self, img_hsv, inner_patch, inner_gaussian_window):
		"""img_hsv needs to be in Hue: [0,360], Saturation: [0,1], Value: [0,1]"""
		assert (not self.HUE_START_INDEX is None), "in computeFeatureModel: HUE_START_INDEX must not be None"
		assert (not self.HUE_END_INDEX is None), "in computeFeatureModel: HUE_END_INDEX must not be None"
		assert (not self.SATURATION_START_INDEX is None), "in computeFeatureModel: SATURATION_START_INDEX must not be None"
		assert (not self.SATURATION_END_INDEX is None), "in computeFeatureModel: SATURATION_END_INDEX must not be None"
		assert (not self.SATURATION_FILTER_START_INDEX is None), "in computeFeatureModel: SATURATION_FILTER_START_INDEX must not be None"
		assert (not self.SATURATION_FILTER_END_INDEX is None), "in computeFeatureModel: SATURATION_FILTER_END_INDEX must not be None"
		
		inner_hue_saturation = cv2.calcHist(\
			[img_hsv[inner_patch.x - inner_patch.size/2 :inner_patch.x + inner_patch.size/2 + 1, \
			inner_patch.y - inner_patch.size/2: inner_patch.y + inner_patch.size/2 + 1]], \
			[0, 1], None, [self.HISTBINNUM_HUE, self.HISTBINNUM_SATURATION], [0, 360.0, 0, 1.0]) # for 2D HS case
		# inner_hue_saturation = cv2.normalize(inner_hue_saturation, norm_type = cv2.NORM_L1)
		self.FEATURE_MODEL = np.zeros(shape = inner_hue_saturation.shape, dtype = inner_hue_saturation.dtype)

		"""update indexes for building feature model other than detect"""
		self.HUE_START_INDEX = int(self.HUE_START_INDEX * float(self.HISTBINNUM_HUE)/ self.HISTBINNUM)
		self.HUE_END_INDEX = int(self.HUE_END_INDEX * float(self.HISTBINNUM_HUE)/ self.HISTBINNUM)
		self.SATURATION_FILTER_START_INDEX = int(self.SATURATION_FILTER_START_INDEX * float(self.HISTBINNUM_SATURATION)/ self.HISTBINNUM)
		self.SATURATION_FILTER_END_INDEX = int(self.SATURATION_FILTER_END_INDEX * float(self.HISTBINNUM_SATURATION)/ self.HISTBINNUM)
		self.SATURATION_START_INDEX = int(self.SATURATION_START_INDEX * float(self.HISTBINNUM_SATURATION)/self.HISTBINNUM)
		self.SATURATION_END_INDEX = int(self.SATURATION_END_INDEX * float(self.HISTBINNUM_SATURATION)/self.HISTBINNUM)

		target_hue_bins = []
		for i in range(self.HUE_START_INDEX, self.HUE_END_INDEX):
			target_hue_bins.append(i % self.HISTBINNUM_HUE)

		"""model to only keep the values in Hue range"""
		for bin in target_hue_bins:
			self.FEATURE_MODEL[bin, self.SATURATION_FILTER_START_INDEX: self.SATURATION_FILTER_END_INDEX] = \
			inner_hue_saturation[bin, self.SATURATION_FILTER_START_INDEX: self.SATURATION_FILTER_END_INDEX]

		aggregated_sum_row = np.zeros(self.HISTBINNUM_SATURATION, dtype = self.FEATURE_MODEL.dtype)
		for i in range(self.HUE_START_INDEX, self.HUE_END_INDEX):
			aggregated_sum_row = aggregated_sum_row + self.FEATURE_MODEL[i%self.HISTBINNUM_HUE,]

		"""aggregate Hue bins"""
		if (self.HUE_END_INDEX > self.HISTBINNUM_HUE):
			self.FEATURE_MODEL = np.concatenate((\
				self.FEATURE_MODEL[self.HUE_END_INDEX % self.HISTBINNUM:self.HUE_START_INDEX, ], \
				aggregated_sum_row.reshape(1, len(aggregated_sum_row)),\
				self.FEATURE_MODEL[self.HUE_END_INDEX:,]), axis = 0)
		else:
			self.FEATURE_MODEL = np.concatenate((\
				self.FEATURE_MODEL[:self.HUE_START_INDEX,], \
				aggregated_sum_row.reshape(1, len(aggregated_sum_row)),\
				self.FEATURE_MODEL[self.HUE_END_INDEX:,]), axis = 0)
		self.FEATURE_MODEL = cv2.normalize(self.FEATURE_MODEL, norm_type = cv2.NORM_L1)

		self.FEATURE_MODEL_BORDER = np.zeros(\
			shape = (len(target_hue_bins),len(range(self.SATURATION_FILTER_START_INDEX, self.SATURATION_FILTER_END_INDEX))),\
			dtype = self.FEATURE_MODEL.dtype)

		"""--- logging results ---"""
		# print "np.sum(self.FEATURE_MODEL):", np.sum(self.FEATURE_MODEL)
		# print "last row:", self.FEATURE_MODEL[self.FEATURE_MODEL.shape[0] - 1]
		plotStatistics.plot2Dhistogram("", "FEATURE_MODEL", self.FEATURE_MODEL, save = False, show = True)
		plotStatistics.plot2Dhistogram("", "FEATURE_MODEL_BORDER", self.FEATURE_MODEL_BORDER, save = False, show = True)
		print "self.FEATURE_MODEL.shape:", self.FEATURE_MODEL.shape
		print "self.FEATURE_MODEL_BORDER.shape:", self.FEATURE_MODEL_BORDER.shape
		# plotStatistics.plot2Dhistogram("", "FEATURE_MODEL_LAST_ROW", \
			# self.FEATURE_MODEL[33:35,], save = False, show = True)


	def fitParadigm(self, img):
		"""
		1. checks the inner hue, saturation hist, finds the mode color and see whether that color constitutes the majority of the inner patch
		2. checks the border hue, saturation hist, make sure that the border does not have the found inner hue color
		3. check for different inner GAUSSIAN_SCALE, default is 3
		returns true or false
		"""
		self.HUEFRACTION = 0.7
		self.SATURATIONFRACTION_INVERSE = 0.1 # maximum border hist
		self.SHRINK_HUE_BIN_FRACTION = 0.99

		img_hsv = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_BGR2HSV)
		gaussian_window = comparePatches.gauss_kernels(self.patch.size, sigma = self.patch.size/self.GAUSSIAN_WINDOW_LENGTH_SIGMA)

		inner_patch_size = comparePatches.getGaussianScale(self.patch.size, self.GAUSSIAN_SCALE_FACTOR, -self.GAUSSIAN_SCALE)
		inner_patch = comparePatches.Patch(self.patch.x, self.patch.y, inner_patch_size)
		inner_gaussian_window = gaussian_window[ \
		gaussian_window.shape[0]/2 - inner_patch.size/2: gaussian_window.shape[0]/2 + inner_patch.size/2 + 1 ,\
		gaussian_window.shape[1]/2 - inner_patch.size/2: gaussian_window.shape[1]/2 + inner_patch.size/2 + 1]

		outer_hue = self.computeHueHistSaturationWeighted(img_hsv, self.patch, gaussian_window)
		outer_saturation = self.computeSaturationHist(img_hsv, self.patch, gaussian_window)

		inner_hue = self.computeHueHistSaturationWeighted(img_hsv, inner_patch, inner_gaussian_window)
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
		if(self.HUE_START_INDEX >= self.HISTBINNUM):
			self.HUE_START_INDEX = self.HUE_START_INDEX % self.HISTBINNUM
			self.HUE_END_INDEX = self.HUE_END_INDEX % self.HISTBINNUM

		"""Acquire Saturation Bin"""
		target_hue_bins = []
		for i in range(self.HUE_START_INDEX, self.HUE_END_INDEX):
			target_hue_bins.append(i % self.HISTBINNUM)
		"""
		SATURATION_START_INDEX, SATURATION_END_INDEX does not need to be Mod before use
		"""
		self.SATURATION_START_INDEX, self.SATURATION_END_INDEX, self.SATURATION_FILTER_START_INDEX, self.SATURATION_FILTER_END_INDEX = \
		self.findSaturationRangeForTargetHueBin(img_hsv, inner_patch, target_hue_bins, inner_gaussian_window)
		plotStatistics.plotOneGivenHist("", "inner_saturation", inner_saturation, save = False, show = True)
		plotStatistics.plotOneGivenHist("", "inner_hue", inner_hue, save = False, show = True)
		
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
		"self.SATURATION_END_INDEX:", self.SATURATION_END_INDEX, \
		"self.SATURATION_FILTER_START_INDEX:", self.SATURATION_FILTER_START_INDEX, \
		"self.SATURATION_FILTER_END_INDEX:", self.SATURATION_FILTER_END_INDEX

		self.computeFeatureModel(img_hsv, inner_patch, inner_gaussian_window)

		return True

	def dissimilarityWith(self, feature_obj):
		"""
		override to use distanceBHATTACHARYYA for 2D histogram of feature.hist and feature.border_hist
		"""
		
		h_s_distance = comparePatches.distanceBHATTACHARYYA(self.hist, feature_obj.hist)
		border_distance = comparePatches.distanceBHATTACHARYYA(self.border_hist, feature_obj.border_hist)
		
		return np.linalg.norm([h_s_distance, border_distance], 2)

	# def dissimilarityWith(self, feature_obj):
	# 	"""
	# 	return the customized dissimilarity measure with another feature_obj of the same type
	# 	"""
	# 	hist = feature_obj.hist
	# 	self.assertHist(hist)

	# 	assert (len(self.border_hist_hue) == len(feature_obj.border_hist_hue)), "In FeatureCentreParadigm: border_hist_hue should have the same length"
	# 	assert (len(self.border_hist_saturation) == len(feature_obj.border_hist_saturation)), "In FeatureCentreParadigm: border_hist_saturation should have the same length"
	# 	assert (len(self.inner_hist_hue) == len(feature_obj.inner_hist_hue)), "In FeatureCentreParadigm: inner_hist_hue should have the same length"
	# 	assert (len(self.inner_hist_saturation) == len(feature_obj.inner_hist_saturation)), "In FeatureCentreParadigm: inner_hist_saturation should have the same length"

	# 	border_hist_hue_distance = comparePatches.Jensen_Shannon_Divergence(self.border_hist_hue, \
	# 		feature_obj.border_hist_hue)
	# 	border_hist_saturation_distance = comparePatches.Jensen_Shannon_Divergence(self.border_hist_saturation, \
	# 		feature_obj.border_hist_saturation)
	# 	inner_hist_hue_distance = comparePatches.Jensen_Shannon_Divergence(self.inner_hist_hue, \
	# 		feature_obj.inner_hist_hue)
	# 	inner_hist_saturation_distance = comparePatches.Jensen_Shannon_Divergence(self.inner_hist_saturation, \
	# 		feature_obj.inner_hist_saturation)

	# 	distance_vector = np.array([border_hist_hue_distance, \
	# 		inner_hist_hue_distance])

	# 	weights = np.ones(len(distance_vector))
	# 	return np.linalg.norm(np.multiply(distance_vector, weights), 2)

	