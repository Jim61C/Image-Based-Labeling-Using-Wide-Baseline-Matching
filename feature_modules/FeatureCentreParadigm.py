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
		self.HISTBINNUM = 16 # adjustable number of bins
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
		     4: try use Saturation * Hue as FEATURE_MODEL
		     5: try 2D histogram -> leads to the problem of dissimilarity metric
		"""
		self.SATURATION_FILTER_START_INDEX = None
		self.SATURATION_FILTER_END_INDEX = None
		
		self.FEATURE_MODEL_HUE = np.zeros(self.HISTBINNUM)
		
		self.FEATURE_MODEL_SATURATION = np.zeros(self.HISTBINNUM)

		self.FEATURE_MODEL = None

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
		if (self.patch.inner_hue_hist_scale_3_gaus_4_centre_paradigm is None):
			self.patch.inner_hue_hist_scale_3_gaus_4_centre_paradigm = self.computeHueHist(img_hsv, inner_patch, inner_gaussian_window)
		if (self.patch.inner_saturation_hist_scale_3_gaus_4_centre_paradigm is None):
			self.patch.inner_saturation_hist_scale_3_gaus_4_centre_paradigm = self.computeSaturationHist(img_hsv, inner_patch, inner_gaussian_window)
		if (self.patch.outer_hue_hist_scale_3_gaus_4_centre_paradigm is None):
			self.patch.outer_hue_hist_scale_3_gaus_4_centre_paradigm = self.computeHueHist(img_hsv, self.patch, gaussian_window)
		if (self.patch.outer_saturation_hist_scale_3_gaus_4_centre_paradigm is None):
			self.patch.outer_saturation_hist_scale_3_gaus_4_centre_paradigm = self.computeSaturationHist(img_hsv, self.patch, gaussian_window)
		
		inner_hist_hue = self.patch.inner_hue_hist_scale_3_gaus_4_centre_paradigm
		inner_hist_saturation = self.patch.inner_saturation_hist_scale_3_gaus_4_centre_paradigm

		outer_hist_hue = self.patch.outer_hue_hist_scale_3_gaus_4_centre_paradigm
		outer_hist_saturation = self.patch.outer_saturation_hist_scale_3_gaus_4_centre_paradigm

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

		"""compute border_hist"""
		border_hist_hue_targetted = np.array([border_hist_hue[i % self.HISTBINNUM] \
			for i in range(self.HUE_START_INDEX, self.HUE_END_INDEX)])
		if(np.sum(border_hist_hue_targetted) == 0 or \
			np.sum(border_hist_saturation[self.SATURATION_START_INDEX:self.SATURATION_END_INDEX]) == 0):
			border_hist = np.zeros(len(range(self.HUE_START_INDEX,self.HUE_END_INDEX)) + \
				len(range(self.SATURATION_START_INDEX,self.SATURATION_END_INDEX)))
		else:
			border_hist = np.concatenate((border_hist_hue_targetted, \
				border_hist_saturation[self.SATURATION_START_INDEX:self.SATURATION_END_INDEX]), axis = 1)

		target_hue_bins = []
		for i in range(self.HUE_START_INDEX, self.HUE_END_INDEX):
			target_hue_bins.append(i % self.HISTBINNUM)
		target_saturation_bins = range(self.SATURATION_START_INDEX, self.SATURATION_END_INDEX)
			
		"""Alternative: make border hist to be target border_hue with target saturation filtered"""
		# border_hist_full = self.borderTargetHueFilteredBySaturation(img_hsv, self.patch, inner_patch, \
		# 	gaussian_window, target_hue_bins, target_saturation_bins)
		# border_hist = np.array([border_hist_full[hue_bin] for hue_bin in target_hue_bins])
		# if(np.sum(border_hist) == 0):
		# 	border_hist = np.concatenate((border_hist, \
		# 		np.zeros(len(range(self.SATURATION_START_INDEX, self.SATURATION_END_INDEX)))), axis = 1)
		# else:
		# 	border_hist = np.concatenate((border_hist, \
		# 		border_hist_saturation[self.SATURATION_START_INDEX:self.SATURATION_END_INDEX]), axis = 1)

		"""
		filter the inner hist hue based the target bins for saturation as well
		"""
		filtered_inner_hist_hue = self.computeHueHistFilterOffHueWithWrongSaturation(\
			img_hsv, \
			inner_patch, \
			inner_gaussian_window, \
			target_hue_bins, \
			range(self.SATURATION_FILTER_START_INDEX, self.SATURATION_FILTER_END_INDEX))

		"""aggregate the target_hue_bins"""
		target_hue_sum = 0.0
		for i in range(self.HUE_START_INDEX, self.HUE_END_INDEX):
			target_hue_sum += filtered_inner_hist_hue[i % self.HISTBINNUM]

		if (self.HUE_END_INDEX > self.HISTBINNUM):
			aggregated_filtered_inner_hist_hue = np.concatenate((\
				filtered_inner_hist_hue[self.HUE_END_INDEX % self.HISTBINNUM:self.HUE_START_INDEX], \
				np.array([target_hue_sum]),\
				filtered_inner_hist_hue[self.HUE_END_INDEX:]), axis = 1)
		else:
			aggregated_filtered_inner_hist_hue = np.concatenate((\
				filtered_inner_hist_hue[:self.HUE_START_INDEX], \
				np.array([target_hue_sum]),\
				filtered_inner_hist_hue[self.HUE_END_INDEX:]), axis = 1)

		"""Try: normalizing border hist so that the weightage of border effect is comparable to that of the inner patch"""
		# self.hist = np.concatenate((inner_hist_hue, inner_hist_saturation, normalize(border_hist, norm = "l1")[0]), axis = 1)
		# self.hist = np.concatenate((filtered_inner_hist_hue, inner_hist_saturation, border_hist), axis = 1)
		self.hist = np.concatenate((aggregated_filtered_inner_hist_hue, border_hist), axis = 1)
		# self.hist = np.concatenate((aggregated_filtered_inner_hist_hue, normalize(border_hist, norm = "l1")[0] * \
		# 	np.sum(border_hist_hue)), axis = 1)
		self.hist = normalize(self.hist, norm='l1')[0] # normalize the histogram using l1
		
		self.inner_hist_hue = normalize(inner_hist_hue, norm = 'l1')[0]
		self.inner_hist_saturation = normalize(inner_hist_saturation, norm = 'l1')[0]
		self.border_hist_hue = normalize(border_hist_hue, norm = 'l1')[0]
		self.border_hist_saturation = normalize(border_hist_saturation, norm = 'l1')[0]
		self.border_hist = border_hist

		# print "self.HUE_START_INDEX: ", self.HUE_START_INDEX, "self.HUE_END_INDEX: ", self.HUE_END_INDEX
		
		# comparePatches.drawPatchesOnImg(np.copy(img),[self.patch, inner_patch], True)
		# plotStatistics.plotOneGivenHist("","inner_hist_hue", inner_hist_hue, save = False, show = True)
		# plotStatistics.plotOneGivenHist("","border_hist_hue", border_hist_hue, save = False, show = True)
		# plotStatistics.plotOneGivenHist("", "inner_hist_saturation", inner_hist_saturation, save = False, show = True)
		# plotStatistics.plotOneGivenHist("", "border_hist_saturation", border_hist_saturation, save = False, show = True)
		# plotStatistics.plotOneGivenHist("", "self.hist", self.hist, save = False, show = True)
		# plotStatistics.plotOneGivenHist("", "FEATURE_MODEL", self.FEATURE_MODEL, save = False, show = True)
		# plotStatistics.plotOneGivenHist("", "filtered_inner_hist_hue", filtered_inner_hist_hue, save = False, show = True)
		

	def featureResponse(self, metric_func = DIST.euclidean):
		assert (not self.hist is None), "Error in FeatureCentreParadigm: calling computeScore before the feature hist is computed!"
		assert (len(self.hist) == len(self.FEATURE_MODEL)), "Error in FeatureCentreParadigm: hist length is not correct!" + \
		"len(self.hist): {self_his_len}, len(self.FEATURE_MODEL): {feature_model_len}".format(\
			self_his_len = len(self.hist), feature_model_len = len(self.FEATURE_MODEL))
		# print "distance:", metric_func(self.hist, self.FEATURE_MODEL)

		return 1.0 / (1.0 + metric_func(self.hist, self.FEATURE_MODEL))
		# return 1.0 / (1.0 + metric_func(\
		# 	np.concatenate((self.hist[:self.HISTBINNUM], self.hist[self.HISTBINNUM*2:]), axis = 1), \
		# 	np.concatenate((self.FEATURE_MODEL[:self.HISTBINNUM], self.FEATURE_MODEL[self.HISTBINNUM*2:]), axis = 1)))

		"""Seperate comparison of response"""
		# assert (len(self.inner_hist_hue) == len(self.FEATURE_MODEL[:self.HISTBINNUM])), \
		# "Seperate feature response, inner hist hue should have the same range"
		# assert (len(self.inner_hist_saturation) == len(self.FEATURE_MODEL[self.HISTBINNUM: self.HISTBINNUM * 2])), \
		# "Seperate feature response, inner hist saturation should have the same range"
		# assert (len(self.border_hist) == len(self.FEATURE_MODEL[self.HISTBINNUM *2:])), \
		# "Seperate feature response, border hist should have the same range"

		# inner_hist_hue_distance = metric_func(self.inner_hist_hue, self.FEATURE_MODEL[:self.HISTBINNUM])
		# inner_hist_saturation_distance = metric_func(self.inner_hist_saturation, self.FEATURE_MODEL[self.HISTBINNUM: self.HISTBINNUM * 2])
		# """TODO: for border_distance can just use euclidean"""
		# border_distance = metric_func(self.border_hist, self.FEATURE_MODEL[self.HISTBINNUM *2:])

		# return 1.0 / (1.0 + np.linalg.norm([inner_hist_hue_distance, inner_hist_saturation_distance, border_distance], 2))

	def computeScore(self):
		if(self.score is None):
			self.score = self.featureResponse(comparePatches.Jensen_Shannon_Divergence)
			# self.score = self.featureResponse()


	def computeFeatureModel(self, inner_hue, inner_saturation):
		assert (not self.HUE_START_INDEX is None), "in computeFeatureModel: HUE_START_INDEX must not be None"
		assert (not self.HUE_END_INDEX is None), "in computeFeatureModel: HUE_START_INDEX must not be None"
		assert (not self.SATURATION_START_INDEX is None), "in computeFeatureModel: HUE_START_INDEX must not be None"
		assert (not self.SATURATION_END_INDEX is None), "in computeFeatureModel: HUE_START_INDEX must not be None"
		assert (len(inner_hue) == self.HISTBINNUM), "in computeFeatureModel:, inner_hue length must be the same as HISTBINNUM"
		assert (len(inner_saturation) == self.HISTBINNUM), "in computeFeatureModel:, inner_saturation length must be the same as HISTBINNUM"
		
		for i in range(self.HUE_START_INDEX, self.HUE_END_INDEX):
			# self.FEATURE_MODEL_HUE[i % self.HISTBINNUM] = 1.0/(self.HUE_END_INDEX - self.HUE_START_INDEX) # hue indexes need mod before use
			self.FEATURE_MODEL_HUE[i % self.HISTBINNUM] = inner_hue[i % self.HISTBINNUM] # hue indexes need mod before use
		for i in range(self.SATURATION_START_INDEX, self.SATURATION_END_INDEX):
			# self.FEATURE_MODEL_SATURATION[i] = 1.0/(self.SATURATION_END_INDEX - self.SATURATION_START_INDEX)
			self.FEATURE_MODEL_SATURATION[i] = inner_saturation[i]
		
		# self.FEATURE_MODEL = np.concatenate(( \
		# 	normalize(self.FEATURE_MODEL_HUE, norm = 'l1')[0], \
		# 	normalize(self.FEATURE_MODEL_SATURATION, norm = 'l1')[0], \
		# 	np.zeros(len(range(self.HUE_START_INDEX,self.HUE_END_INDEX)) + \
		# 	len(range(self.SATURATION_START_INDEX,self.SATURATION_END_INDEX)))), axis = 1) # append the expected border response

		"""aggregate Hue bins"""
		target_hue_sum = 0.0
		for i in range(self.HUE_START_INDEX, self.HUE_END_INDEX):
			target_hue_sum += self.FEATURE_MODEL_HUE[i % self.HISTBINNUM]

		if (self.HUE_END_INDEX > self.HISTBINNUM):
			AGGREGATED_HUE_MODEL = np.concatenate((\
				self.FEATURE_MODEL_HUE[self.HUE_END_INDEX % self.HISTBINNUM:self.HUE_START_INDEX], \
				np.array([target_hue_sum]),\
				self.FEATURE_MODEL_HUE[self.HUE_END_INDEX:]), axis = 1)
		else:
			AGGREGATED_HUE_MODEL = np.concatenate((\
				self.FEATURE_MODEL_HUE[:self.HUE_START_INDEX], \
				np.array([target_hue_sum]),\
				self.FEATURE_MODEL_HUE[self.HUE_END_INDEX:]), axis = 1)

		"""Here did not save saturation hist, if detection fail due to this, revert and adjust featureResponse method instead"""
		self.FEATURE_MODEL = np.concatenate(( \
			normalize(AGGREGATED_HUE_MODEL, norm = 'l1')[0], \
			np.zeros(len(range(self.HUE_START_INDEX,self.HUE_END_INDEX)) + \
			len(range(self.SATURATION_START_INDEX,self.SATURATION_END_INDEX)))), axis = 1) # append the expected border response
		
		self.FEATURE_MODEL = normalize(self.FEATURE_MODEL, norm='l1')[0] # normalize the histogram using l1

		plotStatistics.plotOneGivenHist("", "FEATURE_MODEL", self.FEATURE_MODEL, save = False, show = True)


	def fitParadigm(self, img):
		"""
		1. checks the inner hue, saturation hist, finds the mode color and see whether that color constitutes the majority of the inner patch
		2. checks the border hue, saturation hist, make sure that the border does not have the found inner hue color
		3. check for different inner GAUSSIAN_SCALE, default is 3
		returns true or false
		"""
		self.HUEFRACTION = 0.7
		self.SATURATIONFRACTION_INVERSE = 0.1 # maximum border hist
		self.SHRINK_HUE_BIN_FRACTION = 0.9

		img_hsv = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_BGR2HSV)
		gaussian_window = comparePatches.gauss_kernels(self.patch.size, sigma = self.patch.size/self.GAUSSIAN_WINDOW_LENGTH_SIGMA)

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

		self.computeFeatureModel(inner_hue, inner_saturation)

		return True

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

	