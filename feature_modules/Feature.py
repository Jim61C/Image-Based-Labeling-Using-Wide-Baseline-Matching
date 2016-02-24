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


class Feature(object):
	def __init__(self, patch, id):
		self.id = id
		self.patch = patch
		self.hist = None
		self.score = None
		self.TOP_LEFT_INDEX = 1
		self.TOP_RIGHT_INDEX = 2
		self.BOTTOM_LEFT_INDEX = 3
		self.BOTTOM_RIGHT_INDEX = 4
		self.HISTBINNUM = 16 # default value, override by sub classes
		self.GAUSSIAN_SCALE_FACTOR = 1.2

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

	def targetHueFilteredBySaturation(self, img_hsv, patch, gaussian_window, target_hue_bins, target_saturation_bins):
		hist = np.zeros(len(target_hue_bins))
		ref_x = patch.x - patch.size/2
		ref_y = patch.y - patch.size/2
		for i in range(patch.x - patch.size/2, patch.x + patch.size/2 + 1):
			for j in range(patch.y - patch.size/2, patch.y + patch.size/2 + 1):
				this_hue_bin = int(img_hsv[i][j][0]/360.0 * self.HISTBINNUM)
				if (this_hue_bin == self.HISTBINNUM):
					this_hue_bin = self.HISTBINNUM - 1
				this_saturation_bin = int(img_hsv[i][j][1]/1.0 * self.HISTBINNUM)
				if (this_saturation_bin == self.HISTBINNUM):
					this_saturation_bin = self.HISTBINNUM - 1
				if (this_hue_bin in target_hue_bins and this_saturation_bin in target_saturation_bins):
					hist[target_hue_bins.index(this_hue_bin)] += 1 * gaussian_window[i - ref_x][j - ref_y]
		return hist

	def getSubPatchTargetHueFilteredBySaturation(self, img_hsv, sub_patch_index, target_hue_bins, target_saturation_bins):
		newLen = (self.patch.size+1)/2
		if(newLen % 2 == 0):
			newSize = newLen -1
		else:
			newSize = newLen
		gaussian_window = comparePatches.gauss_kernels(self.patch.size, sigma = self.patch.size/6.0)
		# newSize is the size of the sub patch
		if(sub_patch_index == self.TOP_LEFT_INDEX):
			sub_gaussian_window = gaussian_window[0:newSize,0:newSize]
			sub_patch = comparePatches.Patch(self.patch.x - newLen/2, self.patch.y - newLen/2, newSize, initialize_features = False)
		elif(sub_patch_index == self.TOP_RIGHT_INDEX):
			sub_gaussian_window = gaussian_window[0:newSize, gaussian_window.shape[1] - newSize:gaussian_window.shape[1]]
			sub_patch = comparePatches.Patch(self.patch.x - newLen/2, self.patch.y + newLen/2, newSize, initialize_features = False)
		elif(sub_patch_index == self.BOTTOM_LEFT_INDEX):
			sub_gaussian_window = gaussian_window[gaussian_window.shape[0] - newSize:gaussian_window.shape[0], 0:newSize]
			sub_patch = comparePatches.Patch(self.patch.x + newLen/2, self.patch.y - newLen/2, newSize, initialize_features = False)
		else:
			sub_gaussian_window = gaussian_window[gaussian_window.shape[0] - newSize:gaussian_window.shape[0], gaussian_window.shape[1] - newSize: gaussian_window.shape[1]]
			sub_patch = comparePatches.Patch(self.patch.x + newLen/2, self.patch.y + newLen/2, newSize, initialize_features = False)

		return self.targetHueFilteredBySaturation(img_hsv, sub_patch, sub_gaussian_window, target_hue_bins, target_saturation_bins)


	def computeFeature(self, img, useGaussianSmoothing = True):
		return

	def featureResponse(self):
		return

	def computeScore(self):
		return
		
	def setScore(self, score):
		self.score = score
		return

	def dissimilarityWith(self, hist):
		"""
		Default is l2 distance: overwritten by sub classes for different behaviour
		"""
		self.assertHist(hist)
		# return DIST.euclidean(self.hist, hist)
		return comparePatches.Jensen_Shannon_Divergence(self.hist, hist)

	def assertHist(self, hist):
		assert (len(self.hist) == len(hist)), "Error in feature " + self.id + ": Compared hist must have the same length as self.hist"
		self_hist_norm = int(np.linalg.norm(self.hist, 1))

		assert ( self_hist_norm == 1 or self_hist_norm == 0), "Error in feature " + self.id + ": self.hist must be l1 normalized" + \
		", but get {norm}".format(norm = self_hist_norm)
		
		input_hist_norm = int(np.linalg.norm(hist, 1))
		assert ( input_hist_norm == 1 or input_hist_norm == 0), "Error in feature " + self.id + ": input hist must be l1 normalized" + \
		", but get {norm}".format(norm = input_hist_norm)



