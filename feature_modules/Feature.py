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

	def computeHueHistSaturationWeighted(self, img_hsv, patch, gaussian_window):
		hist = np.zeros(self.HISTBINNUM)
		ref_x = patch.x - patch.size/2
		ref_y = patch.y - patch.size/2
		for i in range(patch.x - patch.size/2, patch.x + patch.size/2 + 1):
			for j in range(patch.y - patch.size/2, patch.y + patch.size/2 + 1):
				this_bin = int(img_hsv[i][j][0]/360.0 * self.HISTBINNUM)
				if (this_bin == self.HISTBINNUM):
					this_bin = self.HISTBINNUM - 1
				# weight by saturation value [0,1]
				if (not gaussian_window is None):
					hist[this_bin] += 1 * gaussian_window[i - ref_x][j - ref_y] * img_hsv[i][j][1]
				else:
					hist[this_bin] += 1 * img_hsv[i][j][1]
		return hist

	def computeHueHist(self,img_hsv, patch,gaussian_window):
		hist = np.zeros(self.HISTBINNUM)
		ref_x = patch.x - patch.size/2
		ref_y = patch.y - patch.size/2
		for i in range(patch.x - patch.size/2, patch.x + patch.size/2 + 1):
			for j in range(patch.y - patch.size/2, patch.y + patch.size/2 + 1):
				this_bin = int(img_hsv[i][j][0]/360.0 * self.HISTBINNUM)
				if (this_bin == self.HISTBINNUM):
					this_bin = self.HISTBINNUM - 1
				if (gaussian_window is None):
					hist[this_bin] += 1
				else:	
					hist[this_bin] += 1 * gaussian_window[i - ref_x][j - ref_y]
				
		return hist

	def computeSaturationHist(self, img_hsv, patch, gaussian_window):
		"""
		img_hsv: Hue: 0-360, Saturation: 0-1, Value: 0-255
		"""
		hist = np.zeros(self.HISTBINNUM)
		ref_x = patch.x - patch.size/2
		ref_y = patch.y - patch.size/2
		for i in range(patch.x - patch.size/2, patch.x + patch.size/2 + 1):
			for j in range(patch.y - patch.size/2, patch.y + patch.size/2 + 1):
				this_bin = int(img_hsv[i][j][1]/1.0 * self.HISTBINNUM)
				if (this_bin == self.HISTBINNUM):
					this_bin = self.HISTBINNUM - 1
				if (gaussian_window is None):
					hist[this_bin] += 1
				else:
					hist[this_bin] += 1 * gaussian_window[i - ref_x][j - ref_y]
				
		return hist


	def withinPatch(self, patch, i, j):
		"""
		checkes if the given indexes: i, j are within the patch
		"""
		if (i < patch.x + patch.size/2 + 1 and \
			j < patch.y + patch.size/2 + 1 and \
			i >= patch.x - patch.size/2 and \
			j >= patch.y - patch.size/2):
			return True
		else:
			return False


	def borderTargetHueFilteredBySaturation(self, img_hsv, patch, inner_patch, gaussian_window, target_hue_bins, target_saturation_bins):
		"""
		img_hsv: Hue: 0-360.0, Saturation: 0-1.0, Value: 0-255.0
		inner_patch: square inner patch with a smaller window size than patch
		return: hist of length equal to self.HISTBINNUM for the targetHueFilteredBySaturation
		"""
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

	def computeHS2DArr(self, img_hsv, patch, gaussian_window):
		"""
		patch: instance of patch to set its hs_2d_arr
		"""
		full_patch_hs = self.computeHS2DWithGaussianWindow(img_hsv, patch, gaussian_window)
		patch.hs_2d_arr.append(full_patch_hs) # full_patch_hs is supposed to be using 6 sigma gaussian window

		newLen = (patch.size+1)/2
		if(newLen % 2 == 0):
			newSize = newLen -1
		else:
			newSize = newLen

		# TOP_LEFT_INDEX  patch.hs_2d_arr[1]
		sub_gaussian_window = gaussian_window[0:newSize,0:newSize]
		sub_patch = comparePatches.Patch(patch.x - newLen/2, patch.y - newLen/2, newSize, initialize_features = False)
		patch.hs_2d_arr.append(self.computeHS2DWithGaussianWindow(img_hsv, sub_patch, sub_gaussian_window))
		# TOP_RIGHT_INDEX  patch.hs_2d_arr[2]
		sub_gaussian_window = gaussian_window[0:newSize, gaussian_window.shape[1] - newSize:gaussian_window.shape[1]]
		sub_patch = comparePatches.Patch(patch.x - newLen/2, patch.y + newLen/2, newSize, initialize_features = False)
		patch.hs_2d_arr.append(self.computeHS2DWithGaussianWindow(img_hsv, sub_patch, sub_gaussian_window))
		# BOTTOM_LEFT_INDEX patch.hs_2d_arr[3]
		sub_gaussian_window = gaussian_window[gaussian_window.shape[0] - newSize:gaussian_window.shape[0], 0:newSize]
		sub_patch = comparePatches.Patch(patch.x + newLen/2, patch.y - newLen/2, newSize, initialize_features = False)
		patch.hs_2d_arr.append(self.computeHS2DWithGaussianWindow(img_hsv, sub_patch, sub_gaussian_window))
		# BOTTOM_RIGHT_INDEX patch.hs_2d_arr[4]
		sub_gaussian_window = gaussian_window[gaussian_window.shape[0] - newSize:gaussian_window.shape[0], gaussian_window.shape[1] - newSize: gaussian_window.shape[1]]
		sub_patch = comparePatches.Patch(patch.x + newLen/2, patch.y + newLen/2, newSize, initialize_features = False)
		patch.hs_2d_arr.append(self.computeHS2DWithGaussianWindow(img_hsv, sub_patch, sub_gaussian_window))

	def computeHS2DWithGaussianWindow(self, img_hsv, patch, gaussian_window):
		"""
		default is 16 * 16, unless self.HISTBINNUM is being extended by subclasses
		"""
		assert (patch.size == len(gaussian_window)), \
		"In computeHS2DWithGaussianWindow, passed in patch size mush match gaussian_window, however, patch.size =={a}, len(gaussian_window) == {b}".format(\
			a = patch.size, b = len(gaussian_window))
		hist = np.zeros(shape = (self.HISTBINNUM, self.HISTBINNUM))
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

				hist[this_hue_bin][this_saturation_bin] += 1 * gaussian_window[i - ref_x][j - ref_y]
		return hist

	def derive1DSaturationFrom2D(self, hist_2d):
		"""
		hist_2d is of shape (Hue bin length, Saturation bin length)
		"""
		assert (self.HISTBINNUM == len(hist_2d)), \
		"hist_2d must have side length of HISTBINNUM but len(hist_2d) is {a} while HISTBINNUM is {b}".format(\
			a = len(hist_2d), b = self.HISTBINNUM)
		hist = np.zeros(self.HISTBINNUM)
		# saturation is col
		for i in range(0, len(hist)):
			hist[i] = np.sum(hist_2d[:,i])
		return hist

	def derive1DHueFrom2D(self, hist_2d):
		"""
		hist_2d is of shape (Hue bin length, Saturation bin length)
		"""
		assert (self.HISTBINNUM == len(hist_2d)), \
		"hist_2d must have side length of HISTBINNUM but len(hist_2d) is {a} while HISTBINNUM is {b}".format(\
			a = len(hist_2d), b = self.HISTBINNUM)
		hist = np.zeros(self.HISTBINNUM)
		# hue is row
		for i in range(0, len(hist)):
			hist[i] = np.sum(hist_2d[i,])
		return hist

	def deriveSaturationHistFilterOffSaturationWithWrongHueFrom2D(self, hist_2d, target_saturation_bins, target_hue_bins):
		"""
		hist_2d is of shape (Hue bin length, Saturation bin length)
		"""
		assert (self.HISTBINNUM == len(hist_2d)), \
		"hist_2d must have side length of HISTBINNUM but len(hist_2d) is {a} while HISTBINNUM is {b}".format(\
			a = len(hist_2d), b = self.HISTBINNUM)
		hist = np.zeros(self.HISTBINNUM)
		# saturation is col
		for i in range(0, len(hist)):
			if (not i in target_saturation_bins):
				hist[i] = np.sum(hist_2d[:,i])
			else:
				hist[i] = np.sum(hist_2d[target_hue_bins,i])
		return hist

	def deriveHueHistFilterOffHueWithWrongSaturationFrom2D(self, hist_2d, target_saturation_bins, target_hue_bins):
		"""
		hist_2d is of shape (Hue bin length, Saturation bin length)
		"""
		assert (self.HISTBINNUM == len(hist_2d)), \
		"hist_2d must have side length of HISTBINNUM but len(hist_2d) is {a} while HISTBINNUM is {b}".format(\
			a = len(hist_2d), b = self.HISTBINNUM)
		hist = np.zeros(self.HISTBINNUM)
		# hue is row
		for i in range(0, len(hist)):
			if (not i in target_hue_bins):
				hist[i] = np.sum(hist_2d[i,])
			else:
				hist[i] = np.sum(hist_2d[i,target_saturation_bins])
		return hist

	def computeSaturationHistFilterOffSaturationWithWrongHue(self, img_hsv, patch, gaussian_window, target_hue_bins, target_saturation_bins):
		"""Do not add to Hue Hist if the hue is in target_saturation_bins but not in the target_saturation_bins"""
		hist = np.zeros(self.HISTBINNUM)
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

				"""If in target_saturation_bins but not correct hue, ignore"""
				if (not (this_saturation_bin in target_saturation_bins and (not this_hue_bin in target_hue_bins))):
					hist[this_saturation_bin] += 1 * gaussian_window[i - ref_x][j - ref_y]
		return hist

	def computeHueHistFilterOffHueWithWrongSaturation(self, img_hsv, patch, gaussian_window, target_hue_bins, target_saturation_bins):
		"""Do not add to Hue Hist if the hue is in target_saturation_bins but not in the target_saturation_bins"""
		hist = np.zeros(self.HISTBINNUM)
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

				"""If in target_hue_bins but not correct saturation, ignore"""
				if (not (this_hue_bin in target_hue_bins and (not this_saturation_bin in target_saturation_bins))):
					hist[this_hue_bin] += 1 * gaussian_window[i - ref_x][j - ref_y]
		return hist

	def getTargetHueAndSaturationBins(self):
		assert (not self.HUE_START_INDEX is None), "In getTargetHueAndSaturationBins: HUE_START_INDEX must not be None"
		assert (not self.HUE_END_INDEX is None), "In getTargetHueAndSaturationBins: HUE_END_INDEX must not be None"
		assert (not self.SATURATION_START_INDEX is None), "In getTargetHueAndSaturationBins: SATURATION_START_INDEX must not be None"
		assert (not self.SATURATION_END_INDEX is None), "In getTargetHueAndSaturationBins: SATURATION_END_INDEX must not be None"

		target_hue_bins = []
		for i in range(self.HUE_START_INDEX, self.HUE_END_INDEX):
			target_hue_bins.append(i % self.HISTBINNUM)
		target_saturation_bins = range(self.SATURATION_START_INDEX, self.SATURATION_END_INDEX)
		return target_hue_bins, target_saturation_bins

	def getSubPatchAndSubPatchGaussianFromSubPatchIndex(self, sub_patch_index):
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

		return sub_patch, sub_gaussian_window, gaussian_window

	def deriveSubPatchTargetHueFilteredBySaturationFrom2DArr(self, img_hsv, sub_patch_index, hs_2d_arr, \
		target_hue_bins, target_saturation_bins):
		"""
		hs_2d_arr: of length 5, hs_2d_arr[0] is full patch hs 2d
		sub_patch_index: one of TOP_LEFT_INDEX(1), TOP_RIGHT_INDEX(2), BOTTOM_LEFT_INDEX(3), BOTTOM_RIGHT_INDEX(4)
		"""
		hs_2d_target = hs_2d_arr[sub_patch_index]
		hist = np.zeros(len(target_hue_bins))
		for this_hue_bin in target_hue_bins:
			hist[target_hue_bins.index(this_hue_bin)] = np.sum(hs_2d_target[this_hue_bin, target_saturation_bins])
		return hist

	def getSubPatchTargetHueFilteredBySaturation(self, img_hsv, sub_patch_index, target_hue_bins, target_saturation_bins):
		sub_patch, sub_gaussian_window, _ = self.getSubPatchAndSubPatchGaussianFromSubPatchIndex(sub_patch_index)
		return self.targetHueFilteredBySaturation(img_hsv, sub_patch, sub_gaussian_window, target_hue_bins, target_saturation_bins)


	def findValueHistForTargetHueBin(self, img_hsv, patch, gaussian_window, target_hue_bins):
		hist = np.zeros(self.HISTBINNUM)
		ref_x = patch.x - patch.size/2
		ref_y = patch.y - patch.size/2
		for i in range(patch.x - patch.size/2, patch.x + patch.size/2 + 1):
			for j in range(patch.y - patch.size/2, patch.y + patch.size/2 + 1):
				this_hue_bin = int(img_hsv[i][j][0]/360.0 * self.HISTBINNUM)
				if (this_hue_bin == self.HISTBINNUM):
					this_hue_bin = self.HISTBINNUM - 1

				this_value_bin = int(img_hsv[i][j][2]/255.0 * self.HISTBINNUM)
				if (this_value_bin == self.HISTBINNUM):
					this_value_bin = self.HISTBINNUM - 1

				if (this_hue_bin in target_hue_bins):
					# print "for ", this_hue_bin, " saturation bin is:", this_saturation_bin
					hist[this_value_bin] += 1 * gaussian_window[i - ref_x][j - ref_y]
		return hist

	def findSaturationHistForTargetHueBin(self, img_hsv, patch, gaussian_window, target_hue_bins):
		hist = np.zeros(self.HISTBINNUM)
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

				if (this_hue_bin in target_hue_bins):
					# print "for ", this_hue_bin, " saturation bin is:", this_saturation_bin
					hist[this_saturation_bin] += 1 * gaussian_window[i - ref_x][j - ref_y]
		return hist

	def findBorderValueHistForTargetHueBin(self, img_hsv, patch, inner_patch, gaussian_window, target_hue_bins):
		"""For border: find the value hist when hue lies in the target hue bins"""
		hist = np.zeros(self.HISTBINNUM)
		ref_x = patch.x - patch.size/2
		ref_y = patch.y - patch.size/2
		for i in range(patch.x - patch.size/2, patch.x + patch.size/2 + 1):
			for j in range(patch.y - patch.size/2, patch.y + patch.size/2 + 1):
				if (not self.withinPatch(inner_patch, i, j)): # only check the border pixels
					this_hue_bin = int(img_hsv[i][j][0]/360.0 * self.HISTBINNUM)
					if (this_hue_bin == self.HISTBINNUM):
						this_hue_bin = self.HISTBINNUM - 1
					this_value_bin = int(img_hsv[i][j][2]/255.0 * self.HISTBINNUM)
					if (this_value_bin == self.HISTBINNUM):
						this_value_bin = self.HISTBINNUM - 1

					if (this_hue_bin in target_hue_bins):
						hist[this_value_bin] += 1 * gaussian_window[i - ref_x][j - ref_y]
		return hist

	def findBorderSaturationHistForTargetHueBin(self, img_hsv, patch, inner_patch, gaussian_window, target_hue_bins):
		"""For border: find the saturation hist when hue lies in the target hue bins"""
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

					if (this_hue_bin in target_hue_bins):
						hist[this_saturation_bin] += 1 * gaussian_window[i - ref_x][j - ref_y]
		return hist


	def findSaturationRangeForTargetHueBin(self, img_hsv, inner_patch, target_hue_bins, inner_gaussian_window):
		"""return: the saturation range of the given target_hue_bins"""

		hist = self.findSaturationHistForTargetHueBin(img_hsv, inner_patch, inner_gaussian_window, target_hue_bins)

		return self.acquireSaturationWorker(hist)

	def findBorderSaturationRangeForTargetHueBin(self, img_hsv, patch, inner_patch, gaussian_window, target_hue_bins):
		"""
		img_hsv: Hue: 0-360, Saturation: 0-1, Value: 0-255
		inner_patch: square inner patch with a smaller window size than patch
		return: hist of length equal to self.HISTBINNUM for the targetHueFilteredBySaturation
		"""
		hist = self.findBorderSaturationHistForTargetHueBin(img_hsv, patch, inner_patch, gaussian_window, target_hue_bins)

		return self.acquireSaturationWorker(hist)
	
	def acquireSaturationWorker(self, hist):
		SATURATION_ACQUIRE_FRACTION = 0.5 # reduced to tackle cases where saturation changes a lot across images of the same patch
		SATURATION_FILTER_FRACTION = 0.1 # used to settle SATURATION FILTER BINS
		SATURATION_ACQUIRE_BIN_NEIGHBOURHOOD = int(10/36.0 * self.HISTBINNUM)
		
		non_zeros_saturation_bins = np.where(hist != 0)[0]
		max_saturation_bin = np.argmax(hist)
		"""acquire bins based on response percentage"""
		# acquired_saturation_bins = []
		# for bin in non_zeros_saturation_bins:
		# 	if (hist[bin] > SATURATION_ACQUIRE_FRACTION * hist[max_saturation_bin]):
		# 		acquired_saturation_bins.append(bin)
		"""
		acquire bins based on neighbourhood, empirically estimate that 7/36 bins up and down away from mode saturation bin will be good
		Or, other than having separate filter_saturation_bins, adjust the SATURATION_ACQUIRE_FRACTION to be lower
		"""
		acquired_saturation_bins = []
		for bin in range(max_saturation_bin - SATURATION_ACQUIRE_BIN_NEIGHBOURHOOD, \
			max_saturation_bin + SATURATION_ACQUIRE_BIN_NEIGHBOURHOOD + 1):
			if (bin >= 0 and bin < self.HISTBINNUM and hist[bin] > SATURATION_ACQUIRE_FRACTION * hist[max_saturation_bin]):
				acquired_saturation_bins.append(bin)

		min_acquired_bin = np.min(acquired_saturation_bins)
		max_acquired_bin = np.max(acquired_saturation_bins) + 1
		filter_saturation_bins = []
		bin_to_go_up_down = (2 * SATURATION_ACQUIRE_BIN_NEIGHBOURHOOD - (max_acquired_bin - min_acquired_bin))/2
		for bin in range(min_acquired_bin - bin_to_go_up_down, max_acquired_bin + bin_to_go_up_down):
			if (bin >=0 and bin < self.HISTBINNUM and hist[bin] > SATURATION_FILTER_FRACTION * hist[max_saturation_bin]):
				filter_saturation_bins.append(bin)
		
		return min_acquired_bin, max_acquired_bin , np.min(filter_saturation_bins), np.max(filter_saturation_bins) + 1

	def setPatch(self, patch):
		self.patch = patch
	
	def computeFeature(self, img, useGaussianSmoothing = True):
		return

	def featureResponse(self):
		return

	def computeScore(self):
		return
		
	def setScore(self, score):
		self.score = score
		return

	def dissimilarityWith(self, feature_obj):
		"""
		Default is l2 distance: overwritten by sub classes for different behaviour
		"""
		hist = feature_obj.hist
		self.assertHist(hist)
		# return DIST.euclidean(self.hist, hist)
		return comparePatches.Jensen_Shannon_Divergence_Unnormalized(self.hist, hist)

	def assertHist(self, hist):
		assert (len(self.hist) == len(hist)), "Error in feature " + self.id + ": Compared hist must have the same length as self.hist"
		self_hist_norm = int(np.linalg.norm(self.hist, 1))

		assert ( self_hist_norm == 1 or self_hist_norm == 0), "Error in feature " + self.id + ": self.hist must be l1 normalized" + \
		", but get {norm}".format(norm = self_hist_norm)
		
		input_hist_norm = int(np.linalg.norm(hist, 1))
		assert ( input_hist_norm == 1 or input_hist_norm == 0), "Error in feature " + self.id + ": input hist must be l1 normalized" + \
		", but get {norm}".format(norm = input_hist_norm)



