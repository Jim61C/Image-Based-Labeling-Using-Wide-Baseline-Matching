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
import cornerResponse


class FeatureCentreHOGParadigm(Feature):
	def __init__(self, patch, id):
		Feature.__init__(self, patch, id)
		# other than the cutted bins of wanted HOG direction, all other HOG bins' values are zero
		self.FEATURE_MODEL_INNER = None
		self.FEATURE_MODEL_BORDER = None
			
		self.inner_HOG = None
		self.border_HOG = None
		self.GAUSSIAN_SCALE = 3
		self.GAUSSIAN_WINDOW_LENGTH_SIGMA = 4.0 # used for border feature case, window length = 4 sigma

	def computeFeature(self, img, useGaussianSmoothing = True):
		inner_patch_size = comparePatches.getGaussianScale(self.patch.size, self.GAUSSIAN_SCALE_FACTOR, -self.GAUSSIAN_SCALE)
		inner_patch = comparePatches.Patch(self.patch.x, self.patch.y, inner_patch_size)

		gaussian_window = comparePatches.gauss_kernels(self.patch.size, sigma = self.patch.size/self.GAUSSIAN_WINDOW_LENGTH_SIGMA)
		inner_gaussian_window = gaussian_window[ \
		gaussian_window.shape[0]/2 - inner_patch.size/2: gaussian_window.shape[0]/2 + inner_patch.size/2 + 1 ,\
		gaussian_window.shape[1]/2 - inner_patch.size/2: gaussian_window.shape[1]/2 + inner_patch.size/2 + 1]

		assert gaussian_window.shape == (self.patch.size, self.patch.size), "outer gaussian_window size not correct"
		assert inner_gaussian_window.shape == (inner_patch.size, inner_patch.size), "inner gaussian_window size not correct"

		"""inner patch HOG"""
		self.outer_HOG = self.patch.computeSinglePatchHOG(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.int), gaussian_window)
		self.inner_HOG = inner_patch.computeSinglePatchHOG(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.int), inner_gaussian_window)
		
		"""border HOG"""
		self.border_HOG = self.patch.HOG_Uncirculated - inner_patch.HOG_Uncirculated
		max_ori = np.argmax(self.border_HOG) # use maximum
		self.border_HOG =  np.array(list(self.border_HOG[max_ori:len(self.border_HOG)]) + \
		list(self.border_HOG[0:max_ori])) # rotate circular hist

		self.hist = np.concatenate((self.inner_HOG, self.border_HOG), axis = 1)
		return


	def featureResponse(self):
		assert (not self.hist is None), "Error in FeatureCentreHOGParadigm: calling computeScore before the feature hist is computed!"
		assert (not self.FEATURE_MODEL_INNER is None), "Error in FeatureCentreHOGParadigm: calling computeScore before FEATURE_MODEL_INNER is computed!"
		assert (not self.FEATURE_MODEL_BORDER is None), "Error in FeatureCentreHOGParadigm: calling computeScore before FEATURE_MODEL_BORDER is computed!"
		
		assert (len(self.inner_HOG) == len(self.FEATURE_MODEL_INNER)), "Error in FeatureCentreHOGParadigm: inner HOG length is not correct!"
		assert (len(self.border_HOG) == len(self.FEATURE_MODEL_BORDER)), "Error in FeatureCentreHOGParadigm: border HOG length is not correct!"
		dissimilarity_inner = comparePatches.Jensen_Shannon_Divergence_Hat(self.inner_HOG, self.FEATURE_MODEL_INNER)
		dissimilarity_border = comparePatches.Jensen_Shannon_Divergence_Hat(self.border_HOG, self.FEATURE_MODEL_BORDER)
		return 1.0 / (1.0 + np.linalg.norm([dissimilarity_inner, dissimilarity_border], 2))
		# return np.sum(self.hist) # if all HOG degree are at the cutted HOG bins, response will be 1.0

	def computeScore(self):
		"""
		set self.score
		"""
		if(self.score is None):
			self.score = self.featureResponse()

	def dissimilarityWith(self, feature_obj):
		assert (not self.hist is None), \
		"Error in FeatureCentreHOGParadigm dissimilarityWith: self hist is not computed!"
		assert (not feature_obj.hist is None), \
		"Error in FeatureCentreHOGParadigm dissimilarityWith: feature_obj hist is not computed!"
		assert (len(self.inner_HOG) == len(feature_obj.inner_HOG)), \
		"Error in FeatureCentreHOGParadigm dissimilarityWith: inner HOG length is not correct!"
		assert (len(self.border_HOG) == len(feature_obj.border_HOG)), \
		"Error in FeatureCentreHOGParadigm dissimilarityWith: border HOG length is not correct!"

		dissimilarity_inner = comparePatches.Jensen_Shannon_Divergence_Hat(self.inner_HOG, feature_obj.inner_HOG)
		dissimilarity_border = comparePatches.Jensen_Shannon_Divergence_Hat(self.border_HOG, feature_obj.border_HOG)
		return np.linalg.norm([dissimilarity_inner, dissimilarity_border], 2)

	def fitParadigm(self, img):
		"""
		it's clicker's responsibility to make sure that the patch is having a shape in the center patch, but nothing in the border
		"""
		BORDER_HOG_FRACTION_THRESH = 0.4
		inner_patch_size = comparePatches.getGaussianScale(self.patch.size, self.GAUSSIAN_SCALE_FACTOR, -self.GAUSSIAN_SCALE)
		inner_patch = comparePatches.Patch(self.patch.x, self.patch.y, inner_patch_size)

		gaussian_window = comparePatches.gauss_kernels(self.patch.size, sigma = self.patch.size/self.GAUSSIAN_WINDOW_LENGTH_SIGMA)
		inner_gaussian_window = gaussian_window[ \
		gaussian_window.shape[0]/2 - inner_patch.size/2: gaussian_window.shape[0]/2 + inner_patch.size/2 + 1 ,\
		gaussian_window.shape[1]/2 - inner_patch.size/2: gaussian_window.shape[1]/2 + inner_patch.size/2 + 1]

		assert gaussian_window.shape == (self.patch.size, self.patch.size), "outer gaussian_window size not correct"
		assert inner_gaussian_window.shape == (inner_patch.size, inner_patch.size), "inner gaussian_window size not correct"

		"""inner patch HOG"""
		outer_HOG = self.patch.computeSinglePatchHOG(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.int), gaussian_window)
		inner_HOG = inner_patch.computeSinglePatchHOG(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.int), inner_gaussian_window)
		
		"""border HOG"""
		border_HOG = self.patch.HOG_Uncirculated - inner_patch.HOG_Uncirculated
		max_ori = np.argmax(border_HOG) # use maximum
		border_HOG =  np.array(list(border_HOG[max_ori:len(border_HOG)]) + \
		list(border_HOG[0:max_ori])) # rotate circular hist

		"""border should be quite plain, not high HOG response"""
		print "np.sum(inner_HOG) / np.sum(outer_HOG):", np.sum(inner_HOG) / np.sum(outer_HOG)
		print "np.sum(border_HOG) / np.sum(outer_HOG):", np.sum(border_HOG) / np.sum(outer_HOG)
		if (np.sum(border_HOG) / np.sum(outer_HOG) > BORDER_HOG_FRACTION_THRESH):
			return False
		self.computeFeatureModel(inner_HOG, border_HOG)
		return True

	def computeFeatureModel(self, inner_HOG, border_HOG):
		self.FEATURE_MODEL_INNER = inner_HOG
		self.FEATURE_MODEL_BORDER = border_HOG
		return







