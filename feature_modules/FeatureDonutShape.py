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

class FeatureDonutShape(Feature):
	def __init__(self, patch, id):
		Feature.__init__(self, patch, id)
		self.HISTBINNUM = 16
		self.FEATURE_MODEL = np.array([0.00351539,  0.57596057,  0.06858266,  0.02048219,  0.02166264,  0.02258441,
									  0.01227705,  0.01083268,  0.00764063,  0.01532651,  0.0044623,   0.,          0.,
									  0.,         0.,          0.,          0.,          0.00621184,  0.01046716,
									  0.01155839,  0.00736686,  0.01444524,  0.0449813,  0.05170737,  0.01959358,
									  0.02875307,  0.04158816,  0.,          0.,          0.,          0.,          0.        ])
		self.FEATURE_MODEL = normalize(self.FEATURE_MODEL, norm='l1')[0] # normalize the histogram using l1

	def computeIntensityHist(self,img_gray, patch,gaussian_window):
		hist = np.zeros(self.HISTBINNUM)
		ref_x = patch.x - patch.size/2
		ref_y = patch.y - patch.size/2
		for i in range(patch.x - patch.size/2, patch.x + patch.size/2 + 1):
			for j in range(patch.y - patch.size/2, patch.y + patch.size/2 + 1):
				this_bin = int(img_gray[i][j]/256.0 * self.HISTBINNUM)
				hist[this_bin] += 1 * gaussian_window[i - ref_x][j - ref_y]
				# hist[this_bin] += 1
		return hist

	def computeFeature(self, img, useGaussianSmoothing = True):
		img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.uint8)
		inner_patch_size = comparePatches.getGaussianScale(self.patch.size, 1.2, -2)
		inner_patch = comparePatches.Patch(self.patch.x, self.patch.y, inner_patch_size)
		gaussian_window = comparePatches.gauss_kernels(self.patch.size, sigma = self.patch.size/4.0)
		inner_gaussian_window = gaussian_window[ gaussian_window.shape[0]/2 - inner_patch.size/2: gaussian_window.shape[0]/2 + inner_patch.size/2 + 1 ,\
		 gaussian_window.shape[1]/2 - inner_patch.size/2: gaussian_window.shape[1]/2 + inner_patch.size/2 + 1]
		assert gaussian_window.shape == (self.patch.size, self.patch.size), "outer gaussian_window size not correct"
		assert inner_gaussian_window.shape == (inner_patch.size, inner_patch.size), "inner gaussian_window size not correct"

		outer_hist = self.computeIntensityHist(img_gray, self.patch, gaussian_window)
		inner_hist = self.computeIntensityHist(img_gray, inner_patch, inner_gaussian_window)
		donut_hist = outer_hist - inner_hist
		self.hist = np.concatenate((inner_hist, donut_hist), axis = 1)
		self.hist = normalize(self.hist, norm='l1')[0] # normalize the histogram using l1
		# print self.hist
		# plt.plot(outer_hist)
		# plt.show()

		# plt.plot(inner_hist)
		# plt.show()

		# plt.plot(donut_hist)
		# plt.show()

		# plt.plot(self.hist)
		# plt.show()

		# comparePatches.drawPatchesOnImg(img_gray,[self.patch, inner_patch], True)

	def featureResponse(self):
		assert (not self.hist is None), "Error in FeatureDonutShape: calling computeScore before the feature hist is computed!"
		# return np.sum(self.hist)
		return 1.0 / (1.0 + DIST.euclidean(self.hist, self.FEATURE_MODEL))

	def computeScore(self):
		if(self.score is None):
			self.score = self.featureResponse()
	