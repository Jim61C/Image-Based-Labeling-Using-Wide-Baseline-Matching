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


class FeatureHeartShape(Feature):
	def __init__(self, patch, id):
		Feature.__init__(self, patch, id)

		self.FEATURE_MODEL = utils.loadArray("FeatureHeartShapeContourModel.npz")
		self.FEATURE_MODEL_NUM_CONTOURS = 2

		self._checkRep()

	def computeFeature(self, img, useGaussianSmoothing = True):
		"""Find contour"""
		roi = img[self.patch.x - self.patch.size/2 : self.patch.x + self.patch.size/2  + 1, \
		self.patch.y - self.patch.size/2 : self.patch.y + self.patch.size/2  + 1]
		# print "roi.shape:", roi.shape
		roi_gray = cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
		ret,thresh = cv2.threshold(roi_gray,127,255,0)
		# cv2.imshow("thresh:", thresh)
		# cv2.waitKey(0)
		contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

		"""
		Model Construction (find heart shape contour)
		TODO: record feature model patch size, and scale (x,y) in contours to the targetted size when computing feature
		"""
		# print "number of contours found:", len(contours)
		# if (len(contours) ==2):
			# cv2.drawContours(roi, [contours[0]], 0, (0,255,0), 3)
			# cv2.imshow("roi with contours:", roi)
			# cv2.waitKey(0)
			# cv2.drawContours(roi, [contours[1]], 0, (0,255,0), 3)
			# cv2.imshow("roi with contours:", roi)
			# cv2.waitKey(0)
			# print "match contours[0] with contours[1]:", cv2.matchShapes(contours[0], contours[1], cv.CV_CONTOURS_MATCH_I2, 0)
			# print "match contours[1] with contours[1]:", cv2.matchShapes(contours[1], contours[1], cv.CV_CONTOURS_MATCH_I2, 0)
			# print "2nd Contour.shape:", contours[1].shape
			# print "2nd Contour:\n", contours[1]
			# utils.saveArray(contours[1], "FeatureHeartShapeContourModel")

		self.hist = contours
		return


	def featureResponse(self):
		assert (not self.hist is None), "Error in FeatureHeartShape: calling computeScore before the feature hist is computed!"
		dissimilarity = sys.maxint
		for contour in self.hist:
			this_dissimilarity = cv2.matchShapes(contour, self.FEATURE_MODEL, cv.CV_CONTOURS_MATCH_I3, 0)
			if (this_dissimilarity < dissimilarity):
				dissimilarity = this_dissimilarity
		return 1.0 / (1.0 + dissimilarity + abs(len(self.hist) - self.FEATURE_MODEL_NUM_CONTOURS))

	def computeScore(self):
		"""
		set self.score
		"""
		if(self.score is None):
			self.score = self.featureResponse()

	def dissimilarityWith(self, feature_obj):
		hist = feature_obj.hist
		self.assertHist(hist)
		return comparePatches.Jensen_Shannon_Divergence(self.hist, hist)

	def _checkRep(self):
		assert (self.id == utils.HEART_SHAPE_FEATURE_ID), "Error in FeatureHeartShape: id is not correctly set: {id}".format(id = self.id)



