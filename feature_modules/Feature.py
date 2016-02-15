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

	def computeFeature(self, img, useGaussianSmoothing = True):
		return

	def featureResponse(self):
		return

	def computeScore(self):
		return
		
	def setScore(self, score):
		self.score = score
		return



