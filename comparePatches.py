import cv2
from cv2 import cv
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import entropy
import pyemd
import math
import cornerResponse
import sys
import drawMatches
import plotStatistics
import operator
from sklearn.preprocessing import normalize
import saveLoadPatch
import itertools
import random
import feature_modules
import scipy.spatial.distance as DIST
from feature_modules import utils
import time

HUE_16BIN_C = np.array(
[[ 1,  2,  3,  4,  5,  6,  7,  8,  9,  8,  7,  6,  5,  4,  3,  2],
 [ 2,  1,  2,  3,  4,  5,  6,  7,  8,  9,  8,  7,  6,  5,  4,  3],
 [ 3,  2,  1,  2,  3,  4,  5,  6,  7,  8,  9,  8,  7,  6,  5,  4],
 [ 4,  3,  2,  1,  2,  3,  4,  5,  6,  7,  8,  9,  8,  7,  6,  5],
 [ 5,  4,  3,  2,  1,  2,  3,  4,  5,  6,  7,  8,  9,  8,  7,  6],
 [ 6,  5,  4,  3,  2,  1,  2,  3,  4,  5,  6,  7,  8,  9,  8,  7],
 [ 7,  6,  5,  4,  3,  2,  1,  2,  3,  4,  5,  6,  7,  8,  9,  8],
 [ 8,  7,  6,  5,  4,  3,  2,  1,  2,  3,  4,  5,  6,  7,  8,  9],
 [ 9,  8,  7,  6,  5,  4,  3,  2,  1,  2,  3,  4,  5,  6,  7,  8],
 [ 8,  9,  8,  7,  6,  5,  4,  3,  2,  1,  2,  3,  4,  5,  6,  7],
 [ 7,  8,  9,  8,  7,  6,  5,  4,  3,  2,  1,  2,  3,  4,  5,  6],
 [ 6,  7,  8,  9,  8,  7,  6,  5,  4,  3,  2,  1,  2,  3,  4,  5],
 [ 5,  6,  7,  8,  9,  8,  7,  6,  5,  4,  3,  2,  1,  2,  3,  4],
 [ 4,  5,  6,  7,  8,  9,  8,  7,  6,  5,  4,  3,  2,  1,  2,  3],
 [ 3,  4,  5,  6,  7,  8,  9,  8,  7,  6,  5,  4,  3,  2,  1,  2],
 [ 2,  3,  4,  5,  6,  7,  8,  9,  8,  7,  6,  5,  4,  3,  2,  1]]).astype(np.float)

HOG_HIST_LEN = 36

HOG_8BIN_C = np.array(
[[ 1,  2,  3,  4,  5,  4,  3,  2,],
 [ 2,  1,  2,  3,  4,  5,  4,  3,],
 [ 3,  2,  1,  2,  3,  4,  5,  4,],
 [ 4,  3,  2,  1,  2,  3,  4,  5,],
 [ 5,  4,  3,  2,  1,  2,  3,  4,],
 [ 4,  5,  4,  3,  2,  1,  2,  3,],
 [ 3,  4,  5,  4,  3,  2,  1,  2,],
 [ 2,  3,  4,  5,  4,  3,  2,  1,]]).astype(np.float)

WEIGHTS_DICT = {
'RGB': 0.0,
'HSV': 0.7,
'CORNER':0.3,
'HOG':0.0
}

# FEATURES = [utils.CENTRE_YELLOW_FEATURE_ID]
# FEATURES = [utils.BORDER_GREEN_FEATURE_ID]
# FEATURES = [utils.SHARP_HOG_FEATURE_ID]
# FEATURES = [utils.CORNERNESS_FEATURE_ID]
# FEATURES = [utils.TOP_RIGHT_YELLOW_FEATURE_ID]
# FEATURES = [utils.BOTTOM_RIGHT_GREEN_FEATURE_ID]
FEATURES = [utils.TOP_LEFT_PURPLE_FEATURE_ID]
# FEATURES = [utils.BOTTOM_RIGHT_YELLOW_FEATURE_ID]
# FEATURES = [utils.DONUT_SHAPE_FEATURE_ID, utils.BOTTOM_RIGHT_NEIGHBOUR_BLUE_FEATURE_ID]
# FEATURES = [utils.BOTTOM_RIGHT_NEIGHBOUR_BLUE_FEATURE_ID]
# FEATURES = [utils.DONUT_SHAPE_FEATURE_ID]
"""
Routine to add a feature: 

1. Add FEATURE_ID string to 'utils.py' and 'FEATURES' array in this 'comparePatches.py'
2. Create corresponding feature object that inherits 'Feature'. 
   Add feature hist computer + feature score computer in the created feature object class
3. Add feature object to feature_arr in 'Patch' class

Eg: feature id: 'BOTTOM_RIGHT_GREEN', feature object class: 'FeatureBottomRightGreen'
"""

class Patch:
	def __init__(self, centreX, centreY, size, initialize_features = True): # rowIndex is x, colIndex is y

		self.x = centreX
		self.y = centreY
		self.size = size
		self.RGBHistArr = [] # RGB histograms array
		self.RGBHist = None #Full patch RGB Histogram
		self.RGBScore = None # individual RGB histogram distinguishability Score
		self.aggregateRGBScore = None # RGB histogram distinguishability Score over its aggregated neighbourhood

		self.HSVHistArr = [] # full patch + sub patches flattened HSV histogram
		self.HSVHist = None # full patch flattened HSV histogram
		self.HSVScore = None

		self.HueHist = None
		self.HueHistArr = [] # including the 2*2 sub patches
		self.SaturationHist = None
		self.SaturationHistArr = [] # including the 2*2 sub patches
		self.ValueHist = None
		self.ValueHistArr = [] # including the 2*2 sub patches

		self.cornerResponseScore = None # patch corner response score, TODO: check how to even it, currently, this score is distributed very uneven

		self.HOGArr = [] # includes HOG on full patch and HOGs on 4 subPatches
		self.HOG = None
		self.HOG_Uncirculated = None
		self.HOGScore = None

		self.overallScore = None

		# ### HOG Different Bin Features ###
		# self.HOG_BIN1 = None
		# self.HOG_BIN1Score = None
		# self.HOG_BIN2 = None
		# self.HOG_BIN2Score = None
		# self.HOG_BIN3 = None
		# self.HOG_BIN3Score = None
		# self.HOG_BIN4 = None
		# self.HOG_BIN4Score = None


		### Array of Feature objects###
		self.feature_arr = []
		if (initialize_features):
			# top right yellow
			self.feature_arr.append(feature_modules.FeatureTopRightYellow(self, utils.TOP_RIGHT_YELLOW_FEATURE_ID))
			# BOTTOM_RIGHT_GREEN: Feature length: 6 bins
			self.feature_arr.append(feature_modules.FeatureBottomRightGreen(self, utils.BOTTOM_RIGHT_GREEN_FEATURE_ID))
			# top left purple
			self.feature_arr.append(feature_modules.FeatureTopLeftPurple(self, utils.TOP_LEFT_PURPLE_FEATURE_ID))
			# Donut shape feature
			self.feature_arr.append(feature_modules.FeatureDonutShape(self, utils.DONUT_SHAPE_FEATURE_ID))
			# neighbour bottom right blue feature
			self.feature_arr.append(feature_modules.FeatureBottomRightNeighbourBlue(self, utils.BOTTOM_RIGHT_NEIGHBOUR_BLUE_FEATURE_ID))
			# bottom right yellow feature
			self.feature_arr.append(feature_modules.FeatureBottomRightYellow(self, utils.BOTTOM_RIGHT_YELLOW_FEATURE_ID))
			# cornerness feature
			# self.feature_arr.append(feature_modules.FeatureCornerness(self, utils.CORNERNESS_FEATURE_ID))
			# sharp HOG feature
			self.feature_arr.append(feature_modules.FeatureSharpHOG(self,utils.SHARP_HOG_FEATURE_ID))
			# border green feature
			self.feature_arr.append(feature_modules.FeatureBorderGreen(self, utils.BORDER_GREEN_FEATURE_ID))
			# center yellow feature
			self.feature_arr.append(feature_modules.FeatureCentreYellow(self, utils.CENTRE_YELLOW_FEATURE_ID))

		###For Algo3, a set of features to use for matching###
		self.feature_to_use = []
		self.feature_weights = None
		self.LDAFeatureScore = None # measure for the uniqueness of the feature sets

	def getFeatureObject(self, id):
		for obj in self.feature_arr:
			if (obj.id == id):
				return obj
		return None # if not found

	# def setHOG_BINScores(self, i, score):
	# 	setattr(self, "HOG_BIN{i}Score".format(i = i), score)

	def setFeatureWeights(self, weights_np_array):
		self.feature_weights = weights_np_array

	def setLDAFeatureScore(self, score):
		self.LDAFeatureScore = score

	def setFeatureToUse(self, features):
		del self.feature_to_use[:]
		self.feature_to_use.extend(features)

	def equals(self, another_patch):
		if(self.x == another_patch.x and self.y == another_patch.y and self.size == another_patch.size):
			return True
		else:
			return False

	def setX(self, _x):
		self.x = _x

	def setY(self, _y):
		self.y = _y

	def setSize(self,_size):
		self.size = _size

	def getSize(self):
		return self.size

	def setRGBHist(self,hist):
		self.RGBHist = hist

	def setHSVHist(self,hist):
		self.HSVHist = hist

	def setRGBScore(self,score):
		self.RGBScore = score

	def setHSVScore(self,score):
		self.HSVScore = score

	def setHOGScore(self, score):
		self.HOGScore = score

	# ### HOG_BINs feaures ###
	# def computeHOG_BINs(self, img, i, useGaussianSmoothing = True):
	# 	if(self.HOG_Uncirculated is None):
	# 		self.computeHOG(img, useGaussianSmoothing)
	# 	setattr(self, "HOG_BIN{i}".format(i = i), self.HOG_Uncirculated[(i-1)*9:i*9])

	def computeHOG(self, img, useGaussianSmoothing = True):
		"""
		HOG with orietation assignment and circular histogram
		img: gray scale
		Instead of compute the 2*2 subpatch HOG, compute a 1) sub circle 2) super circle HOG
		TODO: fine tune orientation, smooth the HOG hist + add more possible orietations (not just the maximum, 0.8 of the maximum as well) for considertaion in matching
		"""
		# Check if HOGArr is already computed
		if(len(self.HOGArr) > 0):
			return
		else:
			self.HOGArr = []

		gaussianSigma = self.size/6.0 # six sigma rule of thumb
		if(useGaussianSmoothing):
			gaussianWindow = gauss_kernels(self.size, gaussianSigma)
		else:
			gaussianWindow = np.ones(shape = (self.size, self.size))
		
		fullPatchHOG = self.computeSinglePatchHOG(img,gaussianWindow)
		self.HOG = fullPatchHOG

		# self.computeSubPatchHOG(img, gaussianWindow)
		# self.HOGArr.append(fullPatchHOG)
		self.computeSubCirclePatchHOG(img, gaussianWindow) # computes the 4 sub circle's HOG, from small to big
		self.HOGArr.append(fullPatchHOG) # append the full patch HOG
		return
	def computeSubCirclePatchHOG(self, img, gaussianWindow):
		"""
		From testing result, sub circle HOG is having a similar performance to SubAndSuperHOG with a slightly poorer performance
		"""
		# numberOfSubCircles = 2
		numberOfSubCircles = 4
		scale = 1.2
		# subCirclePatchs = []
		# for i in xrange(-numberOfSubCircles, numberOfSubCircles + 1, 1):
		# Do just sub patches, no super patches
		for i in xrange(-numberOfSubCircles, 0, 1):
			newSize = getGaussianScale(self.size, scale, i)
			if(self.x - newSize/2 >= 0 and self.x + newSize/2 < img.shape[0] and self.y - newSize/2 >=0 and self.y + newSize/2 < img.shape[1]):
				newSubCirclePatch = Patch(self.x, self.y, newSize)
				# print "new size{i}:".format(i = i), newSubCirclePatch.size
				newSubGaussianWindow = gauss_kernels(newSubCirclePatch.size, newSubCirclePatch.size/6.0)
				self.HOGArr.append(newSubCirclePatch.computeSinglePatchHOG(img, newSubGaussianWindow))
			else:
				raise ValueError("Super Patch circular out of range")

	def computeSubPatchHOG(self, img, gaussianWindow):
		newLen = (self.size+1)/2
		if(newLen % 2 == 0):
			newSize = newLen -1 # since size is supposed to be odd
		else:
			newSize = newLen
		
		top_left_gaussianWindow = gaussianWindow[0:newSize,0:newSize]
		top_right_gaussianWindow = gaussianWindow[0:newSize, gaussianWindow.shape[1] - newSize:gaussianWindow.shape[1]]
		bottom_left_gaussianWindow = gaussianWindow[gaussianWindow.shape[0] - newSize:gaussianWindow.shape[0], 0:newSize]
		bottom_right_gaussianWindow = gaussianWindow[gaussianWindow.shape[0] - newSize:gaussianWindow.shape[0], gaussianWindow.shape[1] - newSize: gaussianWindow.shape[1]]

		top_left_sub_patch = Patch(self.x - newLen/2, self.y - newLen/2, newSize)
		top_right_sub_patch = Patch(self.x - newLen/2, self.y + newLen/2, newSize)
		bottom_left_sub_patch = Patch(self.x + newLen/2, self.y - newLen/2, newSize)
		bottom_right_sub_patch = Patch(self.x + newLen/2, self.y + newLen/2, newSize)

		self.HOGArr.append(top_left_sub_patch.computeSinglePatchHOG(img, top_left_gaussianWindow))
		self.HOGArr.append(top_right_sub_patch.computeSinglePatchHOG(img, top_right_gaussianWindow))
		self.HOGArr.append(bottom_left_sub_patch.computeSinglePatchHOG(img, bottom_left_gaussianWindow))
		self.HOGArr.append(bottom_right_sub_patch.computeSinglePatchHOG(img, bottom_right_gaussianWindow))

	def computeSinglePatchHOG(self, img, gaussianWindow):
		ref_x = self.x - self.size/2
		ref_y = self.y - self.size/2

		# Get Orientation Assignment
		HOG_360_LEN = 360
		hist = np.zeros(HOG_360_LEN)
		bin_adjust_scale = HOG_360_LEN/2.0
		for i in range(self.x - self.size/2 + 1, self.x + self.size/2):
			for j in range(self.y - self.size/2 + 1, self.y + self.size/2):
				gx = float(img[i][j-1]) - float(img[i][j+1])
				gy = float(img[i-1][j]) - float(img[i+1][j])
				"""
				With this configuration, HOG's association with actual image shape:
				Initial positive axis of HOG will be towards up. 
				Clockwise swing must be from high value to low value; -> positive HOG degree
				Anti-clockwise swing must be from low value to high value; -> negative HOG degree
				"""

				mag = np.linalg.norm([gx,gy], 2)
				ori = math.atan2(gy, gx)

				HOG_bin = int(math.floor(ori*bin_adjust_scale/math.pi + bin_adjust_scale))
				HOG_bin = HOG_360_LEN - 1 if (HOG_bin == HOG_360_LEN) else HOG_bin

				hist[HOG_bin] += gaussianWindow[i - ref_x][j - ref_y] * mag

		self.HOG_Uncirculated = self.finalizeHOG(hist) # store the HOG_Uncirculated, it will be from -pi to pi, bin 0 corresponds to -pi
		max_ori = np.argmax(hist) # use maximum
		hist =  list(hist[max_ori:len(hist)]) + list(hist[0:max_ori]) # rotate circular hist

		return self.finalizeHOG(hist)

	def finalizeHOG(self,hist_360_bin):
		"""
		Aggregate the 360 Bin HOG to specified number of bins
		"""
		hist = np.zeros(HOG_HIST_LEN)
		scale = 360/HOG_HIST_LEN
		for i in range(0, len(hist)):
			hist[i] = np.sum(hist_360_bin[i*scale : i*scale + scale])
		return np.array(hist)
		# return normalize(np.array(hist), norm='l1')[0] 

	# TODO: refactor, make a computeColorHistogram as interface to outside, so that we can change the implementation inside willfully
	def computeHSVHistogram(self, img, useGaussianSmoothing = True, computeSeperateHists = False):
		# self.computeFlattenedHSVHistogram(img, useGaussianSmoothing, computeSeperateHists)
		self.computeSeperateHSVHistogram(img, useGaussianSmoothing)

	# compute the seperat H, S, V histograms overall and on the sub patches
	# self.HueHistArr, self.SaturationHistArr, self.ValueHistArr will be of size 5 each
	# TODO: decouple computeSeperateHSVHistogram from computeFlattenedHSVHistogram
	def computeSeperateHSVHistogram(self, img, useGaussianSmoothing = True):
		"""
		Here compute H,S,V channel, but V channel is left out during matching for illuminance invariance
		"""
		# if already computed during feature detection phase
		if(len(self.HueHistArr) == 5 and len(self.SaturationHistArr) == 5):
			return
		else:
			self.HueHistArr = []
			self.SaturationHistArr = []

		gaussianSigma = self.size/6.0 # six sigma rule of thumb
		if(useGaussianSmoothing):
			gaussianWindow = gauss_kernels(self.size, gaussianSigma)
		else:
			gaussianWindow = None
		self.computeSinglePatchHSVHistogram(img, gaussianWindow, True)
		self.computeSubPatchColorHistogram(img, "HSV", gaussianWindow, True)
		self.HueHist = self.HueHistArr[0]
		self.SaturationHist = self.SaturationHistArr[0]
		self.ValueHist = self.ValueHistArr[0]
		return

	def computeFlattenedHSVHistogram(self, img, useGaussianSmoothing, computeSeperateHists):
		if(len(self.HSVHistArr) == 5):
			return
		else:
			self.HSVHistArr = []

		gaussianSigma = self.size/6.0 # six sigma rule of thumb
		if(useGaussianSmoothing):
			gaussianWindow = gauss_kernels(self.size, gaussianSigma)
		else:
			gaussianWindow = None

		fullPatchHSVHist = self.computeSinglePatchHSVHistogram(img, gaussianWindow, computeSeperateHists)
		if(computeSeperateHists):
			self.HueHist = self.HueHistArr[0]
			self.SaturationHist = self.SaturationHistArr[0]
			self.ValueHist = self.ValueHistArr[0]

		self.HSVHist = fullPatchHSVHist
		self.HSVHistArr.append(fullPatchHSVHist)
		top_left_sub_patch, top_right_sub_patch, bottom_left_sub_patch, bottom_right_sub_patch, subHistArr = self.computeSubPatchColorHistogram(img, "HSV", gaussianWindow)
		# print "HSV subHistArr.len:", len(subHistArr)
		self.HSVHistArr = self.HSVHistArr + subHistArr
		return

	def RGBToHSV(self,R,G,B):
		R = R/255.0
		G = G/255.0
		B = B/255.0

		Cmax = max(R,G,B)
		Cmin = min(R,G,B)
		# set V
		V = Cmax

		delta = Cmax - Cmin
		#set S
		S = 0 if (Cmax == 0) else float(delta)/Cmax 

		#set H
		if(delta == 0):
			H = 0
		elif(Cmax == R):
			H = 60 * (((G-B)/delta) % 6)
		elif(Cmax == G):
			H = 60 * ((B-R)/delta + 2)
		elif(Cmax == B):
			H = 60 * ((R-G)/delta + 4)

		return H, S, V 

	# note that img[-2] will wrap around to be img[len-2]
	def computeSinglePatchHSVHistogram(self, img, gaussianWindow = None, computeSeperateHists = False, parentPatch = None):
		# If compute Gaussian Window on the sub patches as well:
		# gaussianSigma = self.size/6.0 # six sigma rule of thumb
		# gaussianWindow = gauss_kernels(self.size, gaussianSigma)

		ref_x = self.x - self.size/2
		ref_y = self.y - self.size/2
		# print "gaussianWindow:", gaussianWindow.shape


		bin_number = 16
		# hist = np.zeros(bin_number**3)
		"""
		16*16 = 256 flattened HS histogram
		"""
		hist = np.zeros(bin_number**2) # try 256 HS only and see how
		"""
		Check if need to computeSeperateHists of H, S, V channel
		"""
		if(computeSeperateHists):
			HueHist = np.zeros(bin_number)
			SaturationHist = np.zeros(bin_number)
			ValueHist = np.zeros(bin_number)
		
		H_bin_size = 360/float(bin_number)
		S_bin_size = 1/float(bin_number)
		V_bin_size = 1/float(bin_number)
		for i in range(self.x - self.size/2, self.x + self.size/2 + 1):
			for j in range(self.y - self.size/2, self.y + self.size/2 + 1):
				B = img[i][j][0]
				G = img[i][j][1]
				R = img[i][j][2]
				
				H, S, V = self.RGBToHSV(R, G, B)
				# print "H,S,V:", H, S, V
				h_bin = bin_number -1 if (H == 360) else int(math.floor(H/H_bin_size))
				s_bin = bin_number -1 if (S == 1) else int(math.floor(S/S_bin_size))
				v_bin = bin_number -1 if (V == 1) else int(math.floor(V/V_bin_size))
				# print "h_bin, s_bin,v_bin:", h_bin,s_bin,v_bin, "\n"
				if(gaussianWindow is None):
					# hist[h_bin * bin_number**2 + s_bin * bin_number + v_bin] += 1
					hist[h_bin * bin_number + s_bin ] += 1
				else:
					# hist[h_bin * bin_number**2 + s_bin * bin_number + v_bin] += gaussianWindow[i - ref_x][j - ref_y]
					hist[h_bin * bin_number + s_bin ] += gaussianWindow[i - ref_x][j - ref_y]

				if(computeSeperateHists):
					if(gaussianWindow is None):
						HueHist[h_bin] += 1
						SaturationHist[s_bin] += 1
						ValueHist[v_bin] += 1
					else:
						HueHist[h_bin] += gaussianWindow[i - ref_x][j - ref_y]
						SaturationHist[s_bin] += gaussianWindow[i - ref_x][j - ref_y]
						ValueHist[v_bin] += gaussianWindow[i - ref_x][j - ref_y]

				# print "smoothed weight added for pixel {i}_{j}".format(i = i, j = j), gaussianWindow[i - ref_x][j - ref_y]

		# print "sum of hist should be 1 (if individual gaussianWindow is applied for each subwidow as well):", np.sum(hist)
		if(computeSeperateHists and parentPatch != None):
			parentPatch.HueHistArr.append(HueHist)
			parentPatch.SaturationHistArr.append(SaturationHist)
			parentPatch.ValueHistArr.append(ValueHist)
		elif(computeSeperateHists):
			self.HueHistArr.append(HueHist)
			self.SaturationHistArr.append(SaturationHist)
			self.ValueHistArr.append(ValueHist)

		return hist
	

	def computeRGBHistogram(self, img):
		fullPatchRGBHist = self.computeSinglePatchRGBHistogram(img)
		self.RGBHist = fullPatchRGBHist
		self.RGBHistArr.append(fullPatchRGBHist)
		top_left_sub_patch, top_right_sub_patch, bottom_left_sub_patch, bottom_right_sub_patch, subHistArr = self.computeSubPatchColorHistogram(img)
		# print "RGB subHistArr.len:", len(subHistArr)
		self.RGBHistArr = self.RGBHistArr + subHistArr
		# print "length of self.RGBHistArr should b 5:", len(self.RGBHistArr)
	
	def computeSinglePatchRGBHistogram(self,img):
		bin_number = 4
		bin_size = 256/bin_number
		hist = np.zeros(bin_number**3)
		# print "hist.length:", len(hist)
		for i in range(self.x - self.size/2, self.x + self.size/2 + 1):
			for j in range(self.y - self.size/2, self.y + self.size/2 + 1):
				B = img[i][j][0]
				G = img[i][j][1]
				R = img[i][j][2]

				b_bin = B/bin_size
				g_bin = G/bin_size
				r_bin = R/bin_size

				hist[r_bin * bin_number**2 + g_bin * bin_number + b_bin] += 1

		return hist

	def computeSubPatchColorHistogram(self, img, histogramfunction = "RGB", gaussianWindow = None, computeSeperateHists = False):
		newLen = (self.size+1)/2
		if(newLen % 2 == 0):
			newSize = newLen -1 # since size is supposed to be odd
		else:
			newSize = newLen

		subHistArr = []
		
		if(gaussianWindow is None):
			top_left_gaussianWindow = None
			top_right_gaussianWindow = None
			bottom_left_gaussianWindow = None
			bottom_right_gaussianWindow = None
		else:
			top_left_gaussianWindow = gaussianWindow[0:newSize,0:newSize]
			top_right_gaussianWindow = gaussianWindow[0:newSize, gaussianWindow.shape[1] - newSize:gaussianWindow.shape[1]]
			bottom_left_gaussianWindow = gaussianWindow[gaussianWindow.shape[0] - newSize:gaussianWindow.shape[0], 0:newSize]
			bottom_right_gaussianWindow = gaussianWindow[gaussianWindow.shape[0] - newSize:gaussianWindow.shape[0], gaussianWindow.shape[1] - newSize: gaussianWindow.shape[1]]

		top_left_sub_patch = Patch(self.x - newLen/2, self.y - newLen/2, newSize)
		top_right_sub_patch = Patch(self.x - newLen/2, self.y + newLen/2, newSize)
		bottom_left_sub_patch = Patch(self.x + newLen/2, self.y - newLen/2, newSize)
		bottom_right_sub_patch = Patch(self.x + newLen/2, self.y + newLen/2, newSize)
		
		if(histogramfunction == "RGB"):
			subHistArr.append(top_left_sub_patch.computeSinglePatchRGBHistogram(img))
			subHistArr.append(top_right_sub_patch.computeSinglePatchRGBHistogram(img))
			subHistArr.append(bottom_left_sub_patch.computeSinglePatchRGBHistogram(img))
			subHistArr.append(bottom_right_sub_patch.computeSinglePatchRGBHistogram(img))
		elif(histogramfunction == "HSV"):
			subHistArr.append(top_left_sub_patch.computeSinglePatchHSVHistogram(img, top_left_gaussianWindow, computeSeperateHists, self))
			subHistArr.append(top_right_sub_patch.computeSinglePatchHSVHistogram (img,top_right_gaussianWindow, computeSeperateHists, self))
			subHistArr.append(bottom_left_sub_patch.computeSinglePatchHSVHistogram(img,bottom_left_gaussianWindow, computeSeperateHists, self))
			subHistArr.append(bottom_right_sub_patch.computeSinglePatchHSVHistogram(img,bottom_right_gaussianWindow, computeSeperateHists, self))

		return top_left_sub_patch, top_right_sub_patch, bottom_left_sub_patch, bottom_right_sub_patch, subHistArr
		
	def computeAggregateRGBScore(self, response):
		thresh = 200
		score = 0;
		for i in range(self.x - self.size/2, self.x + self.size/2 + 1):
			for j in range(self.y - self.size/2, self.y + self.size/2 + 1):
				if(response[i][j] >= thresh):
					score += response[i][j]

		self.aggregateRGBScore = score

	# assume cornerResponse is of the image original's shape and has the patches' score at patch centre position
	# response < 0 then it is categorized as edge, response around 0, then flat, response >> 0, good corner
	def setCornerResponseScore(self, cornerResponse, maxResponse, minResponse):
		# print "corner Response at (", self.x, ",", self.y,") is (!= 0): ", cornerResponse[self.x][self.y]
		if(cornerResponse[self.x][self.y] < 0):
			# normalizer = 1.0/abs(minResponse) * 0.09 # since 0.1 is considered usually as a corner, cap the edge value to be < 0.1
			# self.cornerResponseScore = float(abs(cornerResponse[self.x][self.y])) * normalizer

			# Edge is not a good feature
			self.cornerResponseScore = 0.0
		else:			
			normalizer = maxResponse
			self.cornerResponseScore = float(cornerResponse[self.x][self.y])/normalizer
	
	def setOverallScore(self):
		# if(self.cornerResponseScore > 0.01):
		# 	print "at (", self.x, ",", self.y,")"
		# 	print "self.RGBScore:", self.RGBScore
		# 	print "self.cornerResponseScore:", self.cornerResponseScore, "\n"
		overall_score = 0.0
		if(WEIGHTS_DICT['RGB'] != 0 and self.RGBScore != None):
			overall_score += self.RGBScore * WEIGHTS_DICT['RGB']
		if(WEIGHTS_DICT['HSV'] != 0 and self.HSVScore != None):
			overall_score += self.HSVScore * WEIGHTS_DICT['HSV']
		if(WEIGHTS_DICT['CORNER'] != 0 and self.cornerResponseScore != None):
			overall_score += self.cornerResponseScore * WEIGHTS_DICT['CORNER']
		if(WEIGHTS_DICT['HOG'] != 0 and self.HOGScore != None):
			overall_score += self.HOGScore * WEIGHTS_DICT['HOG']
		self.overallScore = overall_score

# get a new gaussian scale based on level and scale factor
def getGaussianScale(originalScale, factor, level):
	if(level < 0):
		newScale = int(originalScale / (factor ** abs(level)))
		newScale = newScale - 1 if (newScale % 2 == 0)  else newScale # make sure scale is odd
	else:
		newScale = int(originalScale * (factor ** abs(level)))
		newScale = newScale + 1  if (newScale % 2 == 0) else newScale # make sure scale is odd
	return newScale

def getDissimilairityHistArrl2(histArr1, histArr2, metricFunc):
	if(len(histArr1) != len(histArr2)):
		return 0.0
	individualScores = np.zeros(len(histArr1))
	for i in range(0, len(histArr1)):
		oneHistScore = metricFunc(histArr1[i], histArr2[i])
		individualScores[i] = oneHistScore
	return np.linalg.norm(individualScores, 2)

# get a guassian kernel of size * size
def gauss_kernels(size,sigma=1.0):
	if size<3:
		size = 3
	m = size/2
	x, y = np.mgrid[-m:m+1, -m:m+1]
	# print x*x + y*y
	# print (x*x + y*y)/(2*sigma*sigma)
	# print -(x*x + y*y)/(2*sigma*sigma)
	kernel = np.exp(-(x*x + y*y)/(2*sigma*sigma))
	# print kernel
	kernel_sum = kernel.sum()
	if not sum==0:
		kernel = kernel/kernel_sum 
	return kernel

def computePatchesRGBHistogram(img,patches):
	for i in range(0, len(patches)):
		patches[i].computeRGBHistogram(img)
		print "compute patch RGB:", i
	return

def computePatchesHSVHistogram(img,patches):
	for i in range(0, len(patches)):
		patches[i].computeHSVHistogram(img)
		print "compute patch HSV:", i
	return

def extractOneRandomPatch(img, sigma):
	x = random.randint(sigma/2, img.shape[0] - sigma/2 - 1) # since randint is inclusive of [a,b]
	y = random.randint(sigma/2, img.shape[1] - sigma/2 - 1)
	return Patch(x, y, sigma)

def alreadyInPatches(rand_patch, patches):
	"""
	return: True if rand_patch is already in patches; False Otherwise
	"""
	for i in range(0, len(patches)):
		if(rand_patch.equals(patches[i])):
			return True
	return False

def extractRandomPatches(img, sigma, num):
	"""
	img: image to extract patch on,
	sigma: patch window size,
	num: number of random patches generated
	"""
	patches = []
	while(len(patches)< num):
		rand_patch = extractOneRandomPatch(img, sigma)
		if(not alreadyInPatches(rand_patch, patches)):
			patches.append(rand_patch)
	return patches

# step 1 means shift by half of the window size, step 2 means shift by one window size, and so on, (circular_expand_level = 2 for SubAndSuperHOG)
def extractPatches(img, sigma, step, circular_expand_scale = 1.2, circular_expand_level = 0):
	print "Step for extract patch:", int(sigma/2*step)
	print img.shape[0]
	print img.shape[1]
	largest_patch_size = getGaussianScale(sigma, circular_expand_scale, circular_expand_level)
	print "largest_patch_size:", largest_patch_size
	patches = []
	# for patch_centre_row_index in np.arange(sigma/2, img.shape[0]- sigma/2,int(sigma/2*step)):
	# 	for patch_centre_col_index in np.arange(sigma/2, img.shape[1]- sigma/2, int(sigma/2*step)):
	for patch_centre_row_index in np.arange(largest_patch_size/2, img.shape[0]- largest_patch_size/2,int(sigma/2*step)):
		for patch_centre_col_index in np.arange(largest_patch_size/2, img.shape[1]- largest_patch_size/2, int(sigma/2*step)):
			# print "patch centre row index:", patch_centre_row_index, ";col index:", patch_centre_col_index 
			thisPatch = Patch(patch_centre_row_index, patch_centre_col_index, sigma)
			patches.append(thisPatch)
	return patches

def klDivergence(hist1, hist2):
	"""
	Note: There will be a runtime waring if sum(hist1) == 0 or sum(hist2) == 0, but does not affect result since it will return 'inf'
	"""
	return entropy(hist1,hist2)

def Jensen_Shannon_Divergence(hist1,hist2):
	# print "hist1:",hist1
	# print "hist2:",hist2
	mean = (hist1 + hist2) / 2
	dist = 0.5 * (klDivergence(hist1,mean) + klDivergence(hist2,mean))
	# print dist
	return dist

def CforHue(histLen):
	C = np.ones(shape = (histLen, histLen))
	for i in range(0, histLen):
		for j in range(i+1, histLen):
			C[i][j] += min(j-i, histLen-(j-i))
			C[j][i] = C[i][j]
	return C

def earthMoverHatDistanceForHue(hist1, hist2):
	"""
	HUE_16BIN_C only
	"""
	if(len(hist1) != 16 or len(hist2) != 16):
		raise ValueError("Length of histogram does not match Hue's Requirement")
	return pyemd.emd(hist1, hist2, HUE_16BIN_C)

def earthMoverHatDistanceForHOG(hist1, hist2):
	"""
	HOG_8BIN_C only
	"""
	if(len(hist1) != 8 or len(hist2) != 8):
		raise ValueError("Length of histogram does not match HOG Hist's Requirement")
	return pyemd.emd(hist1, hist2, HOG_8BIN_C)

def earthMoverHatDistance(hist1,hist2, C = None):
	# 1. pyemd EMD
	if(C is None):
		C = np.ones(shape = (len(hist1), len(hist2)))
		# for i in range(0, len(hist1)):
		# 	for j in range(i, len(hist2)):
		# 		C[i][j] += abs(i-j)
		# 		C[j][i] = C[i][j]
		rows = np.arange(0,len(hist1)).reshape((len(hist1), 1))
		rows = np.repeat(rows, len(hist2), axis = 1)
		cols = np.arange(0, len(hist2)).reshape(1,len(hist2))
		cols = np.repeat(cols, len(hist1), axis = 0)

		C = C + abs(rows - cols)
	# print C
	return pyemd.emd(hist1, hist2, C) # distance matrix needs C needs to be symmetric and float type; extra_mass_penalty used  = np.amax(C)

def Jensen_Shannon_Divergence_Score(row):
	"""
	For Jensen_Shannon_Divergence
	"""
	thresh = 0.5 # threshhold for indicatig large distinguishability for Jensen_Shannon_Divergence
	count = 0.0
	for i in range(0, len(row)):
		count += 0 if (row[i]<thresh) else row[i]
		# count += row[i]
	return count

def computeFullImageHSVHistogram(img):
	"""
	img: the given img portion to compute the full Hue, Saturation, Value Histograms on
	"""
	img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	img_hue_hist = cv2.calcHist([img_hsv],[0], None, [16], [0, 180])
	img_hue_hist = img_hue_hist[:,0]
	img_hue_hist = img_hue_hist/np.sum(img_hue_hist)

	img_saturation_hist = cv2.calcHist([img_hsv],[1], None, [16], [0,256])
	img_saturation_hist = img_saturation_hist[:,0]
	img_saturation_hist = img_saturation_hist/np.sum(img_saturation_hist)

	img_value_hist = cv2.calcHist([img_hsv],[2], None, [16], [0,256])
	img_value_hist = img_value_hist[:,0]
	img_value_hist = img_value_hist/np.sum(img_value_hist)

	return img_hue_hist, img_saturation_hist, img_value_hist

def compareSeperateHSVHists(patch, target_HueHist, target_SaturationHist, target_ValueHist, distancefunction = Jensen_Shannon_Divergence):
	hue_channel_distance = distancefunction(patch.HueHist, target_HueHist)
	saturation_channel_distance = distancefunction(patch.SaturationHist, target_SaturationHist)
	value_channel_distance = distancefunction(patch.ValueHist, target_ValueHist)
	return np.linalg.norm([hue_channel_distance, saturation_channel_distance], 2)

def HOGResponse(HOG):
	high_response_thresh = 0.5
	count = 0.0
	for i in range(0, len(HOG)):
		if(HOG[i] > high_response_thresh):
			# count += 1
			count += HOG[i]
	return count

def similarPatchAlreadySelected(patch, selected_patches, distance_thresh_dict, distancefunction = Jensen_Shannon_Divergence):
	"""
	need to be modified to coordinate with new features
	"""
	for i in range(0, len(selected_patches)):
		if(patch.feature_to_use == selected_patches[i].feature_to_use): # TODO: tackle the case where one set is the other's true subset, then, check score for both sets
			this_dist_score = 0
			dist_thresh = 0
			for j in range(0, len(patch.feature_to_use)):
				dist_thresh += distance_thresh_dict[patch.feature_to_use[j]]**2
				if(patch.feature_to_use[j] == 'HOG'):
					this_dist_score += distancefunction(selected_patches[i].HOG, patch.HOG)**2
				elif(patch.feature_to_use[j] == 'HSV'):
					this_dist_score += compareSeperateHSVHists(patch, \
						selected_patches[i].HueHist, selected_patches[i].SaturationHist, selected_patches[i].ValueHist, distancefunction)**2
			if(math.sqrt(this_dist_score) < math.sqrt(dist_thresh)):
				return True
	return False

def removeDuplicates(sorted_patches, distance_thresh_dict, distancefunction = Jensen_Shannon_Divergence):
	final_sorted_patches = []
	print "total_length of sorted_patches:", len(sorted_patches)

	final_sorted_patches.append(sorted_patches[0])
	i = 1
	while(i< len(sorted_patches)):
		if(not similarPatchAlreadySelected(sorted_patches[i], final_sorted_patches, distance_thresh_dict, distancefunction)):
			final_sorted_patches.append(sorted_patches[i])
		i += 1

	return final_sorted_patches

def findFeatureAttributeToUse(patches):
	feature_attribute_scores = {}
	# HSVScore score
	# print "In findFeatureAttributeToUse, normalized HSV distribution:\n", normalize([patch.HSVScore for patch in patches], norm='l1')
	feature_attribute_scores['HSVScore'] = np.std(normalize([patch.HSVScore for patch in patches], norm='l1')[0]) # l1/l2/max, l2 is default
	# HOGScore score
	# print "In findFeatureAttributeToUse, normalized HOG distribution:\n", normalize([patch.HOGScore for patch in patches], norm='l1')
	feature_attribute_scores['HOGScore'] = np.std(normalize([patch.HOGScore for patch in patches], norm='l1')[0])

	print "In findFeatureAttributeToUse, feature_attribute_scores:", feature_attribute_scores

	return max(feature_attribute_scores.iteritems(), key=operator.itemgetter(1))[0]

### Start of Algo2 for feature detection: Find one feature that makes the distribution of the low pass filtered patches to be of shape of spikes ###
def findDistinguishablePatchesAlgo2(img, sigma, remove_duplicate_thresh_dict, thresh_pass = 0.005, step = 1):
	"""
	thresh_pass: low threshold used for Harris Corner Filtering
	HSVthresh: used for removing similar patches for HSV unique patches
	HOGthresh: used for removing similar patches for HOG unique patches
	"""

	patches = extractPatches(img, sigma,step)
	"""
	Low Pass using Harris Corner to get inital set of potential good patches
	"""
	maxCornerResponse, cornerResponseMatrix = cornerResponse.getHarrisCornerResponse(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), sigma, step)
	filtered_patches = cornerResponse.filter_patches(patches, thresh_pass, cornerResponseMatrix, maxCornerResponse)
	# drawPatchesOnImg(np.copy(img), filtered_patches)

	"""
	Hue Saturation response: Jensen_Shannon_Divergence of patches compared to full image
	"""
	full_image_HueHist, full_image_SaturationHist, full_image_ValueHist = computeFullImageHSVHistogram(img)
	gaussianWindow = gauss_kernels(sigma, sigma/6.0)
	for i in range(0, len(filtered_patches)):
		filtered_patches[i].computeSinglePatchHSVHistogram(img, gaussianWindow, True)
		filtered_patches[i].HueHist = filtered_patches[i].HueHistArr[0]
		filtered_patches[i].SaturationHist = filtered_patches[i].SaturationHistArr[0]
		filtered_patches[i].ValueHist = filtered_patches[i].ValueHistArr[0]
		filtered_patches[i].setHSVScore(compareSeperateHSVHists(filtered_patches[i], full_image_HueHist, full_image_SaturationHist, full_image_ValueHist))
	
	"""
	HOG response
	"""
	img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.int)
	for i in range(0, len(filtered_patches)):
		filtered_patches[i].computeHOG(img_gray, True)
		filtered_patches[i].setHOGScore(HOGResponse(filtered_patches[i].HOG))

	feature_attr_to_use = findFeatureAttributeToUse(filtered_patches)
	# feature_attr_to_use = "HSVScore"
	# sorted_patches = sorted(filtered_patches, key = lambda patch: patch.HSVScore, reverse=True)
	# img = drawPatchesOnImg(img, removeDuplicates(sorted_patches, "HSV", HSVthresh)[0:10],False, None, (255,0,0)) # Blue for Uniqueness on HSV
	
	# sorted_patches = sorted(filtered_patches, key = lambda patch: patch.HOGScore, reverse=True)
	# img = drawPatchesOnImg(img, removeDuplicates(sorted_patches, "HOG", HOGthresh)[0:10],False, None, (0,0,255)) # Red for Uniqueness on HOG

	sorted_patches = sorted(filtered_patches, key = lambda patch: getattr(patch, feature_attr_to_use), reverse=True)
	print "check sorted patches score:"
	for i in range(0, len(sorted_patches)):
		# print sorted_patches[i].HSVScore
		sorted_patches[i].setFeatureToUse([feature_attr_to_use[0:feature_attr_to_use.find('Score')]])
		print getattr(sorted_patches[i], feature_attr_to_use)

	return removeDuplicates(sorted_patches,\
	remove_duplicate_thresh_dict),  feature_attr_to_use[0:feature_attr_to_use.find('Score')], filtered_patches # return sorted_patches using the most distinguishable attributes
	# return removeDuplicates(sorted_patches, "HOG", HOGthresh) # return sorted_patches using HOG
	# return filtered_patches

### Start of Algo3 for feature detection:
### 1. Low pass filter of Harris Corner score.
### 2. For each patch, find a combination of feature that makes it's LDA score high, remove from list if LDA score low for all combinations
def findDistinguishablePatchesAlgo3(img, sigma, remove_duplicate_thresh_dict , harris_thresh_pass = 0.0005, LDA_thresh = 1.0, step = 0.5):
	"""
	sigma, step: used for patch extraction
	harris_thresh_pass: threshhold for filtering the initial set of good patches
	"""
	patches = extractPatches(img, sigma, step)
	"""
	1. High Pass using Harris Corner to get inital set of potential good patches
	"""
	maxCornerResponse, cornerResponseMatrix = cornerResponse.getHarrisCornerResponse(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), sigma, step)
	filtered_patches = cornerResponse.filter_patches(patches, harris_thresh_pass, cornerResponseMatrix, maxCornerResponse)
	# positions = [(patch.x, patch.y) for patch in filtered_patches]
	# drawPatchesOnImg(np.copy(img), filtered_patches)
	"""
	2. Compute Combinatorial LDA score for each of the filtered patches (keep the set of best combination and its score + weights), remove from list if score too low
	"""
	findCombinatorialFeatureScore(img, filtered_patches, sigma, "", step)
	# plt.hist([this_patch.LDAFeatureScore for this_patch in filtered_patches], 50, label = "LDAFeatureScore Distribution for filtered_patches")
	# plt.legend()
	# plt.show()
	i = 0
	while(i < len(filtered_patches)):
		if(filtered_patches[i].LDAFeatureScore < LDA_thresh):
			filtered_patches.pop(i)
		else:
			i += 1
	"""
	3. remove duplicated patches that are very similar
	"""
	sorted_patches = sorted(filtered_patches, key = lambda patch: patch.LDAFeatureScore, reverse = True)

	# return removeDuplicates(sorted_patches,remove_duplicate_thresh_dict)
	return sorted_patches[0:20]

def LDAFeatureScore(this_feature_set, this_feature_weights, testPatch, random_patches, plotHist = False,  path = "", testPatchIndex = 0):
	"""
	this_feature_set: feature sets to consider
	this_feature_weights: weights of features in the set
	return: weighed LDA statistics of this testPatch and random_patches wrt the feature sets
	"""
	test_patch_response = 0
	for i in range(0, len(this_feature_set)):
		cur_feature_obj = testPatch.getFeatureObject(this_feature_set[i])
		cur_feature_weight = this_feature_weights[i]
		test_patch_response  += cur_feature_obj.score * cur_feature_weight

	random_patches_response = []
	for j in range(0, len(random_patches)):
		one_response = 0
		for i in range(0, len(this_feature_set)):
			cur_feature_obj = random_patches[j].getFeatureObject(this_feature_set[i])
			cur_feature_weight = this_feature_weights[i]
			one_response += cur_feature_obj.score * cur_feature_weight
		if(one_response >= 0 ): # do not consider if the response is negative, i.e., this patch not considered under that feature combination
			random_patches_response.append(one_response)
		
	# make the distribution to be np array
	random_patches_response = np.asarray(random_patches_response)

	print "random_patches_response mean:", np.mean(random_patches_response), ", random_patches_response var:", np.var(random_patches_response)
	print "test_patch_response:", test_patch_response

	# plot the distribution and the testPatch response
	if(plotHist):
		plotStatistics.plotResponseDistribution(path+"/hists", this_feature_set, testPatchIndex, test_patch_response, random_patches_response)

	# return (np.mean(random_patches_response) - test_patch_response)**2 / np.var(random_patches_response)
	if(test_patch_response < 0): # response is negative, indicating that this patch should not be considered under this combination
		return 0
	else:
		return (np.mean(random_patches_response) - test_patch_response)**2 / np.var(random_patches_response)

def generateAllFeatureSets(features):
	"""
	return: all subsets of a list of string
	"""
	all_sets = []
	for i in range(1, len(features)+1):
		sets_same_size = list(itertools.combinations(features, i))
		for j in range(0,len(sets_same_size)):
			all_sets.append(list(sets_same_size[j]))
	return all_sets


def setOnePatchScoreForAllFeatures(patch, img, img_gray, gaussianWindow, \
	full_image_HueHist, full_image_SaturationHist, full_image_ValueHist, \
	max_corner_response):
	# HSV Feature
	# patch.computeSinglePatchHSVHistogram(img, gaussianWindow, True)
	# patch.HueHist = patch.HueHistArr[0]
	# patch.SaturationHist = patch.SaturationHistArr[0]
	# patch.ValueHist = patch.ValueHistArr[0]
	patch.computeHSVHistogram(img)
	patch.setHSVScore(compareSeperateHSVHists(patch, full_image_HueHist, full_image_SaturationHist, full_image_ValueHist))

	# HOG Feature
	patch.computeHOG(img_gray, True)
	patch.setHOGScore(HOGResponse(patch.HOG))

	# compute feature and set score for each feature object in feature_arr
	for feature_obj in patch.feature_arr:
		if (isinstance(feature_obj, feature_modules.FeatureCornerness)):
			feature_obj.computeFeature(img, max_corner_response)
		else:
			feature_obj.computeFeature(img)
		feature_obj.computeScore()

	# # HOG Bins Features
	# for i in range(1, 5):
	# 	patch.computeHOG_BINs(img_gray, i, True)
	# 	patch.setHOG_BINScores(i,HOGResponse(getattr(patch, "HOG_BIN{i}".format(i = i))))

def findCombinatorialFeatureScore(img, testPatches, sigma, path = "", step = 0.5):
	"""
	img: the base image,
	testPatches: the set of unique patches;
	sigma: patch size
	path: path to save the LDA distribution, default is "", means do not save
	step: patch extraction and Harris Corner response step
	return: the score of different combination of features in LDA statistics 
	"""
	img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.int)
	gaussianWindow = gauss_kernels(sigma, sigma/6.0)
	full_image_HueHist, full_image_SaturationHist, full_image_ValueHist = computeFullImageHSVHistogram(img)
	max_corner_response, _ = cornerResponse.getHarrisCornerResponse(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), windowSize = sigma, _step = step)

	random_patches = extractRandomPatches(img, sigma, 500)
	print "FEATURES:", FEATURES

	for i in range(0, len(testPatches)):
		setOnePatchScoreForAllFeatures(testPatches[i], img, img_gray, gaussianWindow, \
			full_image_HueHist, full_image_SaturationHist, full_image_ValueHist, \
			max_corner_response)
	print "set score for all features for {count} testPatches done".format(count = len(testPatches))
	for i in range(0, len(random_patches)):
		setOnePatchScoreForAllFeatures(random_patches[i], img, img_gray, gaussianWindow, \
			full_image_HueHist, full_image_SaturationHist, full_image_ValueHist, \
			max_corner_response)
	print "set score for all features for {count} random_patches done".format(count = len(random_patches))


	all_feature_sets = generateAllFeatureSets(FEATURES)
	all_feature_set_weights = []
	feature_sets_score = np.zeros(shape = (len(all_feature_sets), len(testPatches)))
	
	for i in range(0, len(all_feature_sets)):
		this_feature_set = all_feature_sets[i]
		this_feature_weights = np.ones(len(this_feature_set)) # TODO: may need to adjust the weight based what features there are in the set
		all_feature_set_weights.append(this_feature_weights)
		print "checking score for set: ", this_feature_set, "with weights: ", this_feature_weights
		# TODO: increase efficiency: for each set, the distribution of random patches are all the same, no need to calculate for each testPatch[j]
		for j in range(0, len(testPatches)):
			feature_sets_score[i][j] = LDAFeatureScore(this_feature_set, this_feature_weights, testPatches[j], random_patches, path != "", path, j) # only save hist if path is there

	# set patch feature_to_use and LDAFeatureScore
	for i in range(0, len(testPatches)):
		this_patch_scores = feature_sets_score[:,i]
		this_patch_max_score_index = np.where(this_patch_scores == max(this_patch_scores))[0][0] # find the index of feature combination that corresponds to the max score
		testPatches[i].setLDAFeatureScore(this_patch_scores[this_patch_max_score_index])
		testPatches[i].setFeatureToUse(all_feature_sets[this_patch_max_score_index])
		testPatches[i].setFeatureWeights(all_feature_set_weights[this_patch_max_score_index])

	# Log out the feature_sets_score for each testPatch
	print "------------ Logging feature_sets_score for each testPatch ------------"
	for i in range(0, len(testPatches)):
		for j in range(0, len(all_feature_sets)):
			print "testPatch[{i}] ".format(i = i), all_feature_sets[j], " Score: ", feature_sets_score[j][i]
		print ""
	return feature_sets_score



def drawPatchesOnImg(img, patches, show = True, gradiant = None, color = (0,0,255), mark_sequence = False): 
	"""
	patches can be 1. an instance of Patch class 2. A list of patches
	Gradiant is to indiacte the goodness of the match patch, the ligher(redder) the better
	Gradiant is supposed to be  = 1.0/len(patches)
	"""
	if(type(patches).__name__ == "instance"):
		p = patches
		cv2.rectangle(img,(p.y-p.size/2,p.x-p.size/2),(p.y+p.size/2,p.x+p.size/2),color,1) # np.random.randint(0,255,size = 3)
	elif(type(patches) is list):	
		for i in range(0, len(patches)):
			p = patches[i]
			if(mark_sequence):
				cv2.putText(img, "{i}".format(i =i), (patches[i].y, patches[i].x), cv2.FONT_HERSHEY_SIMPLEX, 0.5 ,(0,0,255),1)
			if(gradiant is None):
				cv2.rectangle(img,(p.y-p.size/2,p.x-p.size/2),(p.y+p.size/2,p.x+p.size/2),color,1)
			else:
				cv2.rectangle(img,(p.y-p.size/2,p.x-p.size/2),(p.y+p.size/2,p.x+p.size/2),(color[0]*(1 - gradiant * i),color[1]*(1 - gradiant * i),color[2]*(1 - gradiant * i)),1) # np.random.randint(0,255,size = 3)
	
	if(show):	
		cv2.imshow("draw Patches On Original Image",img)
		cv2.waitKey(0)
	return img

def drawMatchesOnImg(img, imgToMatch, patches, matches, show = True):
	drawPatchesOnImg(img, patches,show = False, mark_sequence = True)
	drawPatchesOnImg(imgToMatch, matches, show = False, mark_sequence = True)

	patch_key_points = []
	match_key_points = []
	for i in range(0, len(patches)):
		patch_key_points.append(cv2.KeyPoint(patches[i].y, patches[i].x, patches[i].size))
	for i in range(0, len(matches)):
		match_key_points.append(cv2.KeyPoint(matches[i].y, matches[i].x, matches[i].size))
	match_indexes = []
	for i in range(0, len(patches)):
		match_indexes.append(cv2.DMatch(i,i,i)) # since patch_key_points[i] -> match_key_points[match_indexes[i]], here patch_key_points[i] -> match_key_points[i]
	matched_img = drawMatches.drawMatches(img, patch_key_points, imgToMatch, match_key_points, match_indexes)
	if(show):
		cv2.imshow("matched_img", matched_img)
		cv2.waitKey(0)
	return matched_img


def populateTestFindDistinguishablePatchesAlgo2(folderName, imgName, sigma):
	img = cv2.imread("images/{folder}/{name}".format(folder = folderName,  name = imgName), 1)
	HSVthresh = 0.5
	HOGthresh = 0.1
	normalize_approach = "l1"
	sorted_patches, feature_to_use, all_filtered_patches = findDistinguishablePatchesAlgo2(img, sigma, {'HSV': HSVthresh, 'HOG':HOGthresh})
	print "End of Find distinguishable patches, feature_to_use:", feature_to_use
	plotStatistics.plotUniquenessDistribution("testUniquePatches/graphs", \
		"HSV_distribution_{folderName}_{imgName}{normalized}".format(folderName = folderName, imgName = imgName[0:imgName.find(".")], normalized = "" if (normalize_approach == "") else "_normalized" + normalize_approach), \
		all_filtered_patches, "HSV", normalize_approach)
	plotStatistics.plotUniquenessDistribution("testUniquePatches/graphs", \
		"HOG_distribution_{folderName}_{imgName}{normalized}".format(folderName = folderName, imgName = imgName[0:imgName.find(".")], normalized = "" if (normalize_approach == "") else "_normalized" + normalize_approach), \
		all_filtered_patches, "HOG", normalize_approach)
	# cv2.imshow("after the process, img:", drawPatchesOnImg(img, sorted_patches,False, None))
	# cv2.waitKey(0)
	# cv2.imwrite("testUniquePatches/UniquePatches_HSVthresh_{HSVthresh}_HOGthresh_{HOGthresh}_{folder}_{img}_sigma{i}.jpg".format(folder = folderName, i = sigma, img = imgName[0:imgName.find(".")], HSVthresh = HSVthresh, HOGthresh = HOGthresh), img)

def populateTestFindDistinguishablePatchesAlgo3(test_folder_name, img_name, sigma = 39, image_db = "images", custom_feature_sets = None):
	# path = matchPatches.createFolder(upperPath, "GaussianWindowOnAWhole", test_folder_name, suffix)
	HSVthresh = 0.5
	HOGthresh = 0.1
	remove_duplicate_thresh_dict ={
		'HSV': HSVthresh,
		'HOG': HOGthresh
	}
	if(not custom_feature_sets is None):
		print "update features!"
		global FEATURES # need global marker to reassign the global variable
		FEATURES = custom_feature_sets

	img = cv2.imread("{image_db}/{folder}/{name}".format(image_db = image_db, folder = test_folder_name,  name = img_name), 1)
	sorted_patches = findDistinguishablePatchesAlgo3(img, sigma, remove_duplicate_thresh_dict)
	for i in range(0, len(sorted_patches)):
		print "feature_to_use for sorted_patches[{i}]: ".format(i = i), sorted_patches[i].feature_to_use
		print "feature_weights for sorted_patches[{i}]: ".format(i = i), sorted_patches[i].feature_weights
		print "LDAFeatureScore:", sorted_patches[i].LDAFeatureScore
		print "Actual TOP_RIGHT_YELLOW Score:", sorted_patches[i].getFeatureObject(utils.TOP_RIGHT_YELLOW_FEATURE_ID).score
		print "" # spacing
	result = drawPatchesOnImg(np.copy(img), sorted_patches, True, None, (0,0,255), True)
	cv2.imwrite("testUniquePatches/algo3/UniquePatches_{features}_{folder}_{file}_simga{i}_GaussianWindowOnAWhole.jpg".format(\
		features = "_".join(FEATURES), \
		folder = test_folder_name, \
		file = img_name[:img_name.find(".")], \
		i = sigma), result)
	saveLoadPatch.savePatchMatches(sorted_patches, 1, \
		"{path}/UniquePatches_{features}_{folder}_{file}_simga{i}_GaussianWindowOnAWhole.csv".format( \
			features = "_".join(FEATURES), \
			path = "testUniquePatches/algo3" , \
			folder = test_folder_name, \
			file = img_name[:img_name.find(".")], \
			i = sigma))

def populateCheckUniquePatchesAlgo3(test_folder_name, img_name, sigma = 39, image_db = "images"):
	img = cv2.imread("{image_db}/{folder}/{name}".format(image_db = image_db, folder = test_folder_name,  name = img_name), 1)
	unique_patches = []
	list_of_patches = saveLoadPatch.loadPatchMatches("{path}/UniquePatches_{features}_{folder}_{file}_simga{i}_GaussianWindowOnAWhole.csv".format( \
		features = "_".join(FEATURES), \
		path = "testUniquePatches/algo3" , \
		folder = test_folder_name, \
		file = img_name[:img_name.find(".")], \
		i = sigma))
	for i in range(0, 2): # just append the best one found
		unique_patches.append(list_of_patches[i][0])
	drawPatchesOnImg(np.copy(img), unique_patches, True, None, (0,0,255), True)
	for i in range(0, len(unique_patches)):
		print "checking unique_patches[{i}]".format(i = i)
		for this_feature in FEATURES:
			unique_patches[i].getFeatureObject(this_feature).computeFeature(img)
			unique_patches[i].getFeatureObject(this_feature).computeScore()

			print "unique_patches[{i}] ".format(i = i), \
			", actual {this_feature} Score:".format(this_feature = this_feature), \
			unique_patches[i].getFeatureObject(this_feature).score

			plotStatistics.plotOneGivenHist( \
				"", \
				"{this_feature} unique_patches[{i}]".format(i = i, this_feature = this_feature), \
				unique_patches[i].getFeatureObject(this_feature).hist, \
				save = False, \
				show = True)

def populateTestCombinatorialFeatureScore( \
	test_folder_name, \
	img_name, \
	sigma = 39, \
	upperPath = "testAlgo3", \
	folder_suffix = "_eyeballed_unique_patches", \
	image_db = "images"):

	path = upperPath + "/GaussianWindowOnAWhole/" + test_folder_name + folder_suffix
	img = cv2.imread("{image_db}/{folder}/{name}".format(image_db = image_db, folder = test_folder_name,  name = img_name), 1)
	# turn on if need to write as black and white
	cv2.imwrite("{path}/{folder}_{file}_simga{i}_{suffix}_bw.jpg".format( \
			path = path , \
			folder = test_folder_name, \
			file = "test1", \
			i = sigma, \
			suffix = folder_suffix), cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.uint8))

	testPatches = []
	listOfPatches = saveLoadPatch.loadPatchMatches( \
		"{path}/DistinguishablePatch_{folder}_{file}_simga{i}_GaussianWindowOnAWhole.csv".format( \
			path = path , \
			folder = test_folder_name, \
			file = "test1", \
			i = sigma))
	for i in range(0, len(listOfPatches)):
		testPatches.append(listOfPatches[i][0]) # just append the best match

	drawPatchesOnImg(np.copy(img), testPatches, True, None, (0,0,255), True)
	max_corner_response, _ = cornerResponse.getHarrisCornerResponse(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), windowSize = sigma, _step = 0.5)

	for i in range(0, len(testPatches)):
		print "\ntestPatches[{i}]:".format(i = i)
		for this_feature in FEATURES:
			testPatches[i].getFeatureObject(this_feature).computeFeature(img)
			plotStatistics.plotOneGivenHist(\
				path, \
				"testPatches[{i}] feature {this_feature}".format(i = i, this_feature = this_feature), \
				testPatches[i].getFeatureObject(this_feature).hist, \
				save = False, \
				show = True)
		

	# 	plotStatistics.plotColorHistogram(testPatches[i], img, path+"/hists", "unique_patch[{i}]".format(i = i), save = True, show = True, histToUse = "HSV", useGaussian = True)
	feature_set_scores = findCombinatorialFeatureScore(img, testPatches, sigma, path)
	print feature_set_scores

def main():
	folderNames = ["testset_illuminance1"]
	# folderNames = ["testset_rotation1"]
	### Test Algo2 in finding distinguishable patches ###
	# for i in range(0, len(folderNames)):
	# 	populateTestFindDistinguishablePatchesAlgo2(folderNames[i], "test1.jpg", 39)
	# raise ValueError ("stop for test findDistinguishablePatchesAlgo2")

	## Test combinatorial feature scores on a set of eyeballed patches
	# for i in range(0, len(folderNames)):
	# 	populateTestCombinatorialFeatureScore(folderNames[i], "test1.jpg",39, \
	# 		upperPath = "testAlgo3", \
	# 		folder_suffix = "_eyeballed_unique_patches")
	# raise ValueError("purpose stop for testing populating combinatorial score")

	### Test Algo3 in finding distinguishable patches ###
	start_time = time.time()
	for i in range(0, len(folderNames)):
		populateTestFindDistinguishablePatchesAlgo3(folderNames[i], "test1.jpg", 39)
		# populateCheckUniquePatchesAlgo3(folderNames[i], "test1.jpg", 39)
	print "finished feature extraction in ", time.time() - start_time, "seconds"
	return


if __name__ == "__main__":
	main()