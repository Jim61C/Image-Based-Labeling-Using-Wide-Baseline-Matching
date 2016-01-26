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

class Patch:
	def __init__(self, centreX, centreY, size): # rowIndex is x, colIndex is y

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
		self.HOGScore = None

		self.overallScore = None

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

	def clearHistograms(self):
		self.RGBHistArr = [] 
		self.RGBHist = None 
		self.RGBScore = None
		self.aggregateRGBScore = None

		self.HSVHistArr = []
		self.HSVHist = None
		self.HSVScore = None

		self.HueHist = None
		self.HueHistArr = []
		self.SaturationHist = None
		self.SaturationHistArr = []
		self.ValueHist = None
		self.ValueHistArr = []

		self.cornerResponseScore = None

		self.HOGArr = []
		self.HOG = None
		self.HOGScore = None

		self.overallScore = None

	# img should be in gray scale
	# instead of compute the 2*2 subpatch HOG, compute a 1) sub circle 2)super circle HOG
	def computeHOG(self, img, useGaussianSmoothing = True):
		"""
		HOG with orietation assignment and circular histogram
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

		# print "self.HOG:", self.HOG
		# print "self.HOGArr len:", len(self.HOGArr)
		# print "self.HOGArr:", self.HOGArr
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
				gx = float(img[i][j+1] - img[i][j-1])
				gy = float(img[i-1][j] - img[i+1][j])

				mag = np.linalg.norm([gx,gy], 2)
				ori = math.atan2(gy, gx)

				HOG_bin = int(math.floor(ori*bin_adjust_scale/math.pi + bin_adjust_scale))
				HOG_bin = HOG_360_LEN - 1 if (HOG_bin == HOG_360_LEN) else HOG_bin

				hist[HOG_bin] += gaussianWindow[i - ref_x][j - ref_y] * mag

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
		# print "len(self.HueHistArr):", len(self.HueHistArr)
		# print "len(self.SaturationHistArr):", len(self.SaturationHistArr)
		# print "len(self.ValueHistArr):", len(self.ValueHistArr)

		# print "self.HueHist:", self.HueHist
		# print "self.SaturationHist:", self.SaturationHist
		# print "self.ValueHist:", self.ValueHist
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
		



	# TODO: Try Gaussian Smoothing on Response on this patch of width 'size'
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

def computeDissimilarityMatrix(img, patches, distancefunction, metric = "RGB"):
	dissimilarity = np.zeros(shape = (len(patches),len(patches)))
	for i in range(0, len(patches)):
		dissimilarity[i][i] = 0
		for j in range(i+1, len(patches)):
			if(metric == "RGB"):
				dissimilarity[i][j] = distancefunction(patches[i].RGBHist, patches[j].RGBHist)
			elif(metric == "HSV"):
				# dissimilarity[i][j] = distancefunction(patches[i].HSVHist, patches[j].HSVHist)

				# check if use flattened HSV Histogram Arr 
				if(len(patches[i].HSVHistArr) > 0 and len(patches[j].HSVHistArr) > 0):
					dissimilarity[i][j] = getDissimilairityHistArrl2(patches[i].HSVHistArr, patches[j].HSVHistArr, distancefunction)
				else:
					hue_channel_distance = getDissimilairityHistArrl2(patches[i].HueHistArr, patches[j].HueHistArr,distancefunction)
					saturation_channel_distance = getDissimilairityHistArrl2(patches[i].SaturationHistArr, patches[j].SaturationHistArr,distancefunction)
					# value_channel_distance = getDissimilairityHistArrl2(patches[i].ValueHistArr, patches[j].ValueHistArr,distancefunction)
					dissimilarity[i][j] = np.linalg.norm([hue_channel_distance, saturation_channel_distance], 2)
			elif(metric == "HOG"):
				dissimilarity[i][j] = getDissimilairityHistArrl2(patches[i].HOGArr, patches[j].HOGArr, distancefunction)
			dissimilarity[j][i] = dissimilarity[i][j]
	return dissimilarity

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

"""Note: There will be a runtime waring if sum(hist1) == 0 || sum(hist2) == 0 """
def klDivergence(hist1, hist2):
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
	
	# 2. opencv EMD
	# sig1 = cv.CreateMat(len(hist1), 2, cv.CV_32FC1)
	# sig2 = cv.CreateMat(len(hist2), 2, cv.CV_32FC1) 
	# for i in range(0, len(hist1)):
	# 	cv.Set2D(sig1, i, 0, cv.Scalar(hist1[i]))
	# 	cv.Set2D(sig1, i, 1, cv.Scalar(i))

	# 	cv.Set2D(sig2, i, 0, cv.Scalar(hist2[i]))
	# 	cv.Set2D(sig2, i, 1, cv.Scalar(i))		

	# # mat1 = cv.fromarray(np.reshape(hist1, (len(hist1),1)).astype(np.float32))
	# # mat2 = cv.fromarray(np.reshape(hist2, (len(hist2),1)).astype(np.float32))
	# # print "mat1.type:", mat1.type
	# # print "cv.fromarray(C).type:", cv.fromarray(C.astype(np.float32)).type
	# # return cv.CalcEMD2(mat1,mat2,cv.CV_DIST_USER, distance_func = None, cost_matrix = cv.fromarray(C.astype(np.float32)))
	# return cv.CalcEMD2(sig1,sig2,cv.CV_DIST_L1)

	# # 3. opencv EMD 2D for H, S case
	# sig1 = cv.CreateMat(len(hist1), 3, cv.CV_32FC1)
	# sig2 = cv.CreateMat(len(hist2), 3, cv.CV_32FC1) 
	# for h in range(0, int(math.sqrt(len(hist1)))):
	# 	for s in range(0, int(math.sqrt(len(hist1)))):
	# 		row_index = int(h * math.sqrt(len(hist1)) + s)
	# 		cv.Set2D(sig1, row_index, 0,  cv.Scalar(hist1[row_index]))
	# 		cv.Set2D(sig1, row_index, 1,  cv.Scalar(h))
	# 		cv.Set2D(sig1, row_index, 2,  cv.Scalar(s))

	# 		cv.Set2D(sig2, row_index, 0,  cv.Scalar(hist2[row_index]))
	# 		cv.Set2D(sig2, row_index, 1,  cv.Scalar(h))
	# 		cv.Set2D(sig2, row_index, 2,  cv.Scalar(s))


	# # mat1 = cv.fromarray(np.reshape(hist1, (len(hist1),1)).astype(np.float32))
	# # mat2 = cv.fromarray(np.reshape(hist2, (len(hist2),1)).astype(np.float32))
	# # print "mat1.type:", mat1.type
	# # print "cv.fromarray(C).type:", cv.fromarray(C.astype(np.float32)).type
	# # return cv.CalcEMD2(mat1,mat2,cv.CV_DIST_USER, distance_func = None, cost_matrix = cv.fromarray(C.astype(np.float32)))
	# return cv.CalcEMD2(sig1,sig2,cv.CV_DIST_L1)

# Does not work well for Histograms here, Discard!!!
def chiSquareDistance(hist1, hist2):
	# mat1 = cv.fromarray(np.reshape(hist1, (len(hist1),1)).astype(np.float32))
	# mat2 = cv.fromarray(np.reshape(hist2, (len(hist2),1)).astype(np.float32))
	return cv2.compareHist(hist1.astype(np.float32), hist2.astype(np.float32), cv.CV_COMP_CHISQR)

def dissimilarityMetricOnImagePosition(img, patches, dissimilarity):
	response = np.zeros(shape = (img.shape[0], img.shape[1]))
	for i in range(0, len(patches)):
		if(patches[i].RGBScore is None):
			score = distinguishabilityScore(dissimilarity[i])
			patches[i].setRGBScore(score)
		for row in range(patches[i].x - patches[i].size/2, patches[i].x + patches[i].size/2): # only top and left border count to the response, avoid over count at the borders
			for col in range(patches[i].y - patches[i].size/2, patches[i].y + patches[i].size/2):
				response[row][col] += patches[i].RGBScore

	# plt.pcolor(np.arange(0,response.shape[1],1), np.arange(0, response.shape[0],1), response, cmap = plt.get_cmap('gray')) # for the scale from [0:255], use: plt.imshow(image_matrix[0][0], cmap = plt.get_cmap('gray'), vmin = 0, vmax = 255)
	# plt.gca().invert_yaxis()
	# plt.colorbar()
	# plt.show()
	# plt.clf()
	return response

def distinguishabilityScore(row):
	"""
	For Jensen_Shannon_Divergence
	"""
	thresh = 0.5 # threshhold for indicatig large distinguishability for Jensen_Shannon_Divergence
	count = 0.0
	for i in range(0, len(row)):
		count += 0 if (row[i]<thresh) else row[i]
		# count += row[i]
	return count

# dissimilarity is n*n array
# return a sorted array of indexes of patches based on the metirc: sum/average of dissimilarity with all rest patches
def sortedIndexOfDistinguishablePatches(patches, dissimilarity, metric = "RGB"):
	normalizer = float(dissimilarity.shape[0]) * np.amax(dissimilarity)
	distinguishability = np.zeros(dissimilarity.shape[0])
	for i in range(0, len(distinguishability)):
		score = distinguishabilityScore(dissimilarity[i])
		distinguishability[i] = score/normalizer
		if(metric == "RGB"):
			patches[i].setRGBScore(score)
		elif(metric == "HSV"):
			patches[i].setHSVScore(score)
	return np.argsort(distinguishability)[::-1]

### Start of Algo3 for feature detection: 1. Low pass filter of Harris Corner score. 2. For each patch, find a combination of feature that makes it's LDA score high, remove from list if LDA score low for all combinations###
def findDistinguishablePatchesAlgo3(img, sigma, harris_thresh_pass = 0.005, step = 1):
	"""
	sigma, step: used for patch extraction
	harris_thresh_pass: threshhold for filtering the initial set of good patches
	"""

	patches = extractPatches(img, sigma,step)
	"""
	1. Low Pass using Harris Corner to get inital set of potential good patches
	"""
	maxCornerResponse, cornerResponseMatrix = cornerResponse.getHarrisCornerResponse(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), sigma, step)
	filtered_patches = cornerResponse.filter_patches(patches, thresh_pass, cornerResponseMatrix, maxCornerResponse)
	"""
	2. Compute Combinatorial LDA score for each of the filtered patches (keep the set of best combination and its score + weights), remove from list if score too low
	"""

	return

### Start of Algo2 for feature detection: Find one feature that makes the distribution of the low pass filtered patches to be of shape of spikes###

def computeFullImageHSVHistogram(img):
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
	high_response_thresh = 3.0
	count = 0.0
	for i in range(0, len(HOG)):
		if(HOG[i] > high_response_thresh):
			# count += 1
			count += HOG[i]
	return count

def similarPatchAlreadySelected(patch, selected_patches, metric, distance_thresh, distancefunction = Jensen_Shannon_Divergence):
	if(metric == "HOG"):
		for i in range(0, len(selected_patches)):
			if(distancefunction(selected_patches[i].HOG, patch.HOG) < distance_thresh):
				return True
		return False
	elif(metric == "HSV"):
		for i in range(0, len(selected_patches)):
			if(compareSeperateHSVHists(patch, \
				selected_patches[i].HueHist, selected_patches[i].SaturationHist, selected_patches[i].ValueHist, distancefunction) < distance_thresh):
				return True
		return False

def removeDuplicates(sorted_patches, metric, distance_thresh, distancefunction = Jensen_Shannon_Divergence):
	final_sorted_patches = []
	print "total_length of sorted_patches:", len(sorted_patches)

	final_sorted_patches.append(sorted_patches[0])
	if(metric == "HOG"):
		i = 1
		while(i< len(sorted_patches)):
			if(not similarPatchAlreadySelected(sorted_patches[i], final_sorted_patches, "HOG", distance_thresh)):
				final_sorted_patches.append(sorted_patches[i])
			i += 1
	elif(metric == "HSV"):
		i = 1
		while(i< len(sorted_patches)):
			if(not similarPatchAlreadySelected(sorted_patches[i], final_sorted_patches, "HSV", distance_thresh)):
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


# def checkDistance(sorted_patches, metric, distancefunction = Jensen_Shannon_Divergence):
# 	if(metric == "HSV"):
# 		for i in range(0, len(sorted_patches) - 1):
# 			HueDist = distancefunction(sorted_patches[i].HueHist, sorted_patches[i+1].HueHist)
# 			SaturationDist = distancefunction(sorted_patches[i].SaturationHist, sorted_patches[i+1].SaturationHist)
# 			print "distance between patch ", i ," and patch ", (i+1), " = ", np.linalg.norm([HueDist, SaturationDist], 2)

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
		print getattr(sorted_patches[i], feature_attr_to_use)

	return removeDuplicates(sorted_patches, \
	feature_attr_to_use[0:feature_attr_to_use.find('Score')], \
	remove_duplicate_thresh_dict[feature_attr_to_use[0:feature_attr_to_use.find('Score')]]),  feature_attr_to_use[0:feature_attr_to_use.find('Score')], filtered_patches # return sorted_patches using the most distinguishable attributes
	# return removeDuplicates(sorted_patches, "HOG", HOGthresh) # return sorted_patches using HOG
	# return filtered_patches



def findDistinguishablePatches(img, sigma, step = 1):
	patches = extractPatches(img, sigma,step)
	print "number of patches:", len(patches)

	"""
	---------HSV Uniqueness Score, it is tested that computeFlattenedHSVHistogram performs better than computeSeperateHSVHistogram-------------
	"""
	if(WEIGHTS_DICT['HSV'] > 0):
		# computePatchesRGBHistogram(img,patches)
		computePatchesHSVHistogram(img,patches)
		dissmilarityMatrix = computeDissimilarityMatrix(img, patches, Jensen_Shannon_Divergence, "HSV")
		print "dissmilarityMatrix shape:", dissmilarityMatrix.shape 
		print "np.amin(dissmilarityMatrix):", np.amin(dissmilarityMatrix)
		print "np.amax(dissmilarityMatrix)", np.amax(dissmilarityMatrix)
		# plot the heat map of the dissimilarity matrix
		# heatmap = plt.pcolor(dissmilarityMatrix,  vmin=np.amin(dissmilarityMatrix), vmax=np.amax(dissmilarityMatrix))
		# plt.savefig("dissimilarityHeatMap_{img}.png".format(img = imgName[0:imgName.find(".")]))
		# plt.show()
		# plt.clf()
		normalizer = float(dissmilarityMatrix.shape[0]) * np.amax(dissmilarityMatrix)
		for i in range(0, dissmilarityMatrix.shape[0]):
			score = distinguishabilityScore(dissmilarityMatrix[i])
			# patches[i].setRGBScore(score/normalizer)
			patches[i].setHSVScore(score/normalizer)

	"""
	---------Corner Response Uniqueness Score-------------
	:start from top left window of size sigma x simga, Harris corner score recorded every sigma/2 pixel by default
	"""
	if(WEIGHTS_DICT['CORNER'] > 0):
		maxCornerResponse, cornerResponseMatrix = cornerResponse.getHarrisCornerResponse(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), sigma, step)
		print "cornerResponseMatrix.shape (should be same as img.shape):", cornerResponseMatrix.shape
		print "maxCornerResponse:", maxCornerResponse, " = ", np.amax(cornerResponseMatrix)
		print "most negative Response:", np.amin(cornerResponseMatrix)
		for i in range(0, len(patches)):
			patches[i].setCornerResponseScore(cornerResponseMatrix, np.amax(cornerResponseMatrix), np.amin(cornerResponseMatrix))
			# after HSV and Corner are set, setOverallScore
			# patches[i].setOverallScore()

	"""
	---------HOG Uniqueness Score-------------
	"""
	if(WEIGHTS_DICT['HOG'] > 0):
		img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.int)
		for i in range(0, len(patches)):
			patches[i].computeHOG(img_gray, True)
		HOG_dissimilarity_matrix = computeDissimilarityMatrix(img_gray, patches, Jensen_Shannon_Divergence, "HOG")
		HOG_max_score = 0.0
		for i in range(0, HOG_dissimilarity_matrix.shape[0]):
			score = distinguishabilityScore(HOG_dissimilarity_matrix[i])
			if(score > HOG_max_score):
				HOG_max_score = score
			patches[i].setHOGScore(score)
		for i in range(0, len(patches)):
			patches[i].setHOGScore(patches[i].HOGScore/HOG_max_score)

	"""
	---------- Set Overall Score ------------
	"""
	for i in range(0, len(patches)):
		patches[i].setOverallScore()

	# TODO: Do not just sort based on the distinctiveness score, sometimes, 
	#       two similar patches will be both unique compared to the rest of the image, 
	#       in this case we should go through the sorted list and only pick those have not had similar unique patches appeared before.
	sorted_patches = sorted(patches, key = lambda patch: patch.overallScore, reverse=True)
	print "check sorted score"
	for i in range(0, len(sorted_patches)):
		print sorted_patches[i].overallScore
	return sorted_patches
	# return sorted(patches, key = lambda patch: patch.HSVScore, reverse=True)
	# return sorted(patches, key = lambda patch: patch.HOGScore, reverse=True)
	# return sorted(patches, key = lambda patch: patch.cornerResponseScore, reverse=True)

def LDAFeatureScore(this_feature_set, this_feature_weights, testPatch, random_patches, plotHist = False,  path = "", testPatchIndex = 0):
	"""
	this_feature_set: feature sets to consider
	this_feature_weights: weights of features in the set
	return: weighed LDA statistics of this testPatch and random_patches wrt the feature sets
	"""
	test_patch_response = 0
	for i in range(0, len(this_feature_set)):
		cur_feature_attr = this_feature_set[i] + "Score"
		cur_feature_weight = this_feature_weights[i]
		test_patch_response  += getattr(testPatch, cur_feature_attr) * cur_feature_weight

	random_patches_response = []
	for j in range(0, len(random_patches)):
		one_response = 0
		for i in range(0, len(this_feature_set)):
			cur_feature_attr = this_feature_set[i] + "Score"
			cur_feature_weight = this_feature_weights[i]
			one_response += getattr(random_patches[j], cur_feature_attr) * cur_feature_weight
		random_patches_response.append(one_response)
		
	# make the distribution to be np array
	random_patches_response = np.asarray(random_patches_response)

	# plot the distribution and the testPatch response
	if(plotHist):
		plotStatistics.plotResponseDistribution(path+"/hists", this_feature_set, testPatchIndex, test_patch_response, random_patches_response)

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


def setOnePatchScoreForAllFeatures(patch, img, img_gray, gaussianWindow, full_image_HueHist, full_image_SaturationHist, full_image_ValueHist):
	# HSV Feature
	patch.computeSinglePatchHSVHistogram(img, gaussianWindow, True)
	patch.HueHist = patch.HueHistArr[0]
	patch.SaturationHist = patch.SaturationHistArr[0]
	patch.ValueHist = patch.ValueHistArr[0]
	patch.setHSVScore(compareSeperateHSVHists(patch, full_image_HueHist, full_image_SaturationHist, full_image_ValueHist))

	# HOG Feature
	patch.computeHOG(img_gray, True)
	patch.setHOGScore(HOGResponse(patch.HOG))


def findCombinatorialFeatureScore(img, testPatches, sigma, path):
	"""
	img: the base image,
	testPatches: the set of unique patches;
	return: the score of different combination of features in LDA statistics 
	"""
	img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.int)
	gaussianWindow = gauss_kernels(sigma, sigma/6.0)
	full_image_HueHist, full_image_SaturationHist, full_image_ValueHist = computeFullImageHSVHistogram(img)
	
	random_patches = extractRandomPatches(img, sigma, 200)

	for i in range(0, len(testPatches)):
		setOnePatchScoreForAllFeatures(testPatches[i], img, img_gray, gaussianWindow, full_image_HueHist, full_image_SaturationHist, full_image_ValueHist)
	for i in range(0, len(random_patches)):
		setOnePatchScoreForAllFeatures(random_patches[i], img, img_gray, gaussianWindow, full_image_HueHist, full_image_SaturationHist, full_image_ValueHist)

	features = ["HSV", "HOG"]
	all_feature_sets = generateAllFeatureSets(features)
	feature_sets_score = np.zeros(shape = (len(all_feature_sets), len(testPatches)))
	
	for i in range(0, len(all_feature_sets)):
		this_feature_set = all_feature_sets[i]
		this_feature_weights = np.ones(len(this_feature_set))
		print "checking score for set: ", this_feature_set, "with weights: ", this_feature_weights
		for j in range(0, len(testPatches)):
			feature_sets_score[i][j] = LDAFeatureScore(this_feature_set, this_feature_weights, testPatches[j], random_patches, True, path, j)

	# Log out the feature_sets_score for each testPatch
	print "------------ Logging feature_sets_score for each testPatch ------------"
	for i in range(0, len(testPatches)):
		for j in range(0, len(all_feature_sets)):
			print "testPatch[{i}] ".format(i = i), all_feature_sets[j], " Score: ", feature_sets_score[j][i]
		print ""
	return feature_sets_score


# patches can be 1. an instance of Patch class 2. A list of patches
# Gradiant is to indiacte the goodness of the match patch, the ligher(redder) the better
def drawPatchesOnImg(img, patches, show = True, gradiant = None, color = (0,0,255)): # gradiant is supposed to be  = 1.0/len(patches)
	if(type(patches).__name__ == "instance"):
		p = patches
		cv2.rectangle(img,(p.y-p.size/2,p.x-p.size/2),(p.y+p.size/2,p.x+p.size/2),color,1) # np.random.randint(0,255,size = 3)
	elif(type(patches) is list):	
		for i in range(0, len(patches)):
			p = patches[i]
			if(gradiant is None):
				cv2.rectangle(img,(p.y-p.size/2,p.x-p.size/2),(p.y+p.size/2,p.x+p.size/2),color,1)
			else:
				cv2.rectangle(img,(p.y-p.size/2,p.x-p.size/2),(p.y+p.size/2,p.x+p.size/2),(color[0]*(1 - gradiant * i),color[1]*(1 - gradiant * i),color[2]*(1 - gradiant * i)),1) # np.random.randint(0,255,size = 3)
	
	if(show):	
		cv2.imshow("draw Patches On Original Image",img)
		cv2.waitKey(0)
	return img

def drawMatchesOnImg(img, imgToMatch, patches, matches, show = True):
	drawPatchesOnImg(img, patches, False)
	drawPatchesOnImg(imgToMatch, matches, False)

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

def populateTestCombinatorialFeatureScore(test_folder_name, img_name, sigma = 39, upperPath = "testAlgo2", folder_suffix = "_eyeballed_unique_patches", image_db = "images"):
	path = upperPath + "/GaussianWindowOnAWhole/" + test_folder_name + folder_suffix
	img = cv2.imread("{image_db}/{folder}/{name}".format(image_db = image_db, folder = test_folder_name,  name = img_name), 1)
	testPatches = []
	listOfPatches = saveLoadPatch.loadPatchMatches("{path}/DistinguishablePatch_{folder}_{file}_simga{i}_GaussianWindowOnAWhole.csv".format( \
			path = path , \
			folder = test_folder_name, \
			file = "test1", \
			i = sigma))
	for i in range(0, len(listOfPatches)):
		testPatches.append(listOfPatches[i][0]) # just append the best match
	feature_set_scores = findCombinatorialFeatureScore(img, testPatches, sigma, path)
	print feature_set_scores

def main():
	
	# ---------------------------- Test -----------------------------
	# temp1 = np.zeros(16)
	# temp2 = np.zeros(16)
	# # print chiSquareDistance(temp1, temp2)
	# # for i in range(0, len(temp1)):
	# # 	# temp1[i] = np.random.randint(0,10)
	# # 	temp2[i] = np.random.randint(0,10)
	# # 	# temp1[i] = i
	# # 	temp2[i] = 1
	# temp1[5] = 11
	# temp2[5] = 10
	# temp1[4] = 8
	# temp2[4] = 9
	# print temp1
	# print temp2
	# mean = (temp1 + temp2)/2
	# print mean

	# print earthMoverHatDistanceForHue(temp1, temp2)
	# print earthMoverHatDistance(temp1, temp2)
	# print klDivergence(temp1.astype(float), temp2.astype(float))
	# print Jensen_Shannon_Divergence(temp1, temp2)

	# cv2.imshow("test1",img)
	# cv2.waitKey(0)
	# kp = []
	# keypoint1 = cv2.KeyPoint (100,100, 200, 60)
	# kp.append(keypoint1)
	# temp = cv2.drawKeypoints(img,kp,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
	# cv2.imshow("test1",temp)
	# cv2.waitKey(0)

	### Testing for opencv calcHist function###
	# img = cv2.imread("images/{folder}/{name}".format(folder = "testset4",  name = "test1.jpg"), 1)
	# H, S,V = computeFullImageHSVHistogram(img)

	# img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	# img_hsv_hist = cv2.calcHist([img_hsv],[0,1], None, [16,16], [0, 180, 0,256 ])
	# img_hsv_hist = cv2.calcHist([img_hsv],[1], None, [16], [0,256 ])
	# img_hsv_hist = img_hsv_hist[:,0]
	# img_hsv_hist = img_hsv_hist/np.sum(img_hsv_hist)
	# print len(img_hsv_hist)
	# print img_hsv_hist
	# print np.sum(img_hsv_hist)

	# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.int)
	# print img.dtype
	# print img.shape
	# tempPatches = extractPatches(img, 39, 1)
	# print "number of patches extracted:",len(tempPatches)
	# for i in range(0, len(tempPatches)):
	# 	print "computeHOG for tempPatches[{i}]".format(i =i )
	# 	tempPatches[i].computeHOG(img, True)
	
	### Test HOG related ### 
	# temp = Patch(19, 19, 39)
	# temp.computeHOG(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.int))
	# print HOGResponse(temp.HOG)

	# for i in range(0, len(temp.HOGArr)):
		# print temp.HOGArr[i]
		# print np.sum(temp.HOGArr[i])

	
	# H, S, V = temp.RGBToHSV(222,100,10)
	# print "Test RGBToHSV conversion:", H,", ", S, ", ",  V
	# print CforHue(8)
	# temp = cv2.DMatch(1,1,1)

	# arr = np.array([1,2,3,8,0,10,11,0])
	# max_ori =  np.argmax(arr)
	# arr =  list(arr[max_ori:len(arr)]) + list(arr[0:max_ori])
	# print np.array(arr)

	folderNames = ["testset_illuminance1"]
	### Test Algo2 in finding distinguishable patches ###
	# for i in range(0, len(folderNames)):
	# 	populateTestFindDistinguishablePatchesAlgo2(folderNames[i], "test1.jpg", 39)
	# raise ValueError ("stop for test findDistinguishablePatchesAlgo2")

	### Test combinatorial feature scores on a set of eyeballed patches
	for i in range(0, len(folderNames)):
		populateTestCombinatorialFeatureScore(folderNames[i], "test1.jpg",39)
	raise ValueError ("purpose stop for TestCombinatorialFeatureScore")

	#---------------------------Extract Bright Patches------------------------
	# img = cv2.imread("aggregateRGBResponseImageSize_sigma101.jpg", 0)
	# print img.shape
	# tempPatches = extractPatches(img, 51, 1) # extract patches of size that can capture the distinguishable regions	
	# for i in range(0, len(tempPatches)):
	# 	tempPatches[i].computeAggregateRGBScore(img)
	# 	if(tempPatches[i].aggregateRGBScore>0):
	# 		print tempPatches[i].x, tempPatches[i].y, tempPatches[i].size
	# sortedPatches = sorted(tempPatches, key = lambda patch: patch.aggregateRGBScore, reverse = True)
	# kp = []
	# i = 0
	# while(i< 50):
	# 	print sortedPatches[i].x, sortedPatches[i].y, sortedPatches[i].size, sortedPatches[i].aggregateRGBScore
	# 	# note the x coordinate in the drawing is actually width, and y is the height, which is reverse of our data representation's x and y
	# 	cv2.rectangle(img,(sortedPatches[i].y-sortedPatches[i].size/2,sortedPatches[i].x-sortedPatches[i].size/2),(sortedPatches[i].y+sortedPatches[i].size/2,sortedPatches[i].x+sortedPatches[i].size/2),(0,0,0),2) # np.random.randint(0,255,size = 3)
	# 	# kp.append(cv2.KeyPoint(sortedPatches[i].y, sortedPatches[i].x, sortedPatches[i].size))
	# 	i += 1
	# imgWithKeyPoints = cv2.drawKeypoints(img,kp,flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
	# cv2.imshow("temp", imgWithKeyPoints)
	# cv2.waitKey(0)
	# raise ValueError("test")

	#-------------------------- ComparePatches ---------------------------
	imgName = "test1.jpg"
	folderName = "testset4"
	img = cv2.imread("images/{folder}/{name}".format(folder = folderName,  name = imgName), 1)
	print img.shape

	sigma = 39
	patches = extractPatches(img, sigma,1)
	print "number of patches:", len(patches)
	# computePatchesRGBHistogram(img,patches)
	computePatchesHSVHistogram(img,patches)
	dissmilarityMatrix = computeDissimilarityMatrix(img, patches, Jensen_Shannon_Divergence, "HSV")
	print "dissmilarityMatrix shape:", dissmilarityMatrix.shape 
	print np.amin(dissmilarityMatrix)
	print np.amax(dissmilarityMatrix)
	# plot the heat map of the dissimilarity matrix
	# heatmap = plt.pcolor(dissmilarityMatrix,  vmin=np.amin(dissmilarityMatrix), vmax=np.amax(dissmilarityMatrix))
	# plt.savefig("dissimilarityHeatMap_{img}.png".format(img = imgName[0:imgName.find(".")]))
	# plt.show()
	# plt.clf()

	distinguishableIndividualPatchIndexes = sortedIndexOfDistinguishablePatches(patches, dissmilarityMatrix, "HSV")

	# aggregateDistinguishabilityResponse = dissimilarityMetricOnImagePosition(img,patches,dissmilarityMatrix)

	# aggregateDistinguishabilityResponse = aggregateDistinguishabilityResponse * 255/np.amax(aggregateDistinguishabilityResponse)
	# aggregateDistinguishabilityResponse = aggregateDistinguishabilityResponse.astype(np.uint8)

	# cv2.imwrite("aggregateRGBResponseImageSize_{folder}_{img}_sigma{i}.jpg".format(folder = folderName, i = sigma, img = imgName[0:imgName.find(".")]), aggregateDistinguishabilityResponse)
	# plt.imshow(aggregateDistinguishabilityResponse, cmap = plt.get_cmap('gray'))
	# # plt.show()
	# # plt.clf()

	# aggregateResponsePatches = extractPatches(img, sigma, 1) # extract patches of size that can capture the distinguishable regions
	# for i in range(0, len(aggregateResponsePatches)):
	# 	aggregateResponsePatches[i].computeAggregateRGBScore(aggregateDistinguishabilityResponse)
	# sortedPatches = sorted(aggregateResponsePatches, key = lambda patch: patch.aggregateRGBScore, reverse=True)

	
	# imgCopy = np.copy(img)
	# responseCopy = np.copy(aggregateDistinguishabilityResponse)
	# for i in range(0, 10):
	# 	p = sortedPatches[i]
	# 	# print p.aggregateRGBScore
	# 	cv2.rectangle(responseCopy,(p.y-p.size/2,p.x-p.size/2),(p.y+p.size/2,p.x+p.size/2),(0,0,0),2) # np.random.randint(0,255,size = 3)
	# 	cv2.rectangle(imgCopy,(p.y-p.size/2,p.x-p.size/2),(p.y+p.size/2,p.x+p.size/2),np.random.randint(0,255,size = 3),2) # np.random.randint(0,255,size = 3)
	# cv2.imshow("sorted based on aggregateRGBScore",imgCopy)
	# cv2.imwrite("aggregateRGBScorePatches_{folder}_{img}_sigma{i}.jpg".format(folder =folderName,  i = 51, img = imgName[0:imgName.find(".")]), imgCopy)
	# cv2.waitKey(0)

	# draw the distinguishable patches
	# keypoints = []
	# for i in range(0, 50):
		# p = patches[distinguishableIndividualPatchIndexes[i]]
		# keypoints.append(cv2.KeyPoint(p.x,p.y,p.size))
	# temp = cv2.drawKeypoints(img,keypoints,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

	#plot the top few patches with the most distinguishable histogram
	for i in range(0, 10):
		p = patches[distinguishableIndividualPatchIndexes[i]]
		cv2.rectangle(img,(p.y-p.size/2,p.x-p.size/2),(p.y+p.size/2,p.x+p.size/2),np.random.randint(0,255,size = 3),1)
	# cv2.imwrite("individualRGBScorePatches_{folder}_{img}_sigma{i}.jpg".format(folder = folderName, i = sigma, img = imgName[0:imgName.find(".")]), img)
	# cv2.imshow("sort based on Individual RGB score",img)
	cv2.imwrite("individualNewHSVScorePatches_{folder}_{img}_sigma{i}.jpg".format(folder = folderName, i = sigma, img = imgName[0:imgName.find(".")]), img)
	cv2.imshow("sort based on Individual HSV score",img)
	cv2.waitKey(0)


	return


if __name__ == "__main__":
	main()