import comparePatches
import plotStatistics
import saveLoadPatch
import numpy as np
import cv2
from matplotlib import pyplot as plt
import math
import copy
import os
import time
import sys
from feature_modules import utils

FEATURE_WEIGHTING = {
	'HSV':1.0,
	'HOG':0.0
	# more features later
}

### For testset_illuminance1 ###
# MANUAL_FEATURE_TO_USE = [
# [utils.BOTTOM_RIGHT_GREEN_FEATURE_ID], \
# [utils.BOTTOM_RIGHT_YELLOW_FEATURE_ID], \
# [utils.TOP_RIGHT_YELLOW_FEATURE_ID], \
# [utils.TOP_LEFT_PURPLE_FEATURE_ID], \
# [utils.SHARP_HOG_FEATURE_ID], \
# [utils.DONUT_SHAPE_FEATURE_ID], \
# [utils.DONUT_SHAPE_FEATURE_ID, utils.BOTTOM_RIGHT_NEIGHBOUR_BLUE_FEATURE_ID]
# ]

### For testset_rotation1 ###
# MANUAL_FEATURE_TO_USE = [\
# [utils.BORDER_GREEN_FEATURE_ID], \
# [utils.CENTRE_YELLOW_FEATURE_ID]
# ]

### For testset7 ###
# MANUAL_FEATURE_TO_USE = [ \
# [utils.GENERATED_FEATURE_IDS[0]], \
# [utils.GENERATED_FEATURE_IDS[1]], \
# [utils.GENERATED_FEATURE_IDS[2]] \
# ]
MANUAL_FEATURE_TO_USE = []

ALL_FEATURE_TO_COMPUTE = list(set([item for features in MANUAL_FEATURE_TO_USE for item in features]))


# the lower the score (distance), the better;
def getHSVHistOverAllDistance(patchToMatch, potentialPatch, metricFunc = comparePatches.earthMoverHatDistance):
	# individualScores = np.zeros(len(patchToMatch.HSVHistArr))
	# for j in range(0, len(patchToMatch.HSVHistArr)):
	# 	oneHistScore = comparePatches.Jensen_Shannon_Divergence(patchToMatch.HSVHistArr[j], potentialPatch.HSVHistArr[j])
	# 	individualScores[j] = oneHistScore
	# return np.linalg.norm(individualScores, 2)
	return getHistArrl2Distance(patchToMatch.HSVHistArr, potentialPatch.HSVHistArr, metricFunc)

def getHistArrl2Distance(histArr1, histArr2, metricFunc):
	if(len(histArr1) != len(histArr2)): # if hist array not same size, do not consider for a patch match.
		raise ValueError ("In getHistArrl2Distance, the passed in shape of two histograms should be the same, but passed in length of histArr1 is {i}, length of histArr2 is {j}".format(i = len(histArr1), j =  len(histArr2)))
	individualScores = np.zeros(len(histArr1))
	for i in range(0, len(histArr1)):
		oneHistScore = metricFunc(histArr1[i], histArr2[i])
		individualScores[i] = oneHistScore
	return np.linalg.norm(individualScores, 2)

def getHSVSeperateOverAllDistances(patchToMatch, matchPatches, metricFunc):
	scores = np.zeros(len(matchPatches))
	for i in range(0, len(matchPatches)):
		scores[i] = getHSVSeperateHistAvgl2Distance(patchToMatch, matchPatches[i], metricFunc)
	return scores

def getHSVSeperateHistAvgl2Distance(patchToMatch, potentialPatch, metricFunc):
	if(metricFunc.__name__ == "earthMoverHatDistance"):
		HueHistDist = getHistArrl2Distance(patchToMatch.HueHistArr, potentialPatch.HueHistArr, comparePatches.earthMoverHatDistanceForHue) # special distance function for HUE
		SaturationHistDist = getHistArrl2Distance(patchToMatch.SaturationHistArr, potentialPatch.SaturationHistArr, metricFunc)
		# ValueHistDist = getHistArrl2Distance(patchToMatch.ValueHistArr, potentialPatch.ValueHistArr, metricFunc)
	else:
		HueHistDist = getHistArrl2Distance(patchToMatch.HueHistArr, potentialPatch.HueHistArr, metricFunc)
		SaturationHistDist = getHistArrl2Distance(patchToMatch.SaturationHistArr, potentialPatch.SaturationHistArr, metricFunc)
		# ValueHistDist = getHistArrl2Distance(patchToMatch.ValueHistArr, potentialPatch.ValueHistArr, metricFunc)
	# return np.linalg.norm([HueHistDist, SaturationHistDist, ValueHistDist], 2)
	return np.linalg.norm([HueHistDist,SaturationHistDist], 2)

# use the overall + 2*2 sub patch Jessen Divergence descriptor
# patchToMatch: the original patch to match; 
# matchPactches: list of patches (maybe of different size) as potential matcher; 
# n: number of matches returned; 
# histToUse: the histogram metric to use for comparison, HSV or RGB or ...
# return: A list of size n, the best n matches out of the matchPatches given
# default distance metric is Jensen_Shannon_Divergence since so far the best for seperate HS
def findBestMatches(patchToMatch, matchPatches, n = 1, histToUse = "HSV", metricFunc = comparePatches.Jensen_Shannon_Divergence, useSeperateHSVHists = True):
	"""
	patchToMatch: the test patch
	matchPatches: the array of potential good matches
	n: number of top matches to return, default is just return 1 best match
	"""

	"""HSV, HOG descriptor based"""
	color_distances = np.zeros(len(matchPatches))
	hog_distances = np.zeros(len(matchPatches))
	
	if(FEATURE_WEIGHTING['HSV'] != 0):
		if(useSeperateHSVHists):
			color_distances = getHSVSeperateOverAllDistances(patchToMatch, matchPatches, metricFunc)
		else:
			# color_distances = np.zeros(len(matchPatches))
			# # Compute C for earthMoverHatDistance: size of matchPatches[i] might be different but the length of HSVhist is the same: 4096 for HSV or 256 for HS
			# if(metric == "earthMoverHatDistance"):
			# 	histLen = len(patchToMatch.HSVHist) # poll the length of the flattened HSVHist
			# 	C = np.ones(shape = (histLen, histLen))
			# 	rows = np.arange(0,histLen).reshape((histLen, 1))
			# 	rows = np.repeat(rows, histLen, axis = 1)
			# 	cols = np.arange(0, histLen).reshape((1,histLen))
			# 	cols = np.repeat(cols, histLen, axis = 0)
			# 	C = C + abs(rows - cols)

			# Compute color_distances[i]
			for i in range(0, len(color_distances)):
				if(histToUse == "RGB"):
					individualScores = np.zeros(len(patchToMatch.RGBHistArr))
					for j in range(0, len(patchToMatch.RGBHistArr)):
						oneHistScore = metricFunc(patchToMatch.RGBHistArr[j], matchPatches[i].RGBHistArr[j])
						individualScores[j] = oneHistScore
				elif(histToUse == "HSV"):
					individualScores = np.zeros(len(patchToMatch.HSVHistArr))
					for j in range(0, len(patchToMatch.HSVHistArr)):
						oneHistScore = metricFunc(patchToMatch.HSVHistArr[j], matchPatches[i].HSVHistArr[j])
						individualScores[j] = oneHistScore
				color_distances[i] = np.linalg.norm(individualScores, 2)

	if(FEATURE_WEIGHTING['HOG'] != 0):
		# hog_distances = np.zeros(len(matchPatches))
		for i in range(0, len(hog_distances)):
			# TODO: if use earthMover for HOG, then we should use comparePatches.earthMoverHatDistanceForHOG with the correct C matrix
			hog_distances[i] = getHistArrl2Distance(patchToMatch.HOGArr, matchPatches[i].HOGArr, metricFunc) 
	
	overall_distances = np.sqrt(\
		FEATURE_WEIGHTING['HSV'] * color_distances**2 + \
		FEATURE_WEIGHTING['HOG'] * hog_distances**2)

	sortedIndex = np.argsort(overall_distances) # the lower the (distance) the better

	"""Use patch to patch similarity of the features"""
	# overall_distances = np.zeros(len(matchPatches))
	# for i in range(0, len(overall_distances)):
	# 	distance_vector = []
	# 	for this_feature_id in patchToMatch.feature_to_use:
	# 		# distance_vector.append( \
	# 		# 	metricFunc( \
	# 		# 	patchToMatch.getFeatureObject(this_feature_id).hist, \
	# 		# 	matchPatches[i].getFeatureObject(this_feature_id).hist\
	# 		# 	))
	# 		distance_vector.append(patchToMatch.getFeatureObject(this_feature_id).dissimilarityWith(\
	# 			matchPatches[i].getFeatureObject(this_feature_id)))
	# 	distance_vector = np.asarray(distance_vector)
	# 	assert (len(distance_vector) == len(patchToMatch.feature_weights)), "In findBestMatches: length of distance_vector must be the same as test patch feature_weights"
	# 	overall_distances[i] = np.linalg.norm(np.multiply(distance_vector, patchToMatch.feature_weights), 2)

	# sortedIndex = np.argsort(overall_distances) # the lower the (distance) the better

	"""Use the rank of feature score, if test patch response < mean, choose the lowest score, else if > mean, choose the highest score"""
	# overall_scores = np.zeros(len(matchPatches))
	# for i in range(0, len(overall_scores)):
	# 	score_vector = []
	# 	for this_feature_id in patchToMatch.feature_to_use:
	# 		score_vector.append(matchPatches[i].getFeatureObject(this_feature_id).score)
	# 	score_vector = np.asarray(score_vector)
	# 	assert (len(score_vector) == len(patchToMatch.feature_weights)), "In findBestMatches: length of distance_vector must be the same as test patch feature_weights"
	# 	overall_scores[i] = np.sum(np.multiply(score_vector, patchToMatch.feature_weights))
	
	# if (patchToMatch.is_low_response):
	# 	sortedIndex = np.argsort(overall_scores) # get the lowest scores as possible
	# else:
	# 	sortedIndex = np.argsort(overall_scores)[::-1] # get the highest scores as possible

	# Return the best n matchPatches as a list
	results = []
	for i in range(0, n):
		results.append(matchPatches[sortedIndex[i]])
	return results


def copyPatchesWithScale(matchPatches, scale, level, img):
	results = copy.deepcopy(matchPatches)
	scaleChange = scale * level
	# print scaleChange
	# print results[0].getSize() + scaleChange
	for i in range(0,len(results)):
		newSize = (results[i].getSize() + scaleChange)
		if(results[i].x - newSize/2 >= 0 and results[i].x + newSize/2 < img.shape[0] and results[i].y - newSize/2 >=0 and results[i].y + newSize/2 < img.shape[1]):
			results[i].setSize(newSize)	
	return results

def getGaussianScale(originalScale, factor, level):
	if(level < 0):
		newScale = int(originalScale / (factor ** abs(level)))
		newScale = newScale - 1 if (newScale % 2 == 0)  else newScale
	else:
		newScale = int(originalScale * (factor ** abs(level)))
		newScale = newScale + 1  if (newScale % 2 == 0) else newScale
	return newScale

def adjustNewPatchSizeNewDimension(new_patch_x, new_patch_y, new_patch_size, new_dimension):
	"""
	new_patch_size: the target new new_patch_size
	new_dimension: new_dimension[0] -> number of rows, new_dimension[1] -> number of columns
	return: adjusted new_patch_size so that it fits in the dimension
	"""
	if (new_patch_x - new_patch_size/2 < 0):
		new_patch_size = 2 * new_patch_x + 1
	if (new_patch_y - new_patch_size/2 < 0):
		new_patch_size = 2 * new_patch_y + 1
	if (new_patch_x + new_patch_size/2 >= new_dimension[0]):
		new_patch_size = (new_dimension[0] - new_patch_x) * 2 - 1
	if (new_patch_y + new_patch_size/2 >= new_dimension[1]):
		new_patch_size = (new_dimension[1] - new_patch_y) * 2 - 1
	return new_patch_size

def testFindOnePatchMatch(patchToMatch, patchesArr, k = 1, metricFunc = comparePatches.Jensen_Shannon_Divergence): # k : number of best patches to extract for a sigma level (for one patch array)
	goodMatchesArr = []
	for i in range(0, len(patchesArr)):		
		goodMatches = findBestMatches(patchToMatch, patchesArr[i],k, "HSV", metricFunc)
		goodMatchesArr.append(goodMatches)
	overAllGoodMatches = [item for sublist in goodMatchesArr for item in sublist] # flatten the list of list -> list
	print "In testFindOnePatchMatch: overAllGoodMatches len should be {i}:".format(i = k*len(patchesArr)), len(overAllGoodMatches)
	return findBestMatches(patchToMatch, overAllGoodMatches, k*len(patchesArr), "HSV", metricFunc)
	# return findBestMatches(patchToMatch, overAllGoodMatches, k, "HSV", metricFunc = comparePatches.Jensen_Shannon_Divergence)

def createFolder(upperPath, folderToSave, folderName, suffix):
	"""
	upperPath: root folder,
	folderToSave: "GaussianWindowOnAWhole",
	folderName:test_folder_name,
	suffix: suffix to append to test_folder_name
	"""
	if(not os.path.isdir("./{path}/{folderToSave}/{testFolder}".format(
		path = upperPath, 
		folderToSave = folderToSave, 
		testFolder = folderName + suffix))):
		os.makedirs("./{path}/{folderToSave}/{testFolder}".format(
			path = upperPath, 
			folderToSave = folderToSave, 
			testFolder = folderName + suffix))
	return "./{path}/{folderToSave}/{testFolder}".format(
		path = upperPath, 
		folderToSave = folderToSave, 
		testFolder = folderName + suffix)

def drawCombinedMatchView(thisImg, thisImgToMatch, thisTestPatches, thisListOfMatches, show = False):
	# draw the combined match view at this level
	thisMatchesFound = []
	for j in range(0, len(thisListOfMatches)):
		thisMatchesFound.append(thisListOfMatches[j][0]) # just append the best match
	return comparePatches.drawMatchesOnImg(thisImg, thisImgToMatch, thisTestPatches, thisMatchesFound, show)

def testDescriptorPerformancePyramidWorker(testPatches, img, img_gray,imgToMatch, imgToMatch_gray, sigma, \
	testFolderName,  patchStep = 0.5, useGaussianWindow = True, initialize_features = True):
	"""Build image pyramid using opencv"""
	level = 2
	imgPyd = []
	imgPyd.append(np.copy(img))
	imgToMatchPyd = []
	imgToMatchPyd.append(np.copy(imgToMatch))

	img_down_sample = np.copy(img)
	imgToMatch_down_sample = np.copy(imgToMatch)
	for i in range(0, level):
		img_down_sample = cv2.pyrDown(img_down_sample)
		imgToMatch_down_sample = cv2.pyrDown(imgToMatch_down_sample)
		imgPyd.append(np.copy(img_down_sample))
		imgToMatchPyd.append(np.copy(imgToMatch_down_sample))

	# for i in range(0, len(imgPyd)):
	# 	print imgPyd[i].shape
	# 	cv2.imshow("img pyd [{i}]".format(i = i), imgPyd[i])
	# 	cv2.waitKey(0)
	# 	print imgToMatchPyd[i].shape
	# 	cv2.imshow("imgToMatch pyd [{i}]".format(i = i), imgToMatchPyd[i])
	# 	cv2.waitKey(0)

	"""propogateUp/propogateDown testPatches"""
	testPatchesPyd = []
	testPatchesPyd.append(testPatches)
	for i in range(1, level+1):
		thisLevelPatches = []
		for j in range(0, len(testPatches)):
			newSize = testPatches[j].size/(2**i)
			if(newSize % 2 == 0): # if even newSize, make it odd
				newSize += 1
			newSize = adjustNewPatchSizeNewDimension(testPatches[j].x/(2**i), testPatches[j].y/(2**i), newSize, imgPyd[i].shape)
			newPatch = comparePatches.Patch(testPatches[j].x/(2**i), testPatches[j].y/(2**i), newSize, \
				initialize_features = initialize_features)
			newPatch.setFeatureWeights(testPatches[j].feature_weights)
			newPatch.setFeatureToUse(testPatches[j].feature_to_use)
			thisLevelPatches.append(newPatch)
		testPatchesPyd.append(thisLevelPatches)

	# for i in range(0, len(testPatchesPyd)):
	# 	comparePatches.drawPatchesOnImg(np.copy(imgPyd[i]), testPatchesPyd[i])
	"""do the matching at top level and then repeat matching at lower pyramid level"""
	matchesFound = None # a list of [list of matches] for each test patch
	NUM_PATCH_SIZE_GAUSSIAN = 5 # number of different patch sizes used on top level of pyramid
	for i in xrange(level, -1, -1):
		thisImg = imgPyd[i]
		thisImgToMatch = imgToMatchPyd[i]
		thisImgGray = cv2.cvtColor(thisImg, cv2.COLOR_BGR2GRAY).astype(np.int)
		thisImgToMatchGray = cv2.cvtColor(thisImgToMatch, cv2.COLOR_BGR2GRAY).astype(np.int)
		thisImgHSV = cv2.cvtColor(thisImg.astype(np.float32), cv2.COLOR_BGR2HSV)
		thisImgToMatchHSV = cv2.cvtColor(thisImgToMatch.astype(np.float32), cv2.COLOR_BGR2HSV)
		thisTestPatches = testPatchesPyd[i]


		if(i == level):
			"""if top level of pyramid, run original matching algorithm"""
			thisListOfMatches = testDescriptorPerformanceWorker(thisTestPatches, \
				thisImg, thisImgGray , thisImgHSV, \
				thisImgToMatch, thisImgToMatchGray, thisImgToMatchHSV, \
				sigma/(2**i), testFolderName, NUM_PATCH_SIZE_GAUSSIAN**(i+1), \
				initialize_features = initialize_features) # each gaussian 125 best matches
			matchesFound = thisListOfMatches
			"""draw the combined match view to check at this level"""
			# drawCombinedMatchView(np.copy(thisImg), thisImgToMatch, thisTestPatches, matchesFound, True)
			# for j in range(0, len(matchesFound)):
				# comparePatches.drawPatchesOnImg(np.copy(thisImgToMatch), matchesFound[j], mark_sequence = True)
		else:
			"""otherwise, populate down the potential good matches and rematch"""
			matchesFoundNextLevel = []
			for j in range(0, len(matchesFound)):
				matchesFoundNextLevelOnePatch = []
				for k in range(0, len(matchesFound[j])):
					"""populate down the imgPyd and create new patches"""
					newPatch = comparePatches.Patch(matchesFound[j][k].x*2,matchesFound[j][k].y*2, \
						adjustNewPatchSizeNewDimension(\
							matchesFound[j][k].x*2,\
							matchesFound[j][k].y*2,\
							matchesFound[j][k].size*2 + 1, \
							thisImg.shape), initialize_features = initialize_features)
					# compute Histogram for new potential match patch
					# for this_feature in ALL_FEATURE_TO_COMPUTE:
					# 	newPatch.getFeatureObject(this_feature).computeFeature(thisImgToMatch)
					# 	newPatch.getFeatureObject(this_feature).computeScore()
					if(FEATURE_WEIGHTING['HSV'] != 0):
						newPatch.computeHSVHistogram(thisImgToMatchHSV,useGaussianWindow)
					if(FEATURE_WEIGHTING['HOG'] != 0):
						newPatch.computeHOG(thisImgToMatchGray, useGaussianWindow)
					# append the new potential match patch for current particular one test patch
					matchesFoundNextLevelOnePatch.append(newPatch)
				matchesFoundNextLevel.append(matchesFoundNextLevelOnePatch)

			# rematch
			rematches = []
			for j in range(0, len(thisTestPatches)):
				thisPatchToMatch = thisTestPatches[j]
				# for this_feature in thisPatchToMatch.feature_to_use:
				# 	thisPatchToMatch.getFeatureObject(this_feature).computeFeature(thisImg)
				# 	thisPatchToMatch.getFeatureObject(this_feature).computeScore()
				if(FEATURE_WEIGHTING['HSV'] != 0):
					thisPatchToMatch.computeHSVHistogram(thisImgHSV,useGaussianWindow)
				if(FEATURE_WEIGHTING['HOG'] != 0):
					thisPatchToMatch.computeHOG(thisImgGray, useGaussianWindow)
				rematches.append(findBestMatches(thisPatchToMatch, \
					matchesFoundNextLevel[j], \
					NUM_PATCH_SIZE_GAUSSIAN**(i+1), \
					metricFunc = comparePatches.Jensen_Shannon_Divergence))	
			# free memeory and reassign matchesFound
			del matchesFound[:]
			matchesFound = []
			matchesFound = rematches
			# draw the combined match view to check at this level
			# drawCombinedMatchView(thisImg, thisImgToMatch, thisTestPatches, matchesFound, True)

	"""------For verification at the end of the matching algorithm------"""
	assert len(testPatches) == len(matchesFound), \
	"In testDescriptorPerformancePyramidWorker: number of test patches should be the same as number of array of [matches found]"
	for i in range(0, len(matchesFound)):
		assert len(matchesFound[i]) == NUM_PATCH_SIZE_GAUSSIAN, \
		"final length of matches for each test path should be {n}".format(n = NUM_PATCH_SIZE_GAUSSIAN)

	return matchesFound # a list of [list of good matches] for each test patch


def testDescriptorPerformanceWorker(testPatches, img, img_gray, imgHSV, imgToMatch, imgToMatch_gray, imgToMatchHSV, \
	sigma, testFolderName, k = 1, \
	patchStep = 0.5, useGaussianWindow = True, metricFunc = comparePatches.Jensen_Shannon_Divergence, initialize_features = True):
	#Extract match patches
	matchPatches_origin = comparePatches.extractPatches(imgToMatch, sigma, patchStep, initialize_features = initialize_features)
	print "length of matchPatches:", len(matchPatches_origin)
	# scale = 8 # window pixel window for up and down scaling
	gaussianScaleFactor = 1.2
	patchesArr = []
	patchesArr.append(matchPatches_origin)

	for level in range(-2, 3):
		if(level != 0):
			# patchesArr.append(comparePatches.extractPatches(imgToMatch, sigma + scale * level, patchStep))
			print "new sigma:", getGaussianScale(sigma, gaussianScaleFactor, level)
			patchesArr.append(comparePatches.extractPatches(imgToMatch, getGaussianScale(sigma, gaussianScaleFactor, level), \
				patchStep, initialize_features = initialize_features))
	for i in range(0, len(patchesArr)):
		print "len(patchesArr[{i}]):".format(i = i),len(patchesArr[i])

	#Compute the Features' Hist of the potential matchPatches
	for index in range(0, len(patchesArr)):
		matchPatches = patchesArr[index]
		for i in range(0, len(matchPatches)):
			"""compute all features needed for the potential match patches"""
			# for this_feature in ALL_FEATURE_TO_COMPUTE:
			# 	matchPatches[i].getFeatureObject(this_feature).computeFeature(imgToMatch)
			# 	matchPatches[i].getFeatureObject(this_feature).computeScore()
			"""old descriptor based matching"""
			if(FEATURE_WEIGHTING['HSV'] != 0):
				# test_start_time = time.time()
				matchPatches[i].computeHSVHistogram(imgToMatchHSV,useGaussianWindow)
				# print "compute HSV for match patch: ", i, " time spent:", time.time() - test_start_time
			if(FEATURE_WEIGHTING['HOG'] != 0):
				matchPatches[i].computeHOG(imgToMatch_gray, useGaussianWindow)

		"""-------For logging purpose only: One PatchesArr done!------"""
		print "compute match patches' features of patchArr[{index}] in {f} done".format(i =i, index = index , f = testFolderName)

	testPatchMatches = []
	#loop over all test patches
	for this_test_patch in testPatches:
		"""compute only the features needed for test patches (not needed for full algorithm, since it will be done during detection phase)"""
		# print "feature to use for this_test_patch:", this_test_patch.feature_to_use
		# for this_feature in this_test_patch.feature_to_use:
		# 	this_test_patch.getFeatureObject(this_feature).computeFeature(img)
		# 	this_test_patch.getFeatureObject(this_feature).computeScore()
		"""old descriptor based matching"""
		if(FEATURE_WEIGHTING['HSV'] != 0):
			this_test_patch.computeHSVHistogram(imgHSV, useGaussianWindow)
		if(FEATURE_WEIGHTING['HOG'] != 0):
			this_test_patch.computeHOG(img_gray, useGaussianWindow)
		bestMatches = testFindOnePatchMatch(this_test_patch, patchesArr, k, metricFunc) # will return the best matches (in an array) of the testPatch1
		testPatchMatches.append(bestMatches)

	return testPatchMatches # return a list of [list of good matches for one test patch]



def testDescriptorPerformance(image_db, folderName,testPatches, imgName,imgToMatchName,folderToSave, \
	useGaussianWindow, suffix = "", sigma = 39, upperPath = "testPatchHSV", initialize_features = True):
	"""
	testDescriptorPerformance: given testPatches, run the descriptor matching process;
	:number of gaussian: 5 (2 level down, 2 level up)
	:gaussianScaleFactor: 1.2
	:gaussian window weighting on the whole patch: gaussian sigma = window length / 6
	:sigma: 39
	:patch search step: 0.5 (shift by 1/4 patch length)
	"""
	print "ALL_FEATURE_TO_COMPUTE:", ALL_FEATURE_TO_COMPUTE
	#Create folder for the testset
	if(not os.path.isdir("./{upperPath}/{folderToSave}/{testFolder}".format(
		upperPath = upperPath, 
		folderToSave = folderToSave, 
		testFolder = folderName + suffix))):
		os.makedirs("./{upperPath}/{folderToSave}/{testFolder}".format(
			upperPath = upperPath, 
			folderToSave = folderToSave, 
			testFolder = folderName + suffix))

	img = cv2.imread("{image_db}/{folder}/{name}".format(image_db = image_db, folder = folderName, name = imgName), 1)
	img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.int)
	imgToMatch = cv2.imread("{image_db}/{folder}/{name}".format(image_db = image_db, folder = folderName, name = imgToMatchName), 1)
	imgToMatch_gray = cv2.cvtColor(imgToMatch, cv2.COLOR_BGR2GRAY).astype(np.int)

 	# Move the window by 1/4 
 	patchStep = 0.5
	testPatchMatches = testDescriptorPerformancePyramidWorker(testPatches, img, img_gray,imgToMatch, \
		imgToMatch_gray, sigma, folderName, patchStep, initialize_features = initialize_features)
	# Logging and Saving of match results
	for testPatchIndex in range(0, len(testPatches)):
		this_test_patch = testPatches[testPatchIndex]
		cv2.imwrite("{upperPath}/{folderToSave}/{testFolder}/testPatch{testPatchIndex}_OriginalPatches_{folder}_{file1}_{file2}_simga{i}.jpg".format(
			upperPath = upperPath, 
			folderToSave = folderToSave, 
			testFolder = folderName + suffix, 
			testPatchIndex= testPatchIndex, 
			folder = folderName, 
			file1 = imgName[0:imgName.find(".")], 
			file2 = imgToMatchName[0:imgToMatchName.find(".")], 
			i = sigma), comparePatches.drawPatchesOnImg(np.copy(img), this_test_patch, False))
		
		bestMatches = testPatchMatches[testPatchIndex]
		print "matches found for test patch ", testPatchIndex
		for i in range (0, len(bestMatches)):
			print bestMatches[i].x, ",", bestMatches[i].y, ",",bestMatches[i].size
		
		imgToSave = comparePatches.drawPatchesOnImg(np.copy(imgToMatch),bestMatches, False, 1.0/len(bestMatches))
		cv2.imwrite("{upperPath}/{folderToSave}/{testFolder}/testPatch{testPatchIndex}_GoodMatches_{folder}_{file1}_{file2}_simga{i}_shiftBy{step}_useGaussianWindow_{tf}_5levels.jpg".format(
			upperPath = upperPath, 
			folderToSave = folderToSave, 
			testFolder = folderName + suffix,
			testPatchIndex = testPatchIndex,  
			folder = folderName, 
			file1 = imgName[0:imgName.find(".")], 
			file2 = imgToMatchName[0:imgToMatchName.find(".")], 
			i = sigma, 
			step = patchStep, 
			tf = useGaussianWindow), imgToSave)
	
	flattend_testPatchMatches = [item for sublist in testPatchMatches for item in sublist] # flatten the list of list -> list
	# for each test patch, store the 5 best matches at 5 gaussian levels
	saveLoadPatch.savePatchMatches(flattend_testPatchMatches, 5, "{upperPath}/{folderToSave}/{testFolder}/GoodMatches_{folder}_{file1}_{file2}_simga{i}_shiftBy{step}_useGaussianWindow_{tf}_5levels.csv".format(
		upperPath = upperPath, 
		folderToSave = folderToSave, 
		testFolder = folderName + suffix, 
		folder = folderName, 
		file1 = imgName[0:imgName.find(".")], 
		file2 = imgToMatchName[0:imgToMatchName.find(".")], 
		i = sigma, 
		step = patchStep, 
		tf = useGaussianWindow))
	return testPatchMatches

def checkHistogramOfTruthAndMatchesFound(testPatches, groundTruth, matchesFound, img, imgToMatch, path, saveHist = False, displayHist = True):
	print "path:", path
	if(not os.path.isdir(path)):
		os.makedirs(path)

	for i in range(0, len(testPatches)):
		"""old descriptor checking HSV and HOG hists"""
		testPatches[i].computeHSVHistogram(cv2.cvtColor(img.astype(np.float32), cv2.COLOR_BGR2HSV), True, True)
		groundTruth[i].computeHSVHistogram(cv2.cvtColor(imgToMatch.astype(np.float32), cv2.COLOR_BGR2HSV), True, True)
		matchesFound[i].computeHSVHistogram(cv2.cvtColor(imgToMatch.astype(np.float32), cv2.COLOR_BGR2HSV), True, True)

		testPatches[i].computeHOG(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.int),  True)
		groundTruth[i].computeHOG(cv2.cvtColor(imgToMatch, cv2.COLOR_BGR2GRAY).astype(np.int),  True)
		matchesFound[i].computeHOG(cv2.cvtColor(imgToMatch, cv2.COLOR_BGR2GRAY).astype(np.int), True)

		print "\nfor test patch ", i
		plotStatistics.plotHOGHistCmp(path, \
			"HOGhistCmp{i}".format(i = i), \
			testPatches[i].HOG, "testPatch{i}".format(i = i), \
			matchesFound[i].HOG, "matchFound{i}".format(i = i),\
			groundTruth[i].HOG, "groundTruth{i}".format(i = i), \
			saveHist, displayHist)
		plotStatistics.plotHSVSeperateHistCmp(path, \
			"HSV Seperate Cmp_testPatch{i}".format(i = i), \
			[testPatches[i].HueHist, testPatches[i].SaturationHist, testPatches[i].ValueHist], "testPatch{i}".format(i = i) , \
			[matchesFound[i].HueHist, matchesFound[i].SaturationHist, matchesFound[i].ValueHist],  "matchFound{i}".format(i = i), \
			[groundTruth[i].HueHist, groundTruth[i].SaturationHist, groundTruth[i].ValueHist], "groundTruth{i}".format(i = i), \
			saveHist, displayHist)

		matchDistance = getHSVSeperateHistAvgl2Distance(testPatches[i], matchesFound[i], comparePatches.Jensen_Shannon_Divergence)
		# matchDistance = comparePatches.Jensen_Shannon_Divergence(testPatches[i].HOG, matchesFound[i].HOG)
		print "Matches Found Distance:", matchDistance
		groundTruthDistance = getHSVSeperateHistAvgl2Distance(testPatches[i], groundTruth[i], comparePatches.Jensen_Shannon_Divergence)
		# groundTruthDistance = comparePatches.Jensen_Shannon_Divergence(testPatches[i].HOG, groundTruth[i].HOG)
		print "Gound Truth Distance:", groundTruthDistance
		print "overall Match Found better than Ground Truth?:", groundTruthDistance > matchDistance
		# print "overall Match Found better than Ground Truth? : ",  getHistArrl2Distance(\
		# 	testPatches[i].HOGArr, \
		# 	groundTruth[i].HOGArr, \
		# 	comparePatches.Jensen_Shannon_Divergence) > \
		# getHistArrl2Distance(\
		# 	testPatches[i].HOGArr, \
		# 	matchesFound[i].HOGArr, \
		# 	comparePatches.Jensen_Shannon_Divergence)
	return

def loadPatchesMatchesGroundtruth(upperPath, test_folder_name, folder_suffix, file1 = "test1", file2 = "test2", sigma = 39):
	"""
	upperPath: root folder for the detection and matching results; sub-root folder: 'GaussianWindowOnAWhole'(Default)
	test_folder_name: name of test folder
	folder_suffix: suffix to the test folder specifying what kind of feature algorithm used;
	"""
	testPatches = []
	matchesFound = []
	groundTruth = []

	# load testPatches
	listOfTestPatches = saveLoadPatch.loadPatchMatches("{upperPath}/{folderToSave}/{testFolder}/DistinguishablePatch_{folder}_{file}_simga{i}_GaussianWindowOnAWhole.csv".format(
		upperPath = upperPath,
		folderToSave = "GaussianWindowOnAWhole", 
		testFolder = test_folder_name +folder_suffix, 
		folder = test_folder_name, 
		file = file1, 
		i = sigma))
	for i in range(0, len(listOfTestPatches)):
		testPatches.append(listOfTestPatches[i][0])

	# load matchesFound
	listOfTestPatches = saveLoadPatch.loadPatchMatches("{upperPath}/{folderToSave}/{testFolder}/GoodMatches_{test_folder_name}_test1_test2_simga39_shiftBy0.5_useGaussianWindow_True_5levels.csv".format(
		upperPath = upperPath,
		folderToSave = "GaussianWindowOnAWhole", 
		testFolder = test_folder_name +folder_suffix,
		test_folder_name =  test_folder_name
		))
	for i in range(0, len(listOfTestPatches)):
		matchesFound.append(listOfTestPatches[i][0])

	# load groundTruth
	listOfTestPatches = saveLoadPatch.loadPatchMatches("{upperPath}/{folderToSave}/{testFolder}/GroundTruth_{test_folder_name}_{file1}_{file2}_simga{i}_GaussianWindowOnAWhole.csv".format(
		upperPath = upperPath,
		folderToSave = "GaussianWindowOnAWhole", 
		testFolder = test_folder_name +folder_suffix,
		test_folder_name = test_folder_name,
		file1 = file1,
		file2 = file2,
		i = sigma
		))
	for i in range(0, len(listOfTestPatches)):
		groundTruth.append(listOfTestPatches[i][0])

	return testPatches, matchesFound, groundTruth


def generateStatistics(image_db, upperPath, test_folder_name, folder_suffix, file1 = "test1", file2 = "test2", sigma = 39):
	"""
	TODO: generateStatistics will return the recall, precision and # matches for one pair of image
	"""
	return

def generateHists(image_db, upperPath, test_folder_name, folder_suffix, file1 = "test1", file2 = "test2", sigma = 39):
	"""
	Generates the histogram comparison of different features and store in 'hists/' folder under 'test_folder_name+folder_suffix'
	"""
	img_extension = ".jpg"
	img = cv2.imread("{image_db}/{folder}/{image}".format(image_db = image_db, folder = test_folder_name, image = file1+img_extension), 1)
	imgToMatch = cv2.imread("{image_db}/{folder}/{image}".format(image_db = image_db, folder = test_folder_name, image = file2+img_extension), 1)

	testPatches, matchesFound, groundTruth = loadPatchesMatchesGroundtruth(upperPath, test_folder_name, folder_suffix, file1, file2, sigma)
	checkHistogramOfTruthAndMatchesFound(testPatches, groundTruth, matchesFound, img, imgToMatch, \
		"./{upperPath}/GaussianWindowOnAWhole/{test_folder_name}{folder_suffix}/hists".format(\
			upperPath = upperPath, \
			test_folder_name = test_folder_name, \
			folder_suffix = folder_suffix), saveHist = True, displayHist = False)
	return

def compute_sigma(img):
	sigma = img.shape[0]/20
	sigma = sigma + 1 if (sigma % 2 == 0) else sigma # size of patch needs to be odd number
	return sigma

def findAndSaveDistinguishablePatches(image_db, test_folder_name, test_img_name, folder_suffix, sigma = 39, upperPath = "testPatchHSV"):
	"""
	test_img_name: 'test1.jpg' (Default)
	"""
	HSVthresh = 0.5
	HOGthresh = 0.1
	remove_duplicate_thresh_dict ={
		'HSV': HSVthresh,
		'HOG': HOGthresh
	}
	# read the img with proper name and folder
	img = cv2.imread("{image_db}/{folder}/{image}".format(image_db = image_db, folder = test_folder_name, image = test_img_name), 1)
	
	detect_start_time = time.time()
	print "start detecting: ", detect_start_time
	"""Set the FEATURES for comparePatches findDistinguishablePatchesAlgo3 process"""
	comparePatches.FEATURES = copy.deepcopy(utils.ALL_FEATURE_IDS)
	distinguishablePatches = comparePatches.findDistinguishablePatchesAlgo3(img, sigma, remove_duplicate_thresh_dict)
	print "feature detection spent time: ", time.time() - detect_start_time
	
	"""Set the ALL_FEATURE_TO_COMPUTE"""
	all_feature_to_compute_set = set()
	for patch in distinguishablePatches:
		all_feature_to_compute_set = all_feature_to_compute_set.union(set(patch.feature_to_use))
	global ALL_FEATURE_TO_COMPUTE
	ALL_FEATURE_TO_COMPUTE = list(all_feature_to_compute_set)

	# imwrite/save the unique patches
	imgToSave = comparePatches.drawPatchesOnImg(np.copy(img), distinguishablePatches, False, None, (0,0,255), True)
	path = createFolder(upperPath, "GaussianWindowOnAWhole", test_folder_name, folder_suffix)
	# # imwrite the image with unique patches, including the parameters
	# cv2.imwrite("{path}/UniqueAlgo2_{folder}_{file}_simga{i}_HSVthresh{HSVthresh}_HOGthresh{HOGthresh}_feature_{feature_to_use}.jpg".format( \
	# 	path = path , \
	# 	folder = test_folder_name, \
	# 	file = test_img_name[:test_img_name.find(".")], \
	# 	i = sigma, \
	# 	HSVthresh = HSVthresh, \
	# 	HOGthresh = HOGthresh, \
	# 	feature_to_use = feature_to_use), imgToSave)
	# imwrite the image with unique patches, standard name
	cv2.imwrite("{path}/DistinguishablePatch_{folder}_{file}_simga{i}_GaussianWindowOnAWhole.jpg".format( \
		path = path , \
		folder = test_folder_name, \
		file = test_img_name[:test_img_name.find(".")], \
		i = sigma), imgToSave)
	# save the unique patches coordinates, standard name
	saveLoadPatch.savePatchMatches(distinguishablePatches, 1, \
		"{path}/DistinguishablePatch_{folder}_{file}_simga{i}_GaussianWindowOnAWhole.csv".format( \
			path = path , \
			folder = test_folder_name, \
			file = test_img_name[:test_img_name.find(".")], \
			i = sigma))
	return distinguishablePatches

def findDistinguishablePatchesAndExecuteMatching(image_db, test_folder_name, test1_img_name, test2_img_name, \
	folder_suffix, upperPath = "testMatches", initialize_features = True):
	"""
	image_db: image database folder to read source images from;
	upperPath: root folder for saving the detection/matching results (Default: 'testMatches/'); sub-root folder default: 'GaussianWindowOnAWhole/'
	test_folder_name: name of the test folder containing images of different view points;
	folder_suffix: suffix to the folder to save specifying what kind of feature algorithm used;
	test1_img_name: 'test1.jpg'(Default)
	test2_img_name: 'test2.jpg'(Default)
	"""
	# Check if the test images exist, if not, return
	if(not(os.path.exists("{image_db}/{folder}/{image}".format(image_db = image_db, folder = test_folder_name, image = test1_img_name)) \
		and os.path.exists("{image_db}/{folder}/{image}".format(image_db = image_db, folder = test_folder_name, image = test2_img_name)))):
		print "Test Images does not exist in:", test_folder_name
		return

	sigma = compute_sigma(cv2.imread("{image_db}/{folder}/{image}".format(image_db = image_db, folder = test_folder_name, image = test1_img_name)))
	testPatches = findAndSaveDistinguishablePatches(image_db, test_folder_name, test1_img_name, folder_suffix, sigma, upperPath)
	listOfMatches = testDescriptorPerformance(
		image_db,
		test_folder_name, 
		testPatches, 
		test1_img_name,
		test2_img_name,
		"GaussianWindowOnAWhole",
		True,  
		folder_suffix, 
		sigma,
		upperPath,
		initialize_features)
	matchesFound = []
	for i in range(0, len(listOfMatches)):
		matchesFound.append(listOfMatches[i][0]) # just append the best match
	# imwrite the combined match scene
	cv2.imwrite(createFolder(upperPath, "GaussianWindowOnAWhole", test_folder_name, folder_suffix)+"/_combined_scene_match.jpg",\
		comparePatches.drawMatchesOnImg(\
			cv2.imread("{image_db}/{folder}/{image}".format(image_db = image_db, folder = test_folder_name, image = test1_img_name), 1), \
			cv2.imread("{image_db}/{folder}/{image}".format(image_db = image_db, folder = test_folder_name, image = test2_img_name), 1), \
			testPatches, \
			matchesFound, \
			show = False))

def findDistinguishablePatchesAndExecuteMatchingFromTwoFolders(image_db, test_folder_name1, test_folder_name2, \
	test1_img_name, test2_img_name, \
	folder_suffix, upperPath = "testMatches", initialize_features = True):
	"""
	test1_img_name(including file extension) is from test_folder_name1
	test2_img_name(including file extension) is from test_folder_name2
	"""
	if(not(os.path.exists("{image_db}/{folder}/{image}".format(\
		image_db = image_db, folder = test_folder_name1, image = test1_img_name)) \
		and os.path.exists("{image_db}/{folder}/{image}".format(\
			image_db = image_db, folder = test_folder_name2, image = test2_img_name)))):
		print "Test Images does not exist in:", test_folder_name1, test_folder_name2
		return
	
	img1 = cv2.imread("{image_db}/{folder}/{image}".format(\
				image_db = image_db, folder = test_folder_name1, image = test1_img_name), 1)

	img2 = cv2.imread("{image_db}/{folder}/{image}".format(\
				image_db = image_db, folder = test_folder_name2, image = test2_img_name), 1)
	
	"""create combined_test_folder_name and copy image over"""
	combined_test_folder_name = "{test_folder_name1}_{test_folder_name2}".format(\
		test_folder_name1 = test_folder_name1, test_folder_name2 = test_folder_name2)
	createFolder(".", image_db, combined_test_folder_name, "")
	if (not (os.path.exists("{image_db}/{folder}/{image}".format(\
		image_db = image_db, folder = combined_test_folder_name, image = test1_img_name)))):
		cv2.imwrite("{image_db}/{folder}/{image}".format(\
			image_db = image_db, folder = combined_test_folder_name, image = test1_img_name), img1)
	if (not (os.path.exists("{image_db}/{folder}/{image}".format(\
		image_db = image_db, folder = combined_test_folder_name, image = test2_img_name)))):
		cv2.imwrite("{image_db}/{folder}/{image}".format(\
			image_db = image_db, folder = combined_test_folder_name, image = test2_img_name), img2)

	sigma = compute_sigma(cv2.imread("{image_db}/{folder}/{image}".format(\
		image_db = image_db, folder = combined_test_folder_name, image = test1_img_name)))
	testPatches = findAndSaveDistinguishablePatches(image_db, combined_test_folder_name, test1_img_name, folder_suffix, sigma, upperPath)
	listOfMatches = testDescriptorPerformance(
		image_db,
		combined_test_folder_name, 
		testPatches, 
		test1_img_name,
		test2_img_name,
		"GaussianWindowOnAWhole",
		True,  
		folder_suffix, 
		sigma,
		upperPath,
		initialize_features)
	matchesFound = []
	for i in range(0, len(listOfMatches)):
		matchesFound.append(listOfMatches[i][0]) # just append the best match
	# imwrite the combined match scene
	cv2.imwrite(\
		createFolder(upperPath, "GaussianWindowOnAWhole", combined_test_folder_name, folder_suffix)+"/_combined_scene_match.jpg",\
		comparePatches.drawMatchesOnImg(\
			img1, \
			img2, \
			testPatches, \
			matchesFound, \
			show = False))
	return

def executeMatchingGivenDinstinguishablePatches(image_db, test_folder_name, test1_img_name, test2_img_name, \
	folder_suffix, upperPath = "testMatches", initialize_features = True):
	"""
	image_db: image database folder to read source images from;
	upperPath: root folder for saving the detection/matching results (Default: 'testMatches/'); sub-root folder default: 'GaussianWindowOnAWhole/'
	test_folder_name: name of the test folder containing images of different view points;
	folder_suffix: suffix to the folder to save specifying what kind of feature algorithm used;
	test1_img_name: 'test1.jpg'(Default)
	test2_img_name: 'test2.jpg'(Default)
	"""
	sigma = compute_sigma(cv2.imread("{image_db}/{folder}/{image}".format(
		image_db = image_db, folder = test_folder_name, image = test1_img_name)))
	img = cv2.imread("{image_db}/{folder}/{image}".format(\
		image_db = image_db, folder = test_folder_name, image = test1_img_name), 1)
	
	testPatches = []
	listOfTestPatches = saveLoadPatch.loadPatchMatches(\
		"{upperPath}/{folderToSave}/{testFolder}/DistinguishablePatch_{folder}_{file}_simga{i}_GaussianWindowOnAWhole.csv".format(
		upperPath = upperPath,
		folderToSave = "GaussianWindowOnAWhole", 
		testFolder = test_folder_name +folder_suffix, 
		folder = test_folder_name, 
		file = test1_img_name[:test1_img_name.find(".")], 
		i = sigma), initialize_features = initialize_features)
	for i in range(0, len(listOfTestPatches)):
		testPatches.append(listOfTestPatches[i][0])

	listOfMatches = testDescriptorPerformance(
		image_db,
		test_folder_name, 
		testPatches, 
		test1_img_name,
		test2_img_name,
		"GaussianWindowOnAWhole",
		True,  
		folder_suffix, 
		sigma,
		upperPath,
		initialize_features)

	matchesFound = []
	for i in range(0, len(listOfMatches)):
		matchesFound.append(listOfMatches[i][0]) # just append the best match
	# imwrite the combined match scene
	cv2.imwrite(createFolder(upperPath, "GaussianWindowOnAWhole", test_folder_name, folder_suffix)+"/_combined_scene_match.jpg",\
		comparePatches.drawMatchesOnImg(\
			cv2.imread("{image_db}/{folder}/{image}".format(image_db = image_db, folder = test_folder_name, image = test1_img_name), 1), \
			cv2.imread("{image_db}/{folder}/{image}".format(image_db = image_db, folder = test_folder_name, image = test2_img_name), 1), \
			testPatches, \
			matchesFound, \
			show = False))

def executeMatchingGivenDinstinguishablePatchesFromTwoFolders(image_db, test1_folder_name, test2_folder_name, \
	test1_img_name, test2_img_name, folder_suffix, upperPath = "testLabeling", initialize_features = True):
	"""
	image_db: image database folder to read source images from;
	upperPath: root folder for saving the detection/matching results (Default: 'testMatches/'); sub-root folder default: 'GaussianWindowOnAWhole/'
	test1_folder_name: the folder containing test1_img_name, and there are already distinguishablePatches 
					   for folders at {upperPath{/GaussianWindowOnAWhole/{test1_folder_name}_{test1_folder_name}_{folder_suffix}
	test2_folder_name: the folder containing test2_img_name
	folder_suffix: suffix to the folder to save specifying what kind of feature algorithm used;
	test1_img_name: 'test1.jpg'(Default)
	test2_img_name: 'test2.jpg'(Default)
	"""
	sigma = compute_sigma(cv2.imread("{image_db}/{folder}/{image}".format(
		image_db = image_db, folder = test1_folder_name, image = test1_img_name)))
	img1 = cv2.imread("{image_db}/{folder}/{image}".format(\
		image_db = image_db, folder = test1_folder_name, image = test1_img_name), 1)
	img2 = cv2.imread("{image_db}/{folder}/{image}".format(\
		image_db = image_db, folder = test2_folder_name, image = test2_img_name), 1)
	
	"""create combined_test_folder_name and copy image over"""
	combined_test_folder_name = "{test1_folder_name}_{test2_folder_name}".format(\
		test1_folder_name = test1_folder_name, test2_folder_name = test2_folder_name)
	createFolder(".", image_db, combined_test_folder_name, "")
	if (not (os.path.exists("{image_db}/{folder}/{image}".format(\
		image_db = image_db, folder = combined_test_folder_name, image = test1_img_name)))):
		cv2.imwrite("{image_db}/{folder}/{image}".format(\
			image_db = image_db, folder = combined_test_folder_name, image = test1_img_name), img1)
	if (not (os.path.exists("{image_db}/{folder}/{image}".format(\
		image_db = image_db, folder = combined_test_folder_name, image = test2_img_name)))):
		cv2.imwrite("{image_db}/{folder}/{image}".format(\
			image_db = image_db, folder = combined_test_folder_name, image = test2_img_name), img2)

	testPatches = []
	listOfTestPatches = saveLoadPatch.loadUniquePatchesWithFeatureSet(\
		"{upperPath}/{folderToSave}/{testFolder}/DistinguishablePatch_{folder}_{file}_simga{i}_GaussianWindowOnAWhole.csv".format(
		upperPath = upperPath,
		folderToSave = "GaussianWindowOnAWhole", 
		testFolder = test1_folder_name+ "_" + test1_folder_name +folder_suffix, 
		folder = test1_folder_name + "_" + test1_folder_name, 
		file = test1_img_name[:test1_img_name.find(".")], 
		i = sigma))
	for i in range(0, len(listOfTestPatches)):
		testPatches.append(listOfTestPatches[i])

	# comparePatches.drawPatchesOnImg(np.copy(img1), testPatches, mark_sequence = True)

	"""copy over the distinguishablePatches"""
	createFolder(upperPath, "GaussianWindowOnAWhole", combined_test_folder_name, folder_suffix)
	img_with_distinguishablePatches = cv2.imread(\
		"{upperPath}/{folderToSave}/{testFolder}/DistinguishablePatch_{folder}_{file}_simga{i}_GaussianWindowOnAWhole.jpg".format(
		upperPath = upperPath,
		folderToSave = "GaussianWindowOnAWhole", 
		testFolder = test1_folder_name+ "_" + test1_folder_name +folder_suffix, 
		folder = test1_folder_name + "_" + test1_folder_name, 
		file = test1_img_name[:test1_img_name.find(".")], 
		i = sigma), 1)

	cv2.imwrite( \
		"{upperPath}/GaussianWindowOnAWhole/{test_folder}/DistinguishablePatch_{folder1}_{folder2}_{file}_simga{i}_GaussianWindowOnAWhole.jpg".format( \
			upperPath = upperPath , \
			test_folder = test1_folder_name+ "_" + test2_folder_name +folder_suffix, \
			folder1 = test1_folder_name, \
			folder2 = test2_folder_name, \
			file = test1_img_name[:test1_img_name.find(".")], \
			i = sigma), img_with_distinguishablePatches)
	
	saveLoadPatch.savePatchMatches(testPatches, 1, \
		"{upperPath}/GaussianWindowOnAWhole/{test_folder}/DistinguishablePatch_{folder1}_{folder2}_{file}_simga{i}_GaussianWindowOnAWhole.csv".format( \
			upperPath = upperPath, \
			test_folder = test1_folder_name+ "_" + test2_folder_name +folder_suffix, \
			folder1 = test1_folder_name, \
			folder2 = test2_folder_name, \
			file = test1_img_name[:test1_img_name.find(".")], \
			i = sigma))

	listOfMatches = testDescriptorPerformance(
		image_db,
		combined_test_folder_name, 
		testPatches, 
		test1_img_name,
		test2_img_name,
		"GaussianWindowOnAWhole",
		True,  
		folder_suffix, 
		sigma,
		upperPath,
		initialize_features)

	matchesFound = []
	for i in range(0, len(listOfMatches)):
		matchesFound.append(listOfMatches[i][0]) # just append the best match
	# imwrite the combined match scene
	cv2.imwrite(createFolder(upperPath, "GaussianWindowOnAWhole", combined_test_folder_name, folder_suffix)+"/_combined_scene_match.jpg",\
		comparePatches.drawMatchesOnImg(\
			img1, \
			img2, \
			testPatches, \
			matchesFound, \
			show = False))

"""
TODO: complete populateFeatureMatchingStatistics after the Image DB is found
"""
def populateFeatureMatchingStatistics(image_db, test_folder_name, test1_img_name, test2_img_name, \
	folder_suffix, upperPath = "testPatchHSV"):
	sigma = 39
	level = 5
	testPatches = []
	groundTruth = []
	matchesFound = []

	# read testPatches
	listOfTestPatches = saveLoadPatch.loadUniquePatchesWithFeatureSet(\
		"{upperPath}/{folderToSave}/{testFolder}/DistinguishablePatch_{folder}_{file}_simga{i}_GaussianWindowOnAWhole.csv".format(
		upperPath = upperPath,
		folderToSave = "GaussianWindowOnAWhole", 
		testFolder = test_folder_name +folder_suffix, 
		folder = test_folder_name, 
		file = test1_img_name[:test1_img_name.find(".")], 
		i = sigma))
	for i in range(0, len(listOfTestPatches)):
		testPatches.append(listOfTestPatches[i])
		print listOfTestPatches[i].is_low_response

	# read matchesFound
	testPatchMatches = saveLoadPatch.loadPatchMatches(\
		"{upperPath}/{folderToSave}/{testFolder}/GoodMatches_{folder}_{file1}_{file2}_simga{i}_shiftBy{step}_useGaussianWindow_{tf}_5levels.csv".format(\
		upperPath = upperPath,
		folderToSave = "GaussianWindowOnAWhole", 
		testFolder = test_folder_name +folder_suffix,  
		folder = test_folder_name, 
		file1 = test1_img_name[:test1_img_name.find(".")], 
		file2 = test2_img_name[:test2_img_name.find(".")], 
		i = sigma, 
		step = 0.5, 
		tf = True))
	for i in range(0, len(testPatchMatches)):
		matchesFound.append(testPatchMatches[i][0])
	
	print "read groundTruth"
	# read groundTruth
	listOfGroundTruth = saveLoadPatch.loadPatchMatches(\
		"{upperPath}/{folderToSave}/{testFolder}/GroundTruth_{folder}_{file1}_{file2}_simga{i}_GaussianWindowOnAWhole.csv".format(
		upperPath = upperPath,
		folderToSave = "GaussianWindowOnAWhole", 
		testFolder = test_folder_name +folder_suffix, 
		folder = test_folder_name, 
		file1 = test1_img_name[:test1_img_name.find(".")], 
		file2 = test2_img_name[:test2_img_name.find(".")], 
		i = sigma))
	for i in range(0, len(listOfGroundTruth)):
		groundTruth.append(listOfGroundTruth[i][0])

	correct_color = (0,0,255)
	wrong_color = (255,0,0)
	custom_colors = []
	for i in range(0, len(groundTruth)):
		if (utils.isGoodMatch(matchesFound[i], groundTruth[i])):
			custom_colors.append(correct_color)
		else:
			custom_colors.append(wrong_color)
		### DEBUGGING ###
		if (i == 13):
			print matchesFound[i].x, matchesFound[i].y, matchesFound[i].size 
			print groundTruth[i].x, groundTruth[i].y
	print custom_colors
	# read test1_img_name to img
	img = cv2.imread("{image_db}/{folder}/{image}".format(\
		image_db = image_db, folder = test_folder_name, image = test1_img_name), 1)
	# read test2_img_name to imgToMatch
	imgToMatch = cv2.imread("{image_db}/{folder}/{image}".format(\
		image_db = image_db, folder = test_folder_name, image = test2_img_name), 1)

	distinguished_combined_scene_match = comparePatches.drawMatchesOnImg(np.copy(img), np.copy(imgToMatch), testPatches, matchesFound, \
		show = True, custom_colors = custom_colors)

	# comparePatches.drawMatchesOnImg(np.copy(img), np.copy(imgToMatch), testPatches, groundTruth, \
	# 	show = True)

	cv2.imwrite(createFolder(upperPath, "GaussianWindowOnAWhole", test_folder_name, folder_suffix)+"/_combined_scene_match_distinguished.jpg",\
		distinguished_combined_scene_match)
	
	raise ValueError ("purpose stop for drawing using different color")

	# plot the statistics
	checkHistogramOfTruthAndMatchesFound(testPatches, groundTruth, matchesFound, img, imgToMatch, \
		"./{upperPath}/{folderToSave}/{testFolder}/hists".format(\
			upperPath = upperPath,\
			folderToSave = "GaussianWindowOnAWhole", \
			testFolder = test_folder_name +folder_suffix), True, True)


def mannalPruning(image_db, folder_suffix, upperPath = "testPatchHSV"):
	sigma = 39
	level = 5
	test_patches = []
	ground_truth = []
	matches_found = []

	test_folder_name = raw_input("please input test_folder_name:")
	test1_img_name = raw_input("please input test1_img_name(with extension .jpg):")
	test2_img_name = raw_input("please input test2_img_name(with extension .jpg):")

	# read test_patches
	list_of_test_patches = saveLoadPatch.loadUniquePatchesWithFeatureSet(\
		"{upperPath}/{folderToSave}/{testFolder}/DistinguishablePatch_{folder}_{file}_simga{i}_GaussianWindowOnAWhole.csv".format(
		upperPath = upperPath,
		folderToSave = "GaussianWindowOnAWhole", 
		testFolder = test_folder_name +folder_suffix, 
		folder = test_folder_name, 
		file = test1_img_name[:test1_img_name.find(".")], 
		i = sigma))
	for i in range(0, len(list_of_test_patches)):
		test_patches.append(list_of_test_patches[i])
		print list_of_test_patches[i].is_low_response

	# read matches_found
	test_patch_matches = saveLoadPatch.loadPatchMatches(\
		"{upperPath}/{folderToSave}/{testFolder}/GoodMatches_{folder}_{file1}_{file2}_simga{i}_shiftBy{step}_useGaussianWindow_{tf}_5levels.csv".format(\
		upperPath = upperPath,
		folderToSave = "GaussianWindowOnAWhole", 
		testFolder = test_folder_name +folder_suffix,  
		folder = test_folder_name, 
		file1 = test1_img_name[:test1_img_name.find(".")], 
		file2 = test2_img_name[:test2_img_name.find(".")], 
		i = sigma, 
		step = 0.5, 
		tf = True))
	for i in range(0, len(test_patch_matches)):
		matches_found.append(test_patch_matches[i][0])

	# read test1_img_name to img
	img = cv2.imread("{image_db}/{folder}/{image}".format(\
		image_db = image_db, folder = test_folder_name, image = test1_img_name), 1)
	# read test2_img_name to imgToMatch
	imgToMatch = cv2.imread("{image_db}/{folder}/{image}".format(\
		image_db = image_db, folder = test_folder_name, image = test2_img_name), 1)

	comparePatches.drawMatchesOnImg(np.copy(img), np.copy(imgToMatch), test_patches, matches_found, \
		show = True, draw_rect = False)

	matches_to_keep = []
	index = raw_input("please specify the good match indexes(hit enter to exit):")
	while(index != ""):
		matches_to_keep.append(int(index))
		index = raw_input("please specify the good match indexes(hit enter to exit):")

	print matches_to_keep
		
	test_patches_keep = []
	matches_found_keep = []
	for i in matches_to_keep:
		test_patches_keep.append(test_patches[i])
		matches_found[i].setFeatureToUse(test_patches[i].feature_to_use)
		matches_found[i].setFeatureWeights(test_patches[i].feature_weights)
		matches_found[i].setIsLowResponse(test_patches[i].is_low_response)
		matches_found_keep.append(matches_found[i])

	pruned_match = comparePatches.drawMatchesOnImg(np.copy(img), np.copy(imgToMatch), test_patches_keep, matches_found_keep, \
		show = True)
	cv2.imwrite(createFolder(upperPath, "GaussianWindowOnAWhole", test_folder_name, folder_suffix)+"/_combined_scene_match_pruned.jpg",\
		pruned_match)

	saveLoadPatch.savePatchMatches(matches_found_keep, 1, \
		"{path}/unique_patch_matches_pruned.csv".format( \
			path = createFolder(upperPath, "GaussianWindowOnAWhole", test_folder_name, folder_suffix), \
			))

	return	

def checkActualMatchDistance(test_patch, match_found, img, img_to_match, metric_func = comparePatches.Jensen_Shannon_Divergence):
	"""old descriptor checking HSV and HOG hists"""
	test_patch.computeHSVHistogram(cv2.cvtColor(img.astype(np.float32), cv2.COLOR_BGR2HSV), True, True)
	match_found.computeHSVHistogram(cv2.cvtColor(img_to_match.astype(np.float32), cv2.COLOR_BGR2HSV), True, True)
	return getHSVSeperateHistAvgl2Distance(test_patch, match_found, metric_func)

def checkTestLabelingNumberMatches(image_db, test1_folder_name, test2_folder_name, test1_img_name, test2_img_name, \
	folder_suffix, ground_truth_folder_suffix, upperPath = "testLabeling", tight_criteria = "intersection"):
	"""
	test1_folder_name: the testset containing incoming test image
	test2_folder_name: the testset from the constrained database
	test1_img_name: image name of the incoming test image
	test2_img_name: image name of the imaged matched in the constrained database
	folder_suffix: _descriptor_based_point_01_Harris_from_two_folder
	tight_criteria: default is exactly the same set of unique features
	"""
	sigma = 39
	level = 5
	testPatches = []
	groundTruth = []
	matchesFound = []
	test_folder_name = test1_folder_name + "_" + test2_folder_name

	img = cv2.imread("{image_db}/{test1_folder_name}/{test1_img_name}".format(\
		image_db = image_db,\
		test1_folder_name = test1_folder_name,\
		test1_img_name = test1_img_name), 1)

	img_to_match = cv2.imread("{image_db}/{test2_folder_name}/{test2_img_name}".format(\
		image_db = image_db, \
		test2_folder_name = test2_folder_name, \
		test2_img_name = test2_img_name), 1)

	img_origin = cv2.imread("{image_db}/{test2_folder_name}/{test2_img_name}".format(\
		image_db = image_db, \
		test2_folder_name = test2_folder_name, \
		test2_img_name = "test1.jpg"), 1) # original base image used to construct the database

	# read testPatches
	listOfTestPatches = saveLoadPatch.loadUniquePatchesWithFeatureSet(\
		"{upperPath}/{folderToSave}/{testFolder}/DistinguishablePatch_{folder}_{file}_simga{i}_GaussianWindowOnAWhole.csv".format(
		upperPath = upperPath,
		folderToSave = "GaussianWindowOnAWhole", 
		testFolder = test_folder_name +folder_suffix, 
		folder = test_folder_name, 
		file = test1_img_name[:test1_img_name.find(".")], 
		i = sigma))
	for i in range(0, len(listOfTestPatches)):
		testPatches.append(listOfTestPatches[i])
		# print listOfTestPatches[i].is_low_response

	# read matchesFound
	testPatchMatches = saveLoadPatch.loadPatchMatches(\
		"{upperPath}/{folderToSave}/{testFolder}/GoodMatches_{folder}_{file1}_{file2}_simga{i}_shiftBy{step}_useGaussianWindow_{tf}_5levels.csv".format(\
		upperPath = upperPath,
		folderToSave = "GaussianWindowOnAWhole", 
		testFolder = test_folder_name +folder_suffix,  
		folder = test_folder_name, 
		file1 = test1_img_name[:test1_img_name.find(".")], 
		file2 = test2_img_name[:test2_img_name.find(".")], 
		i = sigma, 
		step = 0.5, 
		tf = True))
	for i in range(0, len(testPatchMatches)):
		matchesFound.append(testPatchMatches[i][0])

	assert (len(matchesFound) == len(testPatches)), \
	"In checkTestLabelingNumberMatches: matchesFound should have the same length as testPatches"

	# assign matchesFound attributes of unique patches related such as feature_to_use and is_low_response
	for i in range(0, len(matchesFound)):
		matchesFound[i].setFeatureToUse(testPatches[i].feature_to_use)
		matchesFound[i].setFeatureWeights(testPatches[i].feature_weights)
		matchesFound[i].setIsLowResponse(testPatches[i].is_low_response)
	
	num_correct_matches = 0
	num_location_matches = 0

	# read groundTruth
	listOfGroundTruth = saveLoadPatch.loadUniquePatchesWithFeatureSet(\
		"{upperPath}/{folderToSave}/{testFolder}/unique_patch_matches_pruned.csv".format(
		upperPath = "testAlgo3",
		folderToSave = "GaussianWindowOnAWhole", 
		testFolder = test2_folder_name + ground_truth_folder_suffix
		))
	for i in range(0, len(listOfGroundTruth)):
		groundTruth.append(listOfGroundTruth[i])

	"""get the maximum dissimilarty of groundTruth"""
	min_dissimilarities = []
	listOfOriginUniquePatches = saveLoadPatch.loadUniquePatchesWithFeatureSet(\
		"{upperPath}/{folderToSave}/{testFolder}/DistinguishablePatch_{testset}_test1_simga39_GaussianWindowOnAWhole.csv".format(
		upperPath = "testAlgo3",
		folderToSave = "GaussianWindowOnAWhole", 
		testFolder = test2_folder_name + ground_truth_folder_suffix,
		testset = test2_folder_name
		))
	for i in range(0, len(groundTruth)):
		for j in range(0, len(listOfOriginUniquePatches)):
			if (groundTruth[i].feature_to_use == listOfOriginUniquePatches[j].feature_to_use):
				min_dissimilarities.append(\
					checkActualMatchDistance(listOfOriginUniquePatches[j], groundTruth[i], img_origin, img_to_match))

	assert len(min_dissimilarities) == len(groundTruth), \
	"length for array of minimum dissimilarty of original match should be the same as the number of pruned groundTruth"


	"""cap the maximum number of tight matches to number of feature patches in the database"""
	matched_ground_truth = {}
	location_matched_ground_truth = {}

	MAX_MATCH_DISSIMILARITY_JENSEN_SHANNON_DIVERGENCE = 0.41

	for i in range(0, len(matchesFound)):
		this_match_found_is_location_match_flag = False # one match patch should correspond to one unique patch in the database
		this_match_found_is_tight_match_flag = False # one match patch should correspond to one unique patch in the database
		for j in range(0, len(groundTruth)):

			if (utils.isGoodMatch(matchesFound[i], groundTruth[j], patch_neighbour_hood = 5)):
				# print "\nmatchesFound[{i}]".format(i = i), "found good location match with groundTruth[{j}]".format(j = j)
				if (not this_match_found_is_location_match_flag and \
					(not j in location_matched_ground_truth)):
					num_location_matches += 1
					this_match_found_is_location_match_flag = True
					location_matched_ground_truth[j] = True
				features_match_found = matchesFound[i].feature_to_use
				features_ground_truth = groundTruth[j].feature_to_use
				
				"""if matchFound has unique features in common with database groundTruth, real match found"""
				good_match = False
				
				if (tight_criteria is None or tight_criteria == "exact"):
					if (set(features_match_found) == set(features_ground_truth) and \
						matchesFound[i].is_low_response == groundTruth[j].is_low_response):
						good_match = True

				elif (tight_criteria == "<="):
					if (set(features_match_found) <= set(features_ground_truth) and \
						matchesFound[i].is_low_response == groundTruth[j].is_low_response):
						good_match = True
				
				elif (tight_criteria == "<= or >="):
					if ((set(features_match_found) <= set(features_ground_truth) or \
						set(features_match_found) >= set(features_ground_truth)) and \
						matchesFound[i].is_low_response == groundTruth[j].is_low_response):
						good_match = True
				
				elif (tight_criteria == "intersection"):
					if ((not set(features_match_found).isdisjoint(set(features_ground_truth))) and \
						matchesFound[i].is_low_response == groundTruth[j].is_low_response):
						good_match = True
				
				if (good_match and \
					(not this_match_found_is_tight_match_flag) and \
					(not j in matched_ground_truth)):
					print "matchesFound[{i}]".format(i = i ), \
					"features_match_found:", features_match_found, " is_low_response? ", matchesFound[i].is_low_response
					print "groundTruth[{j}]".format(j = j), \
					"features_ground_truth:", features_ground_truth, "is_low_response? ", groundTruth[j].is_low_response
					print "real match found at matchesFound[{i}]".format(i = i), "with unique feature set:", features_ground_truth
					actual_match_dist = checkActualMatchDistance(testPatches[i], matchesFound[i], img, img_to_match)
					print "actual match score using homogeneous descriptor:", actual_match_dist
					print "original dissimilarty: ", min_dissimilarities[j]
					if (actual_match_dist < MAX_MATCH_DISSIMILARITY_JENSEN_SHANNON_DIVERGENCE):
					# if (actual_match_dist < 1.1 * min_dissimilarities[j]):
						num_correct_matches += 1
						this_match_found_is_tight_match_flag = True
						matched_ground_truth[j] = True

	print "number of location matches between testLabeling image:", test1_img_name, " from ", test1_folder_name, \
	" and ", test2_img_name, " from ", test2_folder_name, " = ", num_location_matches

	print "number of correct matches between testLabeling image:", test1_img_name, " from ", test1_folder_name, \
	" and ", test2_img_name, " from ", test2_folder_name, " = ", num_correct_matches
	
	return num_correct_matches, num_location_matches

def populateCheckTestLabelingNumMatches(plot_folder_name, folder_suffix, ground_truth_folder_suffix, \
	tight_criteria = "intersection", save = False, show = True):
	path = createFolder(".", "testLabelingPlots", plot_folder_name, "")
	ALL_SCENE_SETS = [2, 3, 5, 7, 9, 10, 12, 13, 19, 23]
	testset_flower_ids = [2, 3, 5, 7, 9, 10, 12, 13, 19, 23]
	incoming_test_ids = [2, 3, 5, 7, 9, 10, 12, 13, 19, 23]

	for i in range(0, len(incoming_test_ids)):
		num_location_matches_arr = [] # arr of location matches
		num_correct_matches_arr = [] # arr of tight matches
		testset_names = [] # hold the xtick names to show
		
		for j in range(0, len(testset_flower_ids)):
			num_correct_matches, num_location_matches = checkTestLabelingNumberMatches(\
				"images", \
				"testset_flower{i}".format(i = incoming_test_ids[i]), \
				"testset_flower{j}".format(j = testset_flower_ids[j]), "test2.jpg", "test3.jpg", \
				folder_suffix, \
				ground_truth_folder_suffix, \
				upperPath = "testLabeling", \
				tight_criteria = tight_criteria)
			num_correct_matches_arr.append(num_correct_matches)
			num_location_matches_arr.append(num_location_matches)
			testset_names.append("testset_flower{testset_count}".format(\
				testset_count = ALL_SCENE_SETS.index(testset_flower_ids[j]) + 1))

		test_image_name = "testset_flower{test_image_index}".format(\
			test_image_index = ALL_SCENE_SETS.index(incoming_test_ids[i]) + 1)
		title = "Test incoming image from scene {scene}".format(\
			scene = test_image_name)
		
		"""plotting matches against database sets"""
		fig, ax = plt.subplots()
		fig.set_figwidth(10)
		plt.margins(0.05) # avoid the markers to be cut off at the border of the plotting area
		plt.subplots_adjust(bottom=0.2) # make sure bottom labels could be seen fully
		# plt.subplots_adjust(right=0.8)
		box = ax.get_position()
		ax.set_position([box.x0 - 0.05, box.y0, box.width * 0.8, box.height])
		plt.xticks(range(0, len(testset_names)), testset_names, rotation='45') # rotate bottom labels
		ax.yaxis.grid(True)
		for tick in ax.xaxis.get_majorticklabels():
		    tick.set_horizontalalignment("right") # make sure the bottom label is precisely at the position
		plt.plot(range(0, len(num_correct_matches_arr)), num_correct_matches_arr, \
			marker='o', markersize = 10, linestyle='--', label = "# correct matches")
		# plt.plot(range(0, len(num_location_matches_arr)), num_location_matches_arr, \
			# marker='^', markersize = 10, linestyle='--', label = "location matches")
		ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
		plt.title(title)
		if (save):
			plt.savefig("{path}/Test incoming image from scene {scene} {plot_folder_name}.png".format(\
				path = path, scene = test_image_name, plot_folder_name = plot_folder_name))
		if (show):
			plt.show()
		plt.clf()

	return


def main():
	upperPath = raw_input("please input upperPath for saving the results:")
	folder_suffix = raw_input("please input folder_suffix:")
	test_folder_name = raw_input("please input test_folder_name which contains the images:")

	"""Necessary step to initialise the features constructed"""
	utils.loadGeneratedFeatureParadigm()
	
	"""Test full automatic algorithm"""
	findDistinguishablePatchesAndExecuteMatching("images", test_folder_name , "test1.jpg", "test3.jpg", \
		folder_suffix, upperPath = upperPath)
	# findDistinguishablePatchesAndExecuteMatchingFromTwoFolders("images", "testset_flower2", "testset_flower2", \
		# "test2.jpg", "test3.jpg", \
		# "_descriptor_based", upperPath = "testLabeling", initialize_features = False)
	# executeMatchingGivenDinstinguishablePatchesFromTwoFolders("images", "testset_flower2", "testset_flower3", \
	# "test2.jpg", "test3.jpg", folder_suffix, upperPath = "testLabeling", initialize_features = False)
	
	"""After Matching Statistics/Pruning"""
	# folder_suffix = raw_input("please input folder_suffix used by previously matching from two testfolders:")
	# ground_truth_folder_suffix = raw_input("please input folder_suffix used by previously image labelling database construction:")
	# populateCheckTestLabelingNumMatches(plot_folder_name, folder_suffix, ground_truth_folder_suffix,\
	 # tight_criteria = "intersection", save = True, show = False)

	# mannalPruning("images", folder_suffix, upperPath = "testAlgo3")

	return

if __name__ == "__main__":
	main()