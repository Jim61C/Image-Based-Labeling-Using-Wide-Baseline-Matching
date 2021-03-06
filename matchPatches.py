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
	'HOG':1.0
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
	# color_distances = np.zeros(len(matchPatches))
	# hog_distances = np.zeros(len(matchPatches))
	
	# if('HSV' in patchToMatch.feature_to_use):
	# 	if(useSeperateHSVHists):
	# 		color_distances = getHSVSeperateOverAllDistances(patchToMatch, matchPatches, metricFunc)
	# 	else:
	# 		# color_distances = np.zeros(len(matchPatches))
	# 		# # Compute C for earthMoverHatDistance: size of matchPatches[i] might be different but the length of HSVhist is the same: 4096 for HSV or 256 for HS
	# 		# if(metric == "earthMoverHatDistance"):
	# 		# 	histLen = len(patchToMatch.HSVHist) # poll the length of the flattened HSVHist
	# 		# 	C = np.ones(shape = (histLen, histLen))
	# 		# 	rows = np.arange(0,histLen).reshape((histLen, 1))
	# 		# 	rows = np.repeat(rows, histLen, axis = 1)
	# 		# 	cols = np.arange(0, histLen).reshape((1,histLen))
	# 		# 	cols = np.repeat(cols, histLen, axis = 0)
	# 		# 	C = C + abs(rows - cols)

	# 		# Compute color_distances[i]
	# 		for i in range(0, len(color_distances)):
	# 			if(histToUse == "RGB"):
	# 				individualScores = np.zeros(len(patchToMatch.RGBHistArr))
	# 				for j in range(0, len(patchToMatch.RGBHistArr)):
	# 					oneHistScore = metricFunc(patchToMatch.RGBHistArr[j], matchPatches[i].RGBHistArr[j])
	# 					individualScores[j] = oneHistScore
	# 			elif(histToUse == "HSV"):
	# 				individualScores = np.zeros(len(patchToMatch.HSVHistArr))
	# 				for j in range(0, len(patchToMatch.HSVHistArr)):
	# 					oneHistScore = metricFunc(patchToMatch.HSVHistArr[j], matchPatches[i].HSVHistArr[j])
	# 					individualScores[j] = oneHistScore
	# 			color_distances[i] = np.linalg.norm(individualScores, 2)

	# if('HOG' in patchToMatch.feature_to_use):
	# 	# hog_distances = np.zeros(len(matchPatches))
	# 	for i in range(0, len(hog_distances)):
	# 		# TODO: if use earthMover for HOG, then we should use comparePatches.earthMoverHatDistanceForHOG with the correct C matrix
	# 		hog_distances[i] = getHistArrl2Distance(patchToMatch.HOGArr, matchPatches[i].HOGArr, metricFunc) 
	
	# overall_distances = np.sqrt(\
	# 	(patchToMatch.feature_weights[patchToMatch.feature_to_use.index('HSV')] if ('HSV' in patchToMatch.feature_to_use) else 0.0) * color_distances**2 + \
	# 	(patchToMatch.feature_weights[patchToMatch.feature_to_use.index('HOG')] if ('HOG' in patchToMatch.feature_to_use) else 0.0) * hog_distances**2)
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
	overall_scores = np.zeros(len(matchPatches))
	for i in range(0, len(overall_scores)):
		score_vector = []
		for this_feature_id in patchToMatch.feature_to_use:
			score_vector.append(matchPatches[i].getFeatureObject(this_feature_id).score)
		score_vector = np.asarray(score_vector)
		assert (len(score_vector) == len(patchToMatch.feature_weights)), "In findBestMatches: length of distance_vector must be the same as test patch feature_weights"
		overall_scores[i] = np.sum(np.multiply(score_vector, patchToMatch.feature_weights))
	
	if (patchToMatch.is_low_response):
		sortedIndex = np.argsort(overall_scores) # get the lowest scores as possible
	else:
		sortedIndex = np.argsort(overall_scores)[::-1] # get the highest scores as possible

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

def testDescriptorPerformancePyramidWorker(testPatches, img, img_gray,imgToMatch, imgToMatch_gray, sigma, testFolderName,  patchStep = 0.5, useGaussianWindow = True):
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
			newPatch = comparePatches.Patch(testPatches[j].x/(2**i), testPatches[j].y/(2**i), newSize)
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
		thisTestPatches = testPatchesPyd[i]

		if(i == level):
			"""if top level of pyramid, run original matching algorithm"""
			thisListOfMatches = testDescriptorPerformanceWorker(thisTestPatches, \
				thisImg, thisImgGray , \
				thisImgToMatch, thisImgToMatchGray, \
				sigma/(2**i), testFolderName, NUM_PATCH_SIZE_GAUSSIAN**(i+1)) # each gaussian 125 best matches
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
							thisImg.shape))
					# compute Histogram for new potential match patch
					for this_feature in ALL_FEATURE_TO_COMPUTE:
						newPatch.getFeatureObject(this_feature).computeFeature(thisImgToMatch)
						newPatch.getFeatureObject(this_feature).computeScore()
					# append the new potential match patch for current particular one test patch
					matchesFoundNextLevelOnePatch.append(newPatch)
				matchesFoundNextLevel.append(matchesFoundNextLevelOnePatch)

			# rematch
			rematches = []
			for j in range(0, len(thisTestPatches)):
				thisPatchToMatch = thisTestPatches[j]
				for this_feature in thisPatchToMatch.feature_to_use:
					thisPatchToMatch.getFeatureObject(this_feature).computeFeature(thisImg)
					thisPatchToMatch.getFeatureObject(this_feature).computeScore()
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


def testDescriptorPerformanceWorker(testPatches, img, img_gray,imgToMatch, imgToMatch_gray, sigma, testFolderName, k = 1, patchStep = 0.5, useGaussianWindow = True, metricFunc = comparePatches.Jensen_Shannon_Divergence):
	#Extract match patches
	matchPatches_origin = comparePatches.extractPatches(imgToMatch, sigma, patchStep)
	print "length of matchPatches:", len(matchPatches_origin)
	# scale = 8 # window pixel window for up and down scaling
	gaussianScaleFactor = 1.2
	patchesArr = []
	patchesArr.append(matchPatches_origin)

	for level in range(-2, 3):
		if(level != 0):
			# patchesArr.append(comparePatches.extractPatches(imgToMatch, sigma + scale * level, patchStep))
			print "new sigma:", getGaussianScale(sigma, gaussianScaleFactor, level)
			patchesArr.append(comparePatches.extractPatches(imgToMatch, getGaussianScale(sigma, gaussianScaleFactor, level), patchStep))
	for i in range(0, len(patchesArr)):
		print "len(patchesArr[{i}]):".format(i = i),len(patchesArr[i])

	#Compute the Features' Hist of the potential matchPatches
	for index in range(0, len(patchesArr)):
		matchPatches = patchesArr[index]
		for i in range(0, len(matchPatches)):
			"""compute all features needed for the potential match patches"""
			for this_feature in ALL_FEATURE_TO_COMPUTE:
				matchPatches[i].getFeatureObject(this_feature).computeFeature(imgToMatch)
				matchPatches[i].getFeatureObject(this_feature).computeScore()

		"""-------For logging purpose only: One PatchesArr done!------"""
		print "compute match patches' features of patchArr[{index}] in {f} done".format(i =i, index = index , f = testFolderName)

	testPatchMatches = []
	#loop over all test patches
	for this_test_patch in testPatches:
		"""compute only the features needed for test patches (not needed for full algorithm, since it will be done during detection phase)"""
		print "feature to use for this_test_patch:", this_test_patch.feature_to_use
		for this_feature in this_test_patch.feature_to_use:
			this_test_patch.getFeatureObject(this_feature).computeFeature(img)
			this_test_patch.getFeatureObject(this_feature).computeScore()
		bestMatches = testFindOnePatchMatch(this_test_patch, patchesArr, k, metricFunc) # will return the best matches (in an array) of the testPatch1
		testPatchMatches.append(bestMatches)

	return testPatchMatches # return a list of [list of good matches for one test patch]



def testDescriptorPerformance(image_db, folderName,testPatches, imgName,imgToMatchName,folderToSave,useGaussianWindow, suffix = "", sigma = 39, upperPath = "testPatchHSV"):
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
	# testPatchMatches = testDescriptorPerformanceWorker(testPatches, img, img_gray,imgToMatch, imgToMatch_gray, sigma, folderName, patchStep)
	testPatchMatches = testDescriptorPerformancePyramidWorker(testPatches, img, img_gray,imgToMatch, imgToMatch_gray, sigma, folderName, patchStep)
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
		testPatches[i].computeHSVHistogram(img, True, True)
		groundTruth[i].computeHSVHistogram(imgToMatch, True, True)
		matchesFound[i].computeHSVHistogram(imgToMatch, True, True)

		testPatches[i].computeHOG(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.int),  True)
		groundTruth[i].computeHOG(cv2.cvtColor(imgToMatch, cv2.COLOR_BGR2GRAY).astype(np.int),  True)
		matchesFound[i].computeHOG(cv2.cvtColor(imgToMatch, cv2.COLOR_BGR2GRAY).astype(np.int), True)

		print "\nfor test patch ", i
		# plotStatistics.plotHOGHistCmp(path, \
		# 	"HOGhistCmp{i}".format(i = i), \
		# 	testPatches[i].HOG, "testPatch{i}".format(i = i), \
		# 	matchesFound[i].HOG, "matchFound{i}".format(i = i),\
		# 	groundTruth[i].HOG, "groundTruth{i}".format(i = i), \
		# 	saveHist, displayHist)

		plotStatistics.plotHSVSeperateHistCmp(path, \
			"HSV Seperate Cmp_testPatch{i}".format(i = i), \
			[testPatches[i].HueHist, testPatches[i].SaturationHist, testPatches[i].ValueHist], "testPatch{i}".format(i = i) , \
			[matchesFound[i].HueHist, matchesFound[i].SaturationHist, matchesFound[i].ValueHist],  "matchFound{i}".format(i = i), \
			[groundTruth[i].HueHist, groundTruth[i].SaturationHist, groundTruth[i].ValueHist], "groundTruth{i}".format(i = i), \
			saveHist, displayHist)

		"""Check the groundTruth and MatchesFound's feature object"""
		for test_patch_feature in testPatches[i].feature_to_use:
			print "testPatches[{i}]".format(i = i), "uses feature:", test_patch_feature
			print "testPatches[{i}]".format(i = i), "computeScore"
			testPatches[i].getFeatureObject(test_patch_feature).computeFeature(img)
			testPatches[i].getFeatureObject(test_patch_feature).computeScore()

			print "groundTruth[{i}]".format(i = i), "computeScore"
			groundTruth[i].getFeatureObject(test_patch_feature).computeFeature(imgToMatch)
			groundTruth[i].getFeatureObject(test_patch_feature).computeScore()

			print "matchesFound[{i}]".format(i = i), "computeScore"
			matchesFound[i].getFeatureObject(test_patch_feature).computeFeature(imgToMatch)
			matchesFound[i].getFeatureObject(test_patch_feature).computeScore()

			"""plot out the computed feature hist for testPatches, groundTruth, matchesFound"""
			plotStatistics.plotOneGivenHist(path,"testPatches[{i}]_{feature}".format(i = i , feature = test_patch_feature), \
				testPatches[i].getFeatureObject(test_patch_feature).hist, save = saveHist, show = displayHist)
			plotStatistics.plotOneGivenHist(path,"groundTruth[{i}]_{feature}".format(i = i , feature = test_patch_feature), \
				groundTruth[i].getFeatureObject(test_patch_feature).hist, save = saveHist, show = displayHist)
			plotStatistics.plotOneGivenHist(path,"matchesFound[{i}]_{feature}".format(i = i, feature = test_patch_feature), \
				matchesFound[i].getFeatureObject(test_patch_feature).hist, save = saveHist, show = displayHist)


			"""plot out this feature model"""
			if(test_patch_feature.find("centre_hog_paradigm_") != -1):
				plotStatistics.plotOneGivenHist(path,"testPatches[{i}]_{feature}_inner_HOG".format(i = i , feature = test_patch_feature), \
				testPatches[i].getFeatureObject(test_patch_feature).inner_HOG, save = saveHist, show = displayHist)
				plotStatistics.plotOneGivenHist(path,"groundTruth[{i}]_{feature}_inner_HOG".format(i = i , feature = test_patch_feature), \
				groundTruth[i].getFeatureObject(test_patch_feature).inner_HOG, save = saveHist, show = displayHist)
				plotStatistics.plotOneGivenHist(path,"matchesFound[{i}]_{feature}_inner_HOG".format(i = i, feature = test_patch_feature), \
				matchesFound[i].getFeatureObject(test_patch_feature).inner_HOG, save = saveHist, show = displayHist)

				plotStatistics.plotOneGivenHist(path,"testPatches[{i}]_{feature}_FEATURE_MODEL_INNER".format(i = i , feature = test_patch_feature), \
				testPatches[i].getFeatureObject(test_patch_feature).FEATURE_MODEL_INNER, save = saveHist, show = displayHist)
				plotStatistics.plotOneGivenHist(path,"testPatches[{i}]_{feature}_FEATURE_MODEL_BORDER".format(i = i , feature = test_patch_feature), \
				testPatches[i].getFeatureObject(test_patch_feature).FEATURE_MODEL_BORDER, save = saveHist, show = displayHist)
			else:
				plotStatistics.plotOneGivenHist(path,"testPatches[{i}]_{feature}_FEATURE_MODEL".format(i = i , feature = test_patch_feature), \
				testPatches[i].getFeatureObject(test_patch_feature).FEATURE_MODEL, save = saveHist, show = displayHist)

			"""print the score of dissimilarityWith"""
			dissimilarity_ground_truth = testPatches[i].getFeatureObject(test_patch_feature).dissimilarityWith(\
				groundTruth[i].getFeatureObject(test_patch_feature))
			dissimilarity_match_found = testPatches[i].getFeatureObject(test_patch_feature).dissimilarityWith(\
				matchesFound[i].getFeatureObject(test_patch_feature))

			print "testPatches[{i}] ".format(i = i), test_patch_feature, \
			" dissimilarity_ground_truth:", dissimilarity_ground_truth, \
			" groundTruth[{i}] actual feature score:".format(i = i), groundTruth[i].getFeatureObject(test_patch_feature).score
			print "testPatches[{i}] ".format(i = i), test_patch_feature, \
			" dissimilarity_match_found:", dissimilarity_match_found, \
			" matchesFound[{i}] actual feature score".format(i = i), matchesFound[i].getFeatureObject(test_patch_feature).score
			print "testPatches[{i}] ".format(i = i), "match found is better? ", dissimilarity_match_found < dissimilarity_ground_truth	

	return

def populate_testset_illuminance1(folder_suffix = "", upperPath = "testPatchHSV"):
	sigma = 39
	testPatches = []
	groundTruth = []
	matchesFound = []
	# for testset testset_illuminance1
	img = cv2.imread("images/testset_illuminance1/test1.jpg", 1)
	imgToMatch = cv2.imread("images/testset_illuminance1/test2.jpg", 1)

	# testPatches.append(comparePatches.Patch(149, 826, sigma)) # test0
	# testPatches.append(comparePatches.Patch(478, 721, sigma)) # test1
	# testPatches.append(comparePatches.Patch(351, 822, sigma)) # test2
	# testPatches.append(comparePatches.Patch(328, 943, sigma)) # test3
	# testPatches.append(comparePatches.Patch(342, 145, sigma)) # test4

	# groundTruth.append(comparePatches.Patch(179, 830, sigma)) # test0
	# groundTruth.append(comparePatches.Patch(501, 728, sigma)) # test1
	# groundTruth.append(comparePatches.Patch(377, 826, sigma)) # test2
	# groundTruth.append(comparePatches.Patch(358, 943, sigma)) # test3
	# groundTruth.append(comparePatches.Patch(360, 165, sigma)) # test4

	# listOfPatches = saveLoadPatch.loadPatchMatches( \
	# "{path}/DistinguishablePatch_{folder}_{file}_simga{i}_GaussianWindowOnAWhole.csv".format( \
	# 	path = upperPath + "/GaussianWindowOnAWhole/" + "testset_illuminance1" + "_eyeballed_unique_patches", \
	# 	folder = "testset_illuminance1", \
	# 	file = "test1", \
	# 	i = sigma))
	# for i in range(0, len(listOfPatches)):
	# 	testPatches.append(listOfPatches[i][0])
	# for this_test_patch in testPatches:
	# 	this_test_patch.setFeatureToUse(MANUAL_FEATURE_TO_USE)
	# 	this_test_patch.setFeatureWeights(np.ones(len(this_test_patch.feature_to_use)))

	for this_feature_set in MANUAL_FEATURE_TO_USE:
		list_of_patches = saveLoadPatch.loadUniquePatchesWithFeatureSet( \
		"{path}/UniquePatches_{feature}_{folder}_{file}_simga{i}_GaussianWindowOnAWhole.csv".format( \
			path = "testUniquePatches/algo3", \
			feature = "_".join(this_feature_set), \
			folder = "testset_illuminance1", \
			file = "test1", \
			i = sigma))
		list_of_patches[0].setFeatureToUse(this_feature_set)
		list_of_patches[0].setFeatureWeights(np.ones(len(list_of_patches[0].feature_to_use)))
		list_of_patches[0].setIsDueToHighResponse() # current test patches are unique all due to a high response under the feature set
		testPatches.append(list_of_patches[0]) # append the best one found

	comparePatches.drawPatchesOnImg(np.copy(img), testPatches, mark_sequence = True)


	
	# listOfPatchMatches = saveLoadPatch.loadPatchMatches(upperPath + "/{folderToSave}/{testFolder}/GoodMatches_{folder}_{file1}_{file2}_simga{i}_shiftBy{step}_useGaussianWindow_{tf}_5levels.csv".format(testFolder = "testset_illuminance1" + folder_suffix, folderToSave = "GaussianWindowOnAWhole", folder = "testset_illuminance1", file1 = "test1", file2 = "test2", i = sigma, step = 0.5, tf = True))
	# for i in range(0, len(listOfPatchMatches)):
	# 	matchesFound.append(listOfPatchMatches[i][0]) # just append the best match
	# comparePatches.drawPatchesOnImg(np.copy(imgToMatch), matchesFound, mark_sequence = True)


	for this_feature_set in MANUAL_FEATURE_TO_USE:
		list_of_patches = saveLoadPatch.loadUniquePatchesWithFeatureSet( \
		"{path}/UniquePatches_{feature}_{folder}_{file}_simga{i}_GaussianWindowOnAWhole.csv".format( \
			path = "testUniquePatches/algo3", \
			feature = "_".join(this_feature_set), \
			folder = "testset_illuminance1", \
			file = "test2", \
			i = sigma))
		list_of_patches[0].setFeatureToUse(this_feature_set)
		list_of_patches[0].setFeatureWeights(np.ones(len(list_of_patches[0].feature_to_use)))
		groundTruth.append(list_of_patches[0]) # append the best one found

	comparePatches.drawPatchesOnImg(np.copy(imgToMatch), groundTruth, mark_sequence = True)

	# cv2.imwrite("testPatchHSV/GaussianWindowOnAWhole/testset_illuminance1"+folder_suffix+"/_testPatches.jpg",comparePatches.drawPatchesOnImg(np.copy(img), testPatches, True))
	# cv2.imwrite("testPatchHSV/GaussianWindowOnAWhole/testset_illuminance1"+folder_suffix+"/_groundTruth.jpg",comparePatches.drawPatchesOnImg(np.copy(imgToMatch), groundTruth, True))
	# cv2.imwrite("testPatchHSV/GaussianWindowOnAWhole/testset_illuminance1"+folder_suffix+"/_matchesFound.jpg",comparePatches.drawPatchesOnImg(np.copy(imgToMatch), matchesFound, True))

	# checkHistogramOfTruthAndMatchesFound(testPatches, groundTruth, matchesFound, img, imgToMatch, "./{upperPath}/GaussianWindowOnAWhole/testset_illuminance1{folder_suffix}/hists".format(upperPath = upperPath, folder_suffix = folder_suffix), saveHist = True, displayHist = False)

	listOfMatches = testDescriptorPerformance("images", "testset_illuminance1", testPatches, "test1.jpg","test2.jpg","GaussianWindowOnAWhole",True,  folder_suffix, sigma, upperPath = upperPath)
	for i in range(0, len(listOfMatches)):
		matchesFound.append(listOfMatches[i][0]) # just append the best match
	cv2.imwrite(upperPath + "/GaussianWindowOnAWhole/testset_illuminance1"+folder_suffix+"/_combined_scene_match.jpg",comparePatches.drawMatchesOnImg(img, imgToMatch, testPatches, matchesFound, show = False))
	return


def populate_testset_illuminance2(folder_suffix = ""):
	sigma = 39
	testPatches = []
	groundTruth = []
	matchesFound = []
	# for testset testset_illuminance2
	img = cv2.imread("images/testset_illuminance2/test1.jpg", 1)
	imgToMatch = cv2.imread("images/testset_illuminance2/test2.jpg", 1)

	print img.shape, ",", imgToMatch.shape
	# plt.imshow(np.dstack((img[:,:,2], img[:,:,1], img[:,:,0])))
	# plt.show()

	testPatches.append(comparePatches.Patch(579, 819, sigma)) # test0
	testPatches.append(comparePatches.Patch(478, 899, sigma)) # test1
	testPatches.append(comparePatches.Patch(270, 742, sigma)) # test2
	testPatches.append(comparePatches.Patch(326, 465, sigma)) # test3
	testPatches.append(comparePatches.Patch(716, 298, sigma)) # test4

	groundTruth.append(comparePatches.Patch(583,808,sigma)) # test0
	groundTruth.append(comparePatches.Patch(482,886,sigma)) # test1
	groundTruth.append(comparePatches.Patch(276,730,sigma)) # test2
	groundTruth.append(comparePatches.Patch(333,458,sigma)) # test3
	groundTruth.append(comparePatches.Patch(716,294,sigma)) # test4

	# listOfPatchMatches = saveLoadPatch.loadPatchMatches("testPatchHSV/{folderToSave}/{testFolder}/GoodMatches_{folder}_{file1}_{file2}_simga{i}_shiftBy{step}_useGaussianWindow_{tf}_5levels.csv".format(testFolder = "testset_illuminance2"+folder_suffix, folderToSave = "GaussianWindowOnAWhole", folder = "testset_illuminance2", file1 = "test1", file2 = "test2", i = sigma, step = 0.5, tf = True))
	# for i in range(0, len(listOfPatchMatches)):
	# 	matchesFound.append(listOfPatchMatches[i][0]) # just append the best match

	# cv2.imwrite("testPatchHSV/GaussianWindowOnAWhole/testset_illuminance2"+folder_suffix+"/_testPatches.jpg",comparePatches.drawPatchesOnImg(np.copy(img), testPatches, True))
	# cv2.imwrite("testPatchHSV/GaussianWindowOnAWhole/testset_illuminance2"+folder_suffix+"/_groundTruth.jpg",comparePatches.drawPatchesOnImg(np.copy(imgToMatch), groundTruth, True))
	# cv2.imwrite("testPatchHSV/GaussianWindowOnAWhole/testset_illuminance2"+folder_suffix+"/_matchesFound.jpg",comparePatches.drawPatchesOnImg(np.copy(imgToMatch), matchesFound, True))

	# checkHistogramOfTruthAndMatchesFound(testPatches, groundTruth, matchesFound, img, imgToMatch, "./testPatchHSV/GaussianWindowOnAWhole/testset_illuminance2"+folder_suffix+"/hists", saveHist = True, displayHist = False)
	
	
	# testDescriptorPerformance("testset_illuminance2", testPatches, "test1.jpg","test2.jpg","GaussianWindowOnAWhole",True,  "_256Bin_HS_earthMover", sigma)
	# testDescriptorPerformance("testset_illuminance2", testPatches, "test1.jpg","test2.jpg","GaussianWindowOnAWhole",True,  "_256Bin_HS_earthMover_pyemd", sigma)
	# testDescriptorPerformance("testset_illuminance2", testPatches, "test1.jpg","test2.jpg","GaussianWindowOnAWhole",True,  "_seperateHS_Jensen_Shannon_Divergence", sigma)
	# testDescriptorPerformance("testset_illuminance2", testPatches, "test1.jpg","test2.jpg","GaussianWindowOnAWhole",True,  "_seperateHS_earthMover", sigma)
	# testDescriptorPerformance("testset_illuminance2", testPatches, "test1.jpg","test2.jpg","GaussianWindowOnAWhole",True,  "_seperateHS_earthMoverHueSpecial", sigma)
	listOfMatches = testDescriptorPerformance("images", "testset_illuminance2", testPatches, "test1.jpg","test2.jpg","GaussianWindowOnAWhole",True,  folder_suffix, sigma)
	for i in range(0, len(listOfMatches)):
		matchesFound.append(listOfMatches[i][0]) # just append the best match
	cv2.imwrite("testPatchHSV/GaussianWindowOnAWhole/testset_illuminance2"+folder_suffix+"/_combined_scene_match.jpg",comparePatches.drawMatchesOnImg(img, imgToMatch, testPatches, matchesFound, show = False))
	return

def populate_testset_rotation1(folder_suffix = "", upperPath = "testAlgo3"):
	sigma = 39
	testPatches = []
	groundTruth = []
	matchesFound = []
	# for testset_rotation1
	img = cv2.imread("images/testset_rotation1/test1.jpg", 1)
	imgToMatch = cv2.imread("images/testset_rotation1/test2.jpg", 1)

	# print img.shape, ",", imgToMatch.shape
	# plt.imshow(np.dstack((img[:,:,2], img[:,:,1], img[:,:,0])))
	# plt.show()
	# plt.imshow(np.dstack((imgToMatch[:,:,2], imgToMatch[:,:,1], imgToMatch[:,:,0])))
	# plt.show()

	# testPatches.append(comparePatches.Patch(230, 492, sigma)) # test0
	# testPatches.append(comparePatches.Patch(189, 492, sigma)) # test1
	# testPatches.append(comparePatches.Patch(181, 59, sigma)) # test2
	# testPatches.append(comparePatches.Patch(552, 765, sigma)) # test3

	# for this_test_patch in testPatches:
	# 	this_test_patch.setFeatureToUse(MANUAL_FEATURE_TO_USE)
	# 	this_test_patch.setFeatureWeights(np.ones(len(this_test_patch.feature_to_use)))

	# groundTruth.append(comparePatches.Patch(261,607,sigma)) # test0
	# groundTruth.append(comparePatches.Patch(230,627,sigma)) # test1
	# groundTruth.append(comparePatches.Patch(26,257,sigma)) # test2
	# groundTruth.append(comparePatches.Patch(684,709,sigma)) # test3
	for this_feature_set in [[utils.BORDER_GREEN_FEATURE_ID], [utils.CENTRE_YELLOW_FEATURE_ID]]:
		list_of_patches = saveLoadPatch.loadUniquePatchesWithFeatureSet( \
		"{path}/UniquePatches_{feature}_{folder}_{file}_simga{i}_GaussianWindowOnAWhole.csv".format( \
			path = "testUniquePatches/algo3", \
			feature = "_".join(this_feature_set), \
			folder = "testset_rotation1", \
			file = "test1", \
			i = sigma))
		list_of_patches[0].setFeatureToUse(this_feature_set)
		list_of_patches[0].setFeatureWeights(np.ones(len(list_of_patches[0].feature_to_use)))
		testPatches.append(list_of_patches[0]) # append the best one found

	comparePatches.drawPatchesOnImg(np.copy(img), testPatches, mark_sequence = True)

	# listOfPatchMatches = saveLoadPatch.loadPatchMatches("testPatchHSV/{folderToSave}/{testFolder}/GoodMatches_{folder}_{file1}_{file2}_simga{i}_shiftBy{step}_useGaussianWindow_{tf}_5levels.csv".format(testFolder = "testset_rotation1"+folder_suffix, folderToSave = "GaussianWindowOnAWhole", folder = "testset_rotation1", file1 = "test1", file2 = "test2", i = sigma, step = 0.5, tf = True))
	# for i in range(0, len(listOfPatchMatches)):
		# matchesFound.append(listOfPatchMatches[i][0]) # just append the best match
	
	# cv2.imwrite("testPatchHSV/GaussianWindowOnAWhole/testset_rotation1"+folder_suffix+"/_testPatches.jpg",comparePatches.drawPatchesOnImg(np.copy(img), testPatches, True))
	# cv2.imwrite("testPatchHSV/GaussianWindowOnAWhole/testset_rotation1"+folder_suffix+"/_groundTruth.jpg",comparePatches.drawPatchesOnImg(np.copy(imgToMatch), groundTruth, True))
	# cv2.imwrite("testPatchHSV/GaussianWindowOnAWhole/testset_rotation1"+folder_suffix+"/_matchesFound.jpg",comparePatches.drawPatchesOnImg(np.copy(imgToMatch), matchesFound, True))

	# cv2.imwrite("testPatchHSV/GaussianWindowOnAWhole/testset_rotation1"+folder_suffix+"/_testPatches_gray.jpg",comparePatches.drawPatchesOnImg(cv2.cvtColor(np.copy(img),cv2.COLOR_BGR2GRAY), testPatches, True))
	# cv2.imwrite("testPatchHSV/GaussianWindowOnAWhole/testset_rotation1"+folder_suffix+"/_groundTruth_gray.jpg",comparePatches.drawPatchesOnImg(cv2.cvtColor(np.copy(imgToMatch),cv2.COLOR_BGR2GRAY), groundTruth, True))
	# cv2.imwrite("testPatchHSV/GaussianWindowOnAWhole/testset_rotation1"+folder_suffix+"/_matchesFound_gray.jpg",comparePatches.drawPatchesOnImg(cv2.cvtColor(np.copy(imgToMatch),cv2.COLOR_BGR2GRAY), matchesFound, True))



	# checkHistogramOfTruthAndMatchesFound(testPatches, groundTruth, matchesFound, img, imgToMatch, "./testPatchHSV/GaussianWindowOnAWhole/testset_rotation1"+folder_suffix+"/hists", saveHist = False, displayHist = True)
	listOfMatches = testDescriptorPerformance( \
		"images", \
		"testset_rotation1", \
		testPatches, \
		"test1.jpg", \
		"test2.jpg", \
		"GaussianWindowOnAWhole", \
		True, \
		folder_suffix, \
		sigma, \
		upperPath = upperPath)
	for i in range(0, len(listOfMatches)):
		matchesFound.append(listOfMatches[i][0]) # just append the best match
	cv2.imwrite(upperPath + "/GaussianWindowOnAWhole/testset_rotation1"+folder_suffix+"/_combined_scene_match.jpg", \
		comparePatches.drawMatchesOnImg(img, imgToMatch, testPatches, matchesFound, show = False))
	return

def populate_testset_rotation2(folder_suffix = ""):
	sigma = 39
	testPatches = []
	groundTruth = []
	matchesFound = []
	# for testset_rotation2
	img = cv2.imread("images/testset_rotation2/test1.jpg", 1)
	imgToMatch = cv2.imread("images/testset_rotation2/test2.jpg", 1)

	print img.shape, ",", imgToMatch.shape
	# plt.imshow(np.dstack((img[:,:,2], img[:,:,1], img[:,:,0])))
	# plt.show()
	# plt.imshow(np.dstack((imgToMatch[:,:,2], imgToMatch[:,:,1], imgToMatch[:,:,0])))
	# plt.show()
	

	# testPatches.append(comparePatches.Patch(575, 817, sigma)) # test0
	testPatches.append(comparePatches.Patch(573, 814, sigma - 8)) # adjusted test0
	testPatches.append(comparePatches.Patch(478, 897, sigma)) # test1
	testPatches.append(comparePatches.Patch(265, 736, sigma)) # test2
	testPatches.append(comparePatches.Patch(321, 457, sigma)) # test3
	testPatches.append(comparePatches.Patch(121, 459, sigma)) # test4
	testPatches.append(comparePatches.Patch(228, 447, sigma)) # test5
	testPatches.append(comparePatches.Patch(700, 292, sigma)) # test6

	# groundTruth.append(comparePatches.Patch(457,872,sigma)) # test0
	# groundTruth.append(comparePatches.Patch(451,874,sigma)) # test0 possible ground truth at sigma = 39
	groundTruth.append(comparePatches.Patch(451,865,sigma)) # test0 possible ground truth at sigma = 39
	# groundTruth.append(comparePatches.Patch(451,883,sigma)) # test0 possible ground truth at sigma = 39
	# groundTruth.append(comparePatches.Patch(460,874,sigma)) # test0 possible ground truth at sigma = 39
	groundTruth.append(comparePatches.Patch(334,916,sigma)) # test1
	groundTruth.append(comparePatches.Patch(191,694,sigma)) # test2
	groundTruth.append(comparePatches.Patch(340,447,sigma)) # test3
	groundTruth.append(comparePatches.Patch(154,384,sigma)) # test4
	groundTruth.append(comparePatches.Patch(258,406,sigma)) # test5
	groundTruth.append(comparePatches.Patch(748,424,sigma)) # test6



	# listOfPatchMatches = saveLoadPatch.loadPatchMatches("testPatchHSV/{folderToSave}/{testFolder}/GoodMatches_{folder}_{file1}_{file2}_simga{i}_shiftBy{step}_useGaussianWindow_{tf}_5levels.csv".format(testFolder = "testset_rotation2" + folder_suffix, folderToSave = "GaussianWindowOnAWhole", folder = "testset_rotation2", file1 = "test1", file2 = "test2", i = sigma, step = 0.5, tf = True))
	# for i in range(0, len(listOfPatchMatches)):
		# matchesFound.append(listOfPatchMatches[i][0]) # just append the best match

	# comparePatches.drawPatchesOnImg(np.copy(img), testPatches, True)
	# comparePatches.drawPatchesOnImg(np.copy(imgToMatch), groundTruth, True)
	# comparePatches.drawPatchesOnImg(np.copy(imgToMatch), matchesFound, True)


	# checkHistogramOfTruthAndMatchesFound(testPatches, groundTruth, matchesFound, img, imgToMatch, "./testPatchHSV/GaussianWindowOnAWhole/testset_rotation2"+folder_suffix+"/hists", saveHist = True, displayHist = False)
	listOfMatches = testDescriptorPerformance("images","testset_rotation2", testPatches, "test1.jpg","test2.jpg","GaussianWindowOnAWhole",True,  folder_suffix, sigma)
	for i in range(0, len(listOfMatches)):
		matchesFound.append(listOfMatches[i][0]) # just append the best match
	cv2.imwrite("testPatchHSV/GaussianWindowOnAWhole/testset_rotation2"+folder_suffix+"/_combined_scene_match.jpg",comparePatches.drawMatchesOnImg(img, imgToMatch, testPatches, matchesFound, show = False))
	return

def populate_testset4(folder_suffix = ""):
	# for testset4
	sigma = 39
	testPatches = []
	groundTruth = []
	matchesFound = []
	img = cv2.imread("images/testset4/test1.jpg", 1)
	imgToMatch = cv2.imread("images/testset4/test2.jpg", 1)

	# plt.imshow(np.dstack((imgToMatch[:,:,2], imgToMatch[:,:,1], imgToMatch[:,:,0])))
	# plt.show()
	


	testPatches.append(comparePatches.Patch(400, 862, sigma)) # test0
	testPatches.append(comparePatches.Patch(310, 575, sigma)) # test1
	# testPatches.append(comparePatches.Patch(556, 20, sigma) ) # test2
	testPatches.append(comparePatches.Patch(379, 424, sigma)) # test2
	testPatches.append(comparePatches.Patch(475, 267, sigma)) # test3
	testPatches.append(comparePatches.Patch(440, 645, sigma)) # test4
	testPatches.append(comparePatches.Patch(592, 494, sigma)) # test5
	testPatches.append(comparePatches.Patch(437, 645, sigma)) # test6

	

	groundTruth.append(comparePatches.Patch(412, 746, sigma + 8)) # test0
	groundTruth.append(comparePatches.Patch(292, 412, sigma)) # test1
	# groundTruth.append(comparePatches.Patch(446, 16, sigma)) # test2
	groundTruth.append(comparePatches.Patch(345, 276, sigma - 6)) # test2
	groundTruth.append(comparePatches.Patch(409, 164, sigma - 6)) # test3
	groundTruth.append(comparePatches.Patch(425, 478, sigma)) # test4
	groundTruth.append(comparePatches.Patch(540, 330, sigma - 6)) # test5
	groundTruth.append(comparePatches.Patch(420, 476, sigma)) # test6

	# listOfPatchMatches = saveLoadPatch.loadPatchMatches("testPatchHSV/{folderToSave}/{testFolder}/GoodMatches_{folder}_{file1}_{file2}_simga{i}_shiftBy{step}_useGaussianWindow_{tf}_5levels.csv".format(testFolder = "testset4" +folder_suffix, folderToSave = "GaussianWindowOnAWhole", folder = "testset4", file1 = "test1", file2 = "test2", i = sigma, step = 0.5, tf = True))
	# for i in range(0, len(listOfPatchMatches)):
	# 	matchesFound.append(listOfPatchMatches[i][0]) # just append the best match
	
	# comparePatches.drawPatchesOnImg(np.copy(img), testPatches, True)
	# comparePatches.drawPatchesOnImg(np.copy(imgToMatch), matchesFound, True)
	# comparePatches.drawPatchesOnImg(np.copy(imgToMatch), groundTruth, True)
	# checkHistogramOfTruthAndMatchesFound(testPatches, groundTruth, matchesFound, img, imgToMatch, "./testPatchHSV/GaussianWindowOnAWhole/testset4"+folder_suffix+"/hists", saveHist = True, displayHist = False )
	
	# testDescriptorPerformance("testset4",testPatches, "test1.jpg","test2.jpg","noGaussianWindow",False, "", sigma)
	listOfMatches = testDescriptorPerformance("images", "testset4", testPatches, "test1.jpg","test2.jpg","GaussianWindowOnAWhole",True, folder_suffix, sigma)
	for i in range(0, len(listOfMatches)):
		matchesFound.append(listOfMatches[i][0]) # just append the best match
	cv2.imwrite("testPatchHSV/GaussianWindowOnAWhole/testset4"+folder_suffix+"/_combined_scene_match.jpg",comparePatches.drawMatchesOnImg(img, imgToMatch, testPatches, matchesFound, show = False))

def populate_testset7(folder_suffix = "", base_img_name = "test1.jpg", target_img_name = "test2.jpg", upperPath = "testAlgo3"):
	sigma = 39
	testPatches = []
	groundTruth = []
	matchesFound = []
	# for testset7
	img = cv2.imread("images/testset7/{base_img_name}".format(base_img_name = base_img_name), 1)
	imgToMatch = cv2.imread("images/testset7/{target_img_name}".format(target_img_name = target_img_name), 1)

	print img.shape

	global MANUAL_FEATURE_TO_USE
	MANUAL_FEATURE_TO_USE = [ \
	[utils.GENERATED_FEATURE_IDS[2]], \
	[utils.GENERATED_FEATURE_IDS[3]], \
	[utils.GENERATED_FEATURE_IDS[4]] \
	]

	print "mannual feature to use:", MANUAL_FEATURE_TO_USE

	"""read test patches"""
	for this_feature_set in MANUAL_FEATURE_TO_USE:
		list_of_patches = saveLoadPatch.loadUniquePatchesWithFeatureSet( \
		"{path}/UniquePatches_{feature}_{folder}_{file}_simga{i}_GaussianWindowOnAWhole.csv".format( \
			path = "testUniquePatches/algo3", \
			feature = "_".join(this_feature_set), \
			folder = "testset7", \
			file = "test1", \
			i = sigma))
		list_of_patches[0].setFeatureToUse(this_feature_set)
		list_of_patches[0].setFeatureWeights(np.ones(len(list_of_patches[0].feature_to_use)))
		testPatches.append(list_of_patches[0]) # append the best one found
	comparePatches.drawPatchesOnImg(np.copy(img), testPatches, mark_sequence = True)

	global ALL_FEATURE_TO_COMPUTE
	ALL_FEATURE_TO_COMPUTE = list(set([feature for patch in testPatches for feature in patch.feature_to_use]))

	""" read ground truth """
	list_of_patches = saveLoadPatch.loadUniquePatchesWithFeatureSet( \
	"{path}/UniquePatches_{feature}_{folder}_{file}_simga{i}_GaussianWindowOnAWhole.csv".format( \
		path = "testUniquePatches/algo3", \
		feature = "_".join([utils.GENERATED_FEATURE_IDS[2]]), \
		folder = "testset7", \
		file = "test3", \
		i = sigma))
	groundTruth.append(list_of_patches[0])
	# groundTruth.append(comparePatches.Patch(269, 979, 23)) # red dot, test3.jpg
	groundTruth.append(comparePatches.Patch(268, 976, 23)) # red dot, test3.jpg
	# groundTruth.append(comparePatches.Patch(272, 976, 23)) # red dot, test3.jpg
	# groundTruth.append(comparePatches.Patch(264, 976, 23)) # red dot, test3.jpg
	# groundTruth.append(comparePatches.Patch(268, 980, 23)) # red dot, test3.jpg
	# groundTruth.append(comparePatches.Patch(268, 972, 23)) # red dot, test3.jpg

	# groundTruth.append(comparePatches.Patch(342, 935, 23)) # heart orange, test3.jpg
	groundTruth.append(comparePatches.Patch(340, 936, 23)) # heart orange, test3.jpg
	# groundTruth.append(comparePatches.Patch(340, 932, 23)) # heart orange, test3.jpg
	# groundTruth.append(comparePatches.Patch(344, 932, 23)) # heart orange, test3.jpg
	# groundTruth.append(comparePatches.Patch(344, 936, 23)) # heart orange, test3.jpg
	# groundTruth.append(comparePatches.Patch(388, 904, 31)) # heart orange usually wrongly matched patch, test3.jpg
	
	for i  in range(0, len(MANUAL_FEATURE_TO_USE)):
		print "setting feature_to_use for groundTruth[{i}]".format(i = i)
		groundTruth[i].setFeatureToUse(MANUAL_FEATURE_TO_USE[i])
		groundTruth[i].setFeatureWeights(np.zeros(len(groundTruth[i].feature_to_use)))
	comparePatches.drawPatchesOnImg(np.copy(imgToMatch), groundTruth, mark_sequence = True)

	""" read matches found """
	# list_of_patches = saveLoadPatch.loadPatchMatches( \
	# 	upperPath + \
	# 	"/{folderToSave}/{testFolder}/GoodMatches_{folder}_{file1}_{file2}_simga{i}_shiftBy{step}_useGaussianWindow_{tf}_5levels.csv".format(\
	# 		testFolder = "testset7" + folder_suffix, \
	# 		folderToSave = "GaussianWindowOnAWhole", \
	# 		folder = "testset7", \
	# 		file1 = "test1", \
	# 		file2 = "test3", \
	# 		i = sigma, \
	# 		step = 0.5, \
	# 		tf = True))
	# for i in range(0, len(list_of_patches)):
	# 	print "patch size of matchesFound[{i}] = ".format( i = i ), list_of_patches[i][0].size
	# 	matchesFound.append(list_of_patches[i][0]) # just append the best match
	# comparePatches.drawPatchesOnImg(np.copy(imgToMatch), matchesFound, mark_sequence = True)
	
	# checkHistogramOfTruthAndMatchesFound( \
	# 	testPatches, \
	# 	groundTruth, \
	# 	matchesFound, \
	# 	img, \
	# 	imgToMatch, \
	# 	"./{upperPath}/GaussianWindowOnAWhole/testset7{folder_suffix}/hists".format(\
	# 		upperPath = upperPath, folder_suffix = folder_suffix), \
	# 	True, \
	# 	True)

	# raise ValueError ("purpose stop for checking hists")
  
	listOfBestMatches = testDescriptorPerformance( \
		"images", \
		"testset7", \
		testPatches, \
		base_img_name, \
		target_img_name, \
		"GaussianWindowOnAWhole", \
		True, \
		folder_suffix, \
		sigma, \
		upperPath = upperPath)
	for i in range(0, len(listOfBestMatches)):
		matchesFound.append(listOfBestMatches[i][0]) # just append the best match
	cv2.imwrite( \
		upperPath + \
		"/GaussianWindowOnAWhole/testset7" + \
		folder_suffix + \
		"/_combined_scene_match_{base_img_name}_{target_img_name}.jpg".format(\
			base_img_name = base_img_name[:base_img_name.find(".")], \
			target_img_name = target_img_name[:target_img_name.find(".")]),\
		comparePatches.drawMatchesOnImg(img, imgToMatch, testPatches, matchesFound, show = True))

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
	
	"""Set the FEATURES for comparePatches findDistinguishablePatchesAlgo3 process"""
	comparePatches.FEATURES = copy.deepcopy(utils.ALL_FEATURE_IDS)
	distinguishablePatches = comparePatches.findDistinguishablePatchesAlgo3(img, sigma, remove_duplicate_thresh_dict)
	
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

def findDistinguishablePatchesAndExecuteMatching(image_db, test_folder_name, test1_img_name, test2_img_name, folder_suffix, upperPath = "testMatches"):
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
		upperPath)
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

	comparePatches.drawMatchesOnImg(np.copy(img), np.copy(imgToMatch), testPatches, matchesFound, \
		show = True, custom_colors = custom_colors)

	comparePatches.drawMatchesOnImg(np.copy(img), np.copy(imgToMatch), testPatches, groundTruth, \
		show = True)
	
	raise ValueError ("purpose stop for drawing using different color")

	# plot the statistics
	checkHistogramOfTruthAndMatchesFound(testPatches, groundTruth, matchesFound, img, imgToMatch, \
		"./{upperPath}/{folderToSave}/{testFolder}/hists".format(\
			upperPath = upperPath,\
			folderToSave = "GaussianWindowOnAWhole", \
			testFolder = test_folder_name +folder_suffix), True, True)


def mannalPruning(image_db, test_folder_name, test1_img_name, test2_img_name, \
	folder_suffix, upperPath = "testPatchHSV"):
	sigma = 39
	level = 5
	test_patches = []
	ground_truth = []
	matches_found = []

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
		show = True)

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

	return	



def main():
	# ---------------------------------TEST DESCRIPTOR PERFORMANCE-----------------------------------
	# folder_suffix = "_HOG_Jensen_Shannon_Divergence"
	# folder_suffix = "_HOG_Circular_subCircle_Jensen_Shannon_Divergence"
	# folder_suffix = "_HOG_16Bin_subAndSuperCircle_Jensen_Shannon_Divergence"
	# folder_suffix = "_HOG_16Bin_subCircle_Jensen_Shannon_Divergence"
	# folder_suffix = "_DistinguishablePatches_CornerResponse_Jensen_Shannon_Divergence"
	# folder_suffix = "_DistinguishablePatches_HSV_Descriptor_seperateHS_Jensen_Shannon_Divergence"
	# folder_suffix = "_DistinguishablePatches_HOG_Jensen_Shannon_Divergence"
	# folder_suffix = "_seperateHS_Jensen_Shannon_Divergence_pyramid"
	# folder_suffix = "_seperateHSV_earthMover"
	# folder_suffix = "_seperateHS_earthMoverHueSpecial"
	# folder_suffix = "_unnormalized_HOG_Ori_Assignment_Jensen_Shannon_Divergence"
	# folder_suffix = "_eyeballed_unique_patches_Jensen_Shannon_Divergence_Response_Based_Saturation_filtered_aggregated_hue_expanded_border_saturation_16bin"
	# folder_suffix = "_eyeballed_unique_patches_Jensen_Shannon_Divergence_Hat_Response_Based_SaturationWeighted_Hue"
	# folder_suffix = "_eyeballed_unique_patches_Jensen_Shannon_Divergence_Hat_Response_Based_SaturationWeighted_Hue_Heart_Contour"
	# folder_suffix = "_full_algo_top20_unique_patches_response_based"
	folder_suffix = "_full_algo_top20_unique_patches_descriptor_based"
	# folder_suffix = "_eyeballed_unique_patches_Jensen_Shannon_Divergence_Hat_Response_Based_SaturationWeighted_Hue_with_Shape"
	# folder_suffix = "_eyeballed_unique_patches_seperateHS_Jensen_Shannon_Divergence_Custom_Dissimilarity_Based"
	# feature_to_use = 'HOG'
	# FEATURE_WEIGHTING[feature_to_use] = 1.0 # no need to use global marker, since not reassigning the global variable
	start_time = time.time()
	print 'start matching:', start_time
	
	"""Necessary step to initialise the features constructed"""
	utils.loadGeneratedFeatureParadigm()

	"""Mannual Tests"""
	# populate_testset_illuminance1(folder_suffix, "testAlgo3")
	# populate_testset_illuminance2(folder_suffix)
	# populate_testset_rotation1(folder_suffix, "testAlgo3")
	# populate_testset_rotation2(folder_suffix)
	# populate_testset4(folder_suffix)
	populate_testset7(folder_suffix, base_img_name = "test1.jpg", target_img_name = "test3.jpg", upperPath = "testAlgo3")
	
	"""Test full automatic algorithm"""
	# findDistinguishablePatchesAndExecuteMatching("images", "testset_flower2", "test1.jpg", "test3.jpg", folder_suffix, upperPath = "testAlgo3")
	# findAndSaveDistinguishablePatches("testset_rotation1", "test1.jpg", folder_suffix)
	# populateFeatureMatchingStatistics("images", "testset8", "test1.jpg", "test2.jpg", folder_suffix, upperPath = "testAlgo3")
	# mannalPruning("images", "testset_flower10", "test1.jpg", "test3.jpg", folder_suffix, upperPath = "testAlgo3")
	# generateHists("images", "testAlgo3", "testset_illuminance1", folder_suffix, file1 = "test1", file2 = "test2", sigma = 39)
	print 'finish matching; time spent:', time.time() - start_time

	return

if __name__ == "__main__":
	main()