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

FEATURE_WEIGHTING = {
	'HSV':0.0,
	'HOG':1.0
}

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
def findBestMatches(patchToMatch, matchPatches, n = 1, histToUse = "HSV", metricFunc = comparePatches.Jensen_Shannon_Divergence, useSeperateHSVHists = True): # metric = "Jensen_Shannon_Divergence"
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
			# TODO: if use earthMover for HOG, then we should use comparePatches.earthMoverHatDistanceForHOG
			hog_distances[i] = getHistArrl2Distance(patchToMatch.HOGArr, matchPatches[i].HOGArr, metricFunc) 
	
	overall_distances = np.sqrt(FEATURE_WEIGHTING['HSV'] * color_distances**2 + FEATURE_WEIGHTING['HOG'] * hog_distances**2)
	sortedIndex = np.argsort(overall_distances) # the lower the (distance) the better
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

def testFindOnePatchMatch(patchToMatch, patchesArr):
	goodMatchesArr = []
	k = 1 # number of best patches to extract for a sigma level (for one patch array)
	for i in range(0, len(patchesArr)):		
		goodMatches = findBestMatches(patchToMatch, patchesArr[i],k, histToUse = "HSV", metricFunc = comparePatches.Jensen_Shannon_Divergence)
		goodMatchesArr.append(goodMatches)
	overAllGoodMatches = [item for sublist in goodMatchesArr for item in sublist] # flatten the list of list -> list
	print "In testFindOnePatchMatch: overAllGoodMatches len should be {i}:".format(i = k*len(patchesArr)), len(overAllGoodMatches)
	return findBestMatches(patchToMatch, overAllGoodMatches, k*len(patchesArr), histToUse = "HSV", metricFunc = comparePatches.Jensen_Shannon_Divergence)
	# return findBestMatches(patchToMatch, overAllGoodMatches, k,histToUse = "HSV", metricFunc = comparePatches.Jensen_Shannon_Divergence)

def createFolder(folderToSave, folderName, suffix, upperPath):
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

def testDescriptorPerformancePyramid():
	"""Build image pyramid using opencv"""

	"""propogateUp/propogateDown testPatches"""
	return


def testDescriptorPerformance(folderName,testPatches, imgName,imgToMatchName,folderToSave,useGaussianWindow, suffix = "", sigma = 39, upperPath = "testPatchHSV"):
	"""
	testDescriptorPerformance: given testPatches, run the descriptor matching process;
	:number of gaussian: 5 (2 level down, 2 level up)
	:gaussianScaleFactor: 1.2
	:gaussian window weighting on the whole patch: gaussian sigma = window length / 6
	:sigma: 39
	:patch search step: 0.5 (shift by 1/4 patch length)
	"""

	#Create folder for the testset
	if(not os.path.isdir("./{upperPath}/{folderToSave}/{testFolder}".format(
		upperPath = upperPath, 
		folderToSave = folderToSave, 
		testFolder = folderName + suffix))):
		os.makedirs("./{upperPath}/{folderToSave}/{testFolder}".format(
			upperPath = upperPath, 
			folderToSave = folderToSave, 
			testFolder = folderName + suffix))

	img = cv2.imread("images/{folder}/{name}".format(folder = folderName, name = imgName), 1)
	img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.int)
	imgToMatch = cv2.imread("images/{folder}/{name}".format(folder = folderName, name = imgToMatchName), 1)
	imgToMatch_gray = cv2.cvtColor(imgToMatch, cv2.COLOR_BGR2GRAY).astype(np.int)

	#Extract match patches
	patchStep = 0.5 # shift the patch by 1/4 patch len
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

	#Compute the Color Hist of the matchPatches
	for index in range(0, len(patchesArr)):
		matchPatches = patchesArr[index]
		for i in range(0, len(matchPatches)):
			# print "computeRGBHistogram for matchPatches[{i}]".format(i = i) 
			# matchPatches[i].computeRGBHistogram(imgToMatch)
			if(FEATURE_WEIGHTING['HSV'] != 0):
				print "computeHSVHistogram for matchPatch[{i}] of patchArr[{index}] in {f}".format(i =i, index = index , f = folderName)
				matchPatches[i].computeHSVHistogram(imgToMatch,useGaussianWindow)
			if(FEATURE_WEIGHTING['HOG'] != 0):
				print "computeHOG for matchPatch[{i}] of patchArr[{index}] in {f}".format(i =i, index = index , f = folderName)
				matchPatches[i].computeHOG(imgToMatch_gray, useGaussianWindow)
		# One PatchesArr done!
		if(FEATURE_WEIGHTING['HSV'] != 0):
			print "computeHSVHistogram of patchArr[{index}] in {f} done".format(i =i, index = index , f = folderName)
		if(FEATURE_WEIGHTING['HOG'] != 0):
			print "computeHOG of patchArr[{index}] in {f} done".format(i =i, index = index , f = folderName)


	testPatchMatches = []
	#loop over all test cases
	for testPatchIndex in range(0, len(testPatches)):
		testPatch1 = testPatches[testPatchIndex]
		# testPatch1.computeRGBHistogram(img)
		if(FEATURE_WEIGHTING['HSV'] != 0):
			testPatch1.computeHSVHistogram(img, useGaussianWindow)
		if(FEATURE_WEIGHTING['HOG'] != 0):
			testPatch1.computeHOG(img_gray, useGaussianWindow)
		cv2.imwrite("{upperPath}/{folderToSave}/{testFolder}/testPatch{testPatchIndex}_OriginalPatches_{folder}_{file1}_{file2}_simga{i}.jpg".format(
			upperPath = upperPath, 
			testPatchIndex= testPatchIndex, 
			testFolder = folderName + suffix, 
			folderToSave = folderToSave, 
			folder = folderName, 
			file1 = imgName[0:imgName.find(".")], 
			file2 = imgToMatchName[0:imgToMatchName.find(".")], 
			i = sigma), comparePatches.drawPatchesOnImg(np.copy(img), testPatch1, False))
		
		bestMatch = testFindOnePatchMatch(testPatch1, patchesArr) # will return the best matches of the testPatch1
		print "matches found for test patch ", testPatchIndex
		for i in range (0, len(bestMatch)):
			print bestMatch[i].x, ",", bestMatch[i].y, ",",bestMatch[i].size
		testPatchMatches.append(bestMatch)
		imgToSave = comparePatches.drawPatchesOnImg(np.copy(imgToMatch),bestMatch, False, 1.0/len(bestMatch))
		cv2.imwrite("{upperPath}/{folderToSave}/{testFolder}/testPatch{testPatchIndex}_GoodMatches_{folder}_{file1}_{file2}_simga{i}_shiftBy{step}_useGaussianWindow_{tf}_5levels.jpg".format(
			upperPath = upperPath, 
			testPatchIndex = testPatchIndex, 
			testFolder = folderName + suffix, 
			folderToSave = folderToSave, 
			folder = folderName, 
			file1 = imgName[0:imgName.find(".")], 
			file2 = imgToMatchName[0:imgToMatchName.find(".")], 
			i = sigma, 
			step = patchStep, 
			tf = useGaussianWindow), imgToSave)
	
	
	testPatchMatches = [item for sublist in testPatchMatches for item in sublist] # flatten the list of list -> list
	saveLoadPatch.savePatchMatches(testPatchMatches, 5, "{upperPath}/{folderToSave}/{testFolder}/GoodMatches_{folder}_{file1}_{file2}_simga{i}_shiftBy{step}_useGaussianWindow_{tf}_5levels.csv".format(
		upperPath = upperPath, 
		testFolder = folderName + suffix, 
		folderToSave = folderToSave, 
		folder = folderName, 
		file1 = imgName[0:imgName.find(".")], 
		file2 = imgToMatchName[0:imgToMatchName.find(".")], 
		i = sigma, 
		step = patchStep, 
		tf = useGaussianWindow))
	return testPatchMatches

def checkHistogramOfTruthAndMatchesFound(testPatches, groundTruth, matchesFound, img, imgToMatch, path, saveHist = False, displayHist = True):
	if(not os.path.isdir(path)):
		os.makedirs(path)

	# testPatches[0].computeHSVHistogram(img, True, True)
	# histLen = len(testPatches[0].HSVHist) # poll the length of the flattened HSVHist
	# C = np.ones(shape = (histLen, histLen))
	# rows = np.arange(0,histLen).reshape((histLen, 1))
	# rows = np.repeat(rows, histLen, axis = 1)
	# cols = np.arange(0, histLen).reshape((1,histLen))
	# cols = np.repeat(cols, histLen, axis = 0)
	# C = C + abs(rows - cols)

	for i in range(0, len(testPatches)):
	# for i in range(2, 5):
		# plot the overall H,S,V seperate Histogram 
		# plotStatistics.plotColorHistogram(testPatches[i], img, path,"testPatch{i}".format(i = i), saveHist, displayHist)
		# plotStatistics.plotColorHistogram(groundTruth[i], imgToMatch, path,"groundTruth{i}".format(i = i), saveHist, displayHist)
		# plotStatistics.plotColorHistogram(matchesFound[i], imgToMatch, path,"matchFound{i}".format(i = i),saveHist,displayHist)
		
		testPatches[i].computeHSVHistogram(img, True, True)
		groundTruth[i].computeHSVHistogram(imgToMatch, True, True)
		matchesFound[i].computeHSVHistogram(imgToMatch, True, True)

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
		
		
		# plotStatistics.plotHSVHistCmp(path, "HSVhistCmp{i}".format(i = i), testPatches[i].HSVHist, "testPatch{i}".format(i = i), matchesFound[i].HSVHist, "matchFound{i}".format(i = i), groundTruth[i].HSVHist,"groundTruth{i}".format(i = i), saveHist, displayHist)
		# for j in range(1, len(testPatches[i].HSVHistArr)):
			# subpatch_matchDistance = comparePatches.earthMoverHatDistance(testPatches[i].HSVHistArr[j], matchesFound[i].HSVHistArr[j])
			# subpatch_groundTruthDistance = comparePatches.earthMoverHatDistance(testPatches[i].HSVHistArr[j], groundTruth[i].HSVHistArr[j])
			# print "subpatch", j, " Match found distance:", subpatch_matchDistance
			# print "subpatch", j, " Ground Truth distance:", subpatch_groundTruthDistance
			# print "sub patch ", j, " Match Found is Better than Groud Truth? :", subpatch_groundTruthDistance >  subpatch_matchDistance
			# plotStatistics.plotHSVHistCmp(path, "HSVHistCmp{i}_subpatch{j}".format(i = i , j = j), testPatches[i].HSVHistArr[j], "testPatch{i}_subpatch{j}".format(i = i, j = j), matchesFound[i].HSVHistArr[j], "matchFound{i}_subpatch{j}".format(i = i, j = j), groundTruth[i].HSVHistArr[j],"groundTruth{i}_subpatch{j}".format(i = i, j = j), saveHist, displayHist)
		for j in range(1, len(testPatches[i].HOGArr)):
			subpatch_matchDistance = comparePatches.Jensen_Shannon_Divergence(testPatches[i].HOGArr[j], matchesFound[i].HOGArr[j])
			subpatch_groundTruthDistance = comparePatches.Jensen_Shannon_Divergence(testPatches[i].HOGArr[j], groundTruth[i].HOGArr[j])
			print "subpatch", j, " Match found distance:", subpatch_matchDistance
			print "subpatch", j, " Ground Truth distance:", subpatch_groundTruthDistance
			print "sub patch ", j, " Match Found is Better than Groud Truth? :", subpatch_groundTruthDistance >  subpatch_matchDistance
			plotStatistics.plotHOGHistCmp(path, \
				"HOGhistCmp{i}_subpatch{j}".format(i = i , j = j), \
				testPatches[i].HOGArr[j], "testPatch{i}_subpatch{j}".format(i = i, j = j), \
				matchesFound[i].HOGArr[j], "matchFound{i}_subpatch{j}".format(i = i, j = j), \
				groundTruth[i].HOGArr[j],"groundTruth{i}_subpatch{j}".format(i = i, j = j), \
				saveHist, displayHist)

		
		# print "all sum(hist) should add to 1:", np.sum(testPatches[i].HSVHist), ", ", np.sum(matchesFound[i].HSVHist), "," ,np.sum(groundTruth[i].HSVHist)
		print "seperate H, S, V hists all sum(hist) should add to 1:", np.sum(testPatches[i].HueHist), ", ", np.sum(matchesFound[i].SaturationHist), "," ,np.sum(groundTruth[i].ValueHist)
		# print "HSVHist len:", len(testPatches[i].HSVHist)
		# matchDistance = getHSVHistOverAllDistance(testPatches[i], matchesFound[i])
		# matchDistance = comparePatches.Jensen_Shannon_Divergence(testPatches[i].HSVHist, matchesFound[i].HSVHist)
		# matchDistance = comparePatches.earthMoverHatDistance(testPatches[i].HSVHist, matchesFound[i].HSVHist, C)
		# matchDistance = comparePatches.chiSquareDistance(testPatches[i].HSVHist, matchesFound[i].HSVHist)
		# matchDistance = getHSVSeperateHistAvgl2Distance(testPatches[i], matchesFound[i], comparePatches.earthMoverHatDistance)
		# matchDistance = getHistArrl2Distance([testPatches[i].HueHist, testPatches[i].SaturationHist, testPatches[i].ValueHist],[matchesFound[i].HueHist,matchesFound[i].SaturationHist,matchesFound[i].ValueHist],comparePatches.Jensen_Shannon_Divergence)
		# matchDistance = getHistArrl2Distance([testPatches[i].HueHist , testPatches[i].SaturationHist],[matchesFound[i].HueHist,matchesFound[i].SaturationHist],comparePatches.Jensen_Shannon_Divergence)
		matchDistance = comparePatches.Jensen_Shannon_Divergence(testPatches[i].HOG, matchesFound[i].HOG)
		print "Matches Found Distance:", matchDistance
		print "Matches Found Seperate HSV Hist avg l2 distance:", getHSVSeperateHistAvgl2Distance(testPatches[i], matchesFound[i], comparePatches.Jensen_Shannon_Divergence)
		# groundTruthDistance = getHSVHistOverAllDistance(testPatches[i], groundTruth[i])
		# groundTruthDistance = comparePatches.Jensen_Shannon_Divergence(testPatches[i].HSVHist, groundTruth[i].HSVHist)
		# groundTruthDistance = comparePatches.earthMoverHatDistance(testPatches[i].HSVHist, groundTruth[i].HSVHist, C)
		# groundTruthDistance = comparePatches.chiSquareDistance(testPatches[i].HSVHist, groundTruth[i].HSVHist)
		# groundTruthDistance = getHSVSeperateHistAvgl2Distance(testPatches[i], groundTruth[i], comparePatches.Jensen_Shannon_Divergence)
		# groundTruthDistance = getHistArrl2Distance([testPatches[i].HueHist, testPatches[i].SaturationHist],[groundTruth[i].HueHist, groundTruth[i].SaturationHist],comparePatches.Jensen_Shannon_Divergence)
		groundTruthDistance = comparePatches.Jensen_Shannon_Divergence(testPatches[i].HOG, groundTruth[i].HOG)
		print "Gound Truth Distance:", groundTruthDistance
		print "Ground Truth Seperate HSV Hist avg l2 distance:", getHSVSeperateHistAvgl2Distance(testPatches[i], groundTruth[i], comparePatches.Jensen_Shannon_Divergence)
		print "fullpatch Match Found is Better than Groud Truth? :", groundTruthDistance > matchDistance 
		# print "overall Match Found better than Ground Truth? : ", getHSVHistOverAllDistance(testPatches[i], groundTruth[i]) > getHSVHistOverAllDistance(testPatches[i], matchesFound[i])
		# print "overall Match Found better than Ground Truth Seperate HSV? : ",  getHSVSeperateHistAvgl2Distance(testPatches[i], groundTruth[i], comparePatches.Jensen_Shannon_Divergence) > getHSVSeperateHistAvgl2Distance(testPatches[i], matchesFound[i], comparePatches.Jensen_Shannon_Divergence)
		print "overall Match Found better than Ground Truth? : ",  getHistArrl2Distance(testPatches[i].HOGArr, groundTruth[i].HOGArr, comparePatches.Jensen_Shannon_Divergence) > getHistArrl2Distance(testPatches[i].HOGArr, matchesFound[i].HOGArr, comparePatches.Jensen_Shannon_Divergence)
		
	return

def populate_testset_illuminance1(folder_suffix = ""):
	sigma = 39
	testPatches = []
	groundTruth = []
	matchesFound = []
	# for testset testset_illuminance1
	img = cv2.imread("images/testset_illuminance1/test1.jpg", 1)
	imgToMatch = cv2.imread("images/testset_illuminance1/test2.jpg", 1)
	# plt.imshow(np.dstack((img[:,:,2], img[:,:,1], img[:,:,0])))
	# plt.show()

	testPatches.append(comparePatches.Patch(149, 826, sigma)) # test0
	testPatches.append(comparePatches.Patch(478, 721, sigma)) # test1
	testPatches.append(comparePatches.Patch(351, 822, sigma)) # test2
	testPatches.append(comparePatches.Patch(328, 943, sigma)) # test3
	testPatches.append(comparePatches.Patch(342, 145, sigma)) # test4

	groundTruth.append(comparePatches.Patch(179, 830, sigma)) # test0
	groundTruth.append(comparePatches.Patch(501, 728, sigma)) # test1
	groundTruth.append(comparePatches.Patch(377, 826, sigma)) # test2
	groundTruth.append(comparePatches.Patch(358, 943, sigma)) # test3
	groundTruth.append(comparePatches.Patch(360, 165, sigma)) # test4


	# listOfPatchMatches = saveLoadPatch.loadPatchMatches("testPatchHSV/{folderToSave}/{testFolder}/GoodMatches_{folder}_{file1}_{file2}_simga{i}_shiftBy{step}_useGaussianWindow_{tf}_5levels.csv".format(testFolder = "testset_illuminance1_256Bin_HS", folderToSave = "GaussianWindowOnAWhole", folder = "testset_illuminance1", file1 = "test1", file2 = "test2", i = sigma, step = 0.5, tf = True))
	# listOfPatchMatches = saveLoadPatch.loadPatchMatches("testPatchHSV/{folderToSave}/{testFolder}/GoodMatches_{folder}_{file1}_{file2}_simga{i}_shiftBy{step}_useGaussianWindow_{tf}_5levels.csv".format(testFolder = "testset_illuminance1_256Bin_HS_earthMover", folderToSave = "GaussianWindowOnAWhole", folder = "testset_illuminance1", file1 = "test1", file2 = "test2", i = sigma, step = 0.5, tf = True))
	# listOfPatchMatches = saveLoadPatch.loadPatchMatches("testPatchHSV/{folderToSave}/{testFolder}/GoodMatches_{folder}_{file1}_{file2}_simga{i}_shiftBy{step}_useGaussianWindow_{tf}_5levels.csv".format(testFolder = "testset_illuminance1" + folder_suffix, folderToSave = "GaussianWindowOnAWhole", folder = "testset_illuminance1", file1 = "test1", file2 = "test2", i = sigma, step = 0.5, tf = True))
	# for i in range(0, len(listOfPatchMatches)):
	# 	matchesFound.append(listOfPatchMatches[i][0]) # just append the best match

	# cv2.imwrite("testPatchHSV/GaussianWindowOnAWhole/testset_illuminance1"+folder_suffix+"/_testPatches.jpg",comparePatches.drawPatchesOnImg(np.copy(img), testPatches, True))
	# cv2.imwrite("testPatchHSV/GaussianWindowOnAWhole/testset_illuminance1"+folder_suffix+"/_groundTruth.jpg",comparePatches.drawPatchesOnImg(np.copy(imgToMatch), groundTruth, True))
	# cv2.imwrite("testPatchHSV/GaussianWindowOnAWhole/testset_illuminance1"+folder_suffix+"/_matchesFound.jpg",comparePatches.drawPatchesOnImg(np.copy(imgToMatch), matchesFound, True))


	# checkHistogramOfTruthAndMatchesFound(testPatches, groundTruth, matchesFound, img, imgToMatch, "./testPatchHSV/GaussianWindowOnAWhole/testset_illuminance1_256Bin_HS/hists", saveHist = False, displayHist = False)
	# checkHistogramOfTruthAndMatchesFound(testPatches, groundTruth, matchesFound, img, imgToMatch, "./testPatchHSV/GaussianWindowOnAWhole/testset_illuminance1_256Bin_HS_earthMover/hists", saveHist = False, displayHist = False)
	# checkHistogramOfTruthAndMatchesFound(testPatches, groundTruth, matchesFound, img, imgToMatch, "./testPatchHSV/GaussianWindowOnAWhole/testset_illuminance1_seperateHS_Jensen_Shannon_Divergence/hists", saveHist = True, displayHist = False)
	# checkHistogramOfTruthAndMatchesFound(testPatches, groundTruth, matchesFound, img, imgToMatch, "./testPatchHSV/GaussianWindowOnAWhole/testset_illuminance1"+folder_suffix+"/hists", saveHist = True, displayHist = False)
	
	# testDescriptorPerformance("testset_illuminance1", testPatches, "test1.jpg","test2.jpg","GaussianWindowOnAWhole",True,  "_256Bin_HS_earthMover", sigma)
	# testDescriptorPerformance("testset_illuminance1", testPatches, "test1.jpg","test2.jpg","GaussianWindowOnAWhole",True,  "_256Bin_HS_earthMover_pyemd", sigma)
	# testDescriptorPerformance("testset_illuminance1", testPatches, "test1.jpg","test2.jpg","GaussianWindowOnAWhole",True,  "_seperateHS_Jensen_Shannon_Divergence", sigma)
	# testDescriptorPerformance("testset_illuminance1", testPatches, "test1.jpg","test2.jpg","GaussianWindowOnAWhole",True,  "_seperateHS_earthMover", sigma)
	# testDescriptorPerformance("testset_illuminance1", testPatches, "test1.jpg","test2.jpg","GaussianWindowOnAWhole",True,  "_seperateHS_earthMoverHueSpecial", sigma)
	testDescriptorPerformance("testset_illuminance1", testPatches, "test1.jpg","test2.jpg","GaussianWindowOnAWhole",True,  folder_suffix, sigma)
	
	# cv2.imwrite("testPatchHSV/GaussianWindowOnAWhole/testset_illuminance1"+folder_suffix+"/_combined_scene_match.jpg",comparePatches.drawMatchesOnImg(img, imgToMatch, testPatches, matchesFound, show = True))
	
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
	testDescriptorPerformance("testset_illuminance2", testPatches, "test1.jpg","test2.jpg","GaussianWindowOnAWhole",True,  folder_suffix, sigma)
	
	# cv2.imwrite("testPatchHSV/GaussianWindowOnAWhole/testset_illuminance2"+folder_suffix+"/_combined_scene_match.jpg",comparePatches.drawMatchesOnImg(img, imgToMatch, testPatches, matchesFound, show = True))
	return

def populate_testset_rotation1(folder_suffix = ""):
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

	

	testPatches.append(comparePatches.Patch(230, 492, sigma)) # test0
	testPatches.append(comparePatches.Patch(189, 492, sigma)) # test1
	testPatches.append(comparePatches.Patch(181, 59, sigma)) # test2
	testPatches.append(comparePatches.Patch(552, 765, sigma)) # test3

	groundTruth.append(comparePatches.Patch(261,607,sigma)) # test0
	groundTruth.append(comparePatches.Patch(230,627,sigma)) # test1
	groundTruth.append(comparePatches.Patch(26,257,sigma)) # test2
	groundTruth.append(comparePatches.Patch(684,709,sigma)) # test3

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
	testDescriptorPerformance("testset_rotation1", testPatches, "test1.jpg","test2.jpg","GaussianWindowOnAWhole",True,  folder_suffix, sigma)
	# cv2.imwrite("testPatchHSV/GaussianWindowOnAWhole/testset_rotation1"+folder_suffix+"/_combined_scene_match.jpg",comparePatches.drawMatchesOnImg(img, imgToMatch, testPatches, matchesFound, show = True))
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
	testDescriptorPerformance("testset_rotation2", testPatches, "test1.jpg","test2.jpg","GaussianWindowOnAWhole",True,  folder_suffix, sigma)
	# cv2.imwrite("testPatchHSV/GaussianWindowOnAWhole/testset_rotation2"+folder_suffix+"/_combined_scene_match.jpg",comparePatches.drawMatchesOnImg(img, imgToMatch, testPatches, matchesFound, show = True))
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
	testDescriptorPerformance("testset4", testPatches, "test1.jpg","test2.jpg","GaussianWindowOnAWhole",True, folder_suffix, sigma)
	# cv2.imwrite("testPatchHSV/GaussianWindowOnAWhole/testset4"+folder_suffix+"/_combined_scene_match.jpg",comparePatches.drawMatchesOnImg(img, imgToMatch, testPatches, matchesFound, show = True))

def populate_testset7(folder_suffix = ""):
	sigma = 39
	testPatches = []
	groundTruth = []
	matchesFound = []
	# for testset7
	img = cv2.imread("images/testset7/test1.jpg", 1)
	imgToMatch = cv2.imread("images/testset7/test2.jpg", 1)

	# plt.imshow(np.dstack((img[:,:,2], img[:,:,1], img[:,:,0])))
	# plt.imshow(np.dstack((imgToMatch[:,:,2], imgToMatch[:,:,1], imgToMatch[:,:,0])))
	# plt.show()
	# raise ValueError
	
	testPatches.append(comparePatches.Patch(433, 792, sigma)) # test0
	testPatches.append(comparePatches.Patch(325, 899, sigma)) # test1
	testPatches.append(comparePatches.Patch(700, 759, sigma)) # test2
	testPatches.append(comparePatches.Patch(700, 530, sigma - 8)) # test3
	testPatches.append(comparePatches.Patch(484, 352, sigma)) # test4
	testPatches.append(comparePatches.Patch(80, 722, sigma)) # test5
	testPatches.append(comparePatches.Patch(162, 455, sigma)) # test6

	# matchesFound.append(comparePatches.Patch(358 , 918 , 31)) # test0
	# matchesFound.append(comparePatches.Patch(274 , 981 , 31)) # test1
	# matchesFound.append(comparePatches.Patch(554 , 876 , 31)) # test2
	# # matchesFound.append(comparePatches.Patch(106 , 904 , 31)) # test3 is mismatch using 64 bins
	# # matchesFound.append(comparePatches.Patch(727 , 565 , 27)) # test3 is mismatch using 4096 bins
	# matchesFound.append(comparePatches.Patch(606 , 782 , 47)) # test3 is mismatch using HS 256 bins
	# matchesFound.append(comparePatches.Patch(610 , 701 , 31 )) # test3 HS 256 bins new testPatch3 is correct!
	# matchesFound.append(comparePatches.Patch(91 , 415 , 39))  # test4 is mismatch due to patch size and search, test4 is good match if we use patchSize 39 + 8 + 8 = 55
	# matchesFound.append(comparePatches.Patch(100 , 883 , 39)) # test5
	# matchesFound.append(comparePatches.Patch(588 , 994 , 57)) # test6 is mismatch using 64 bins

	groundTruth.append(comparePatches.Patch(355 , 916 , 31)) # test0
	groundTruth.append(comparePatches.Patch(276 , 982 , 27)) # test1
	groundTruth.append(comparePatches.Patch(560 , 884 , 29)) # test2
	groundTruth.append(comparePatches.Patch(615 , 700 , 31)) # test3
	groundTruth.append(comparePatches.Patch(451 , 514 , 39)) # test4
	# # groundTruth.append(comparePatches.Patch(460 , 514 , 39)) # test4
	# # groundTruth.append(comparePatches.Patch(451 , 523 , 39)) # test4
	# # groundTruth.append(comparePatches.Patch(460 , 523 , 39)) # test4
	groundTruth.append(comparePatches.Patch(100, 880 , 39)) # test5
	groundTruth.append(comparePatches.Patch(146 , 657 , 37)) # test6


	# # MatchesFound using 4096Bin HSV Histogram
	# testPatchMatches = saveLoadPatch.load_data("testPatchHSV/{folderToSave}/{testFolder}/GoodMatches_{folder}_{file1}_{file2}_simga{i}_shiftBy{step}_useGaussianWindow_{tf}_5levels.dat".format(testFolder = "testset7_4096Bin_newTestPatch3", folderToSave = "GaussianWindowOnAWhole", folder = "testset7", file1 = "test1", file2 = "test2", i = sigma, step = 0.5, tf = True)) #testset7_256Bin_HS_newTestPatch3
	# print len(testPatchMatches)
	# level = 5
	# for i in range(0, len(testPatchMatches)):
	# 	# if(i/level == 0):
	# 	# 	matchesFound.append(comparePatches.Patch(testPatchMatches[i].x, testPatchMatches[i].y, testPatchMatches[i].size))
	# 	# 	break; # check the histogram of the first testPatch (since it is not accurately matched)
	# 	if(i%level == 0):
	# 		matchesFound.append(comparePatches.Patch(testPatchMatches[i].x, testPatchMatches[i].y, testPatchMatches[i].size))
	# 		print "matches found for test patch " , i/level
	# 	print testPatchMatches[i].x, ", ", testPatchMatches[i].y, ", ", testPatchMatches[i].size
	# print "should be 7:", len(matchesFound)
	
	# listOfPatchMatches = saveLoadPatch.loadPatchMatches("testPatchHSV/{folderToSave}/{testFolder}/GoodMatches_{folder}_{file1}_{file2}_simga{i}_shiftBy{step}_useGaussianWindow_{tf}_5levels.csv".format(testFolder = "testset7"+folder_suffix, folderToSave = "GaussianWindowOnAWhole", folder = "testset7", file1 = "test1", file2 = "test2", i = sigma, step = 0.5, tf = True))
	# for i in range(0, len(listOfPatchMatches)):
	# 	matchesFound.append(listOfPatchMatches[i][0]) # just append the best match
	
	# cv2.imwrite("testPatchHSV/GaussianWindowOnAWhole/testset7"+folder_suffix+"/_testPatches.jpg",comparePatches.drawPatchesOnImg(np.copy(img), testPatches, True))
	# cv2.imwrite("testPatchHSV/GaussianWindowOnAWhole/testset7"+folder_suffix+"/_matchesFound.jpg",comparePatches.drawPatchesOnImg(np.copy(imgToMatch), matchesFound, True, None))
	# cv2.imwrite("testPatchHSV/GaussianWindowOnAWhole/testset7"+folder_suffix+"/_groundTruth.jpg",comparePatches.drawPatchesOnImg(np.copy(imgToMatch), groundTruth, True))
	
	# checkHistogramOfTruthAndMatchesFound(testPatches, groundTruth, matchesFound, img, imgToMatch, "./testPatchHSV/GaussianWindowOnAWhole/testset7"+folder_suffix+"/hists", True, False)
	
	
	# bestMatchesInTest2 = testDescriptorPerformance("testset7", testPatches, "test1.jpg","test2.jpg","GaussianWindowOnAWhole",True,  folder_suffix, sigma)
	# testDescriptorPerformance("testset7", bestMatchesInTest2, "test2.jpg","test3.jpg","GaussianWindowOnAWhole",True, "_4096Bin", sigma)
	testDescriptorPerformance("testset7", testPatches, "test1.jpg","test3.jpg","GaussianWindowOnAWhole",True, "_4096Bin", sigma)
	# cv2.imwrite("testPatchHSV/GaussianWindowOnAWhole/testset7"+folder_suffix+"/_combined_scene_match.jpg",comparePatches.drawMatchesOnImg(img, imgToMatch, testPatches, matchesFound, show = True))

def findAndSaveDistinguishablePatches(test_folder_name, test_img_name, folder_suffix, sigma = 39, upperPath = "testPatchHSV"):
	img = cv2.imread("images/{folder}/{image}".format(folder = test_folder_name, image = test_img_name), 1)
	distinguishablePatches = comparePatches.findDistinguishablePatches(img, sigma)
	distinguishablePatches = distinguishablePatches[0:10]
	imgToSave = comparePatches.drawPatchesOnImg(np.copy(img), distinguishablePatches, False)
	path = createFolder("GaussianWindowOnAWhole", test_folder_name, folder_suffix, upperPath)
	cv2.imwrite("{path}/DistinguishablePatch_{folder}_{file}_simga{i}_GaussianWindowOnAWhole.jpg".format(path = path , folder = test_folder_name, file = test_img_name, i = sigma), imgToSave)
	saveLoadPatch.savePatchMatches(distinguishablePatches, 1, "{path}/HOG_DistinguishablePatch_{folder}_{file}_simga{i}_GaussianWindowOnAWhole.csv".format(path = path , folder = test_folder_name, file = test_img_name, i = sigma))
	return distinguishablePatches

def findDistinguishablePatchesAndExecuteMatching(test_folder_name, test1_img_name, test2_img_name, folder_suffix, upperPath = "testMatches"):
	sigma = 39
	testPatches = findAndSaveDistinguishablePatches(test_folder_name, test1_img_name, folder_suffix, sigma, upperPath)
	testDescriptorPerformance(
		test_folder_name, 
		testPatches, 
		test1_img_name,
		test2_img_name,
		"GaussianWindowOnAWhole",
		True,  
		folder_suffix, 
		sigma,
		upperPath)

"""TODO: complete populateFeatureMatchingTest after the Image DB is found"""
def populateFeatureMatchingTest(test_folder_name, test1_img_name, test2_img_name, folder_suffix, upperPath = "testPatchHSV"):
	sigma = 39
	level = 5
	testPatches = []
	groundTruth = []
	matchesFound = []
	# TODO: from test_folder_name, test1_img_name, test2_img_name, get the testPatches and groundTruth
	listOfPatchMatches = saveLoadPatch.loadPatchMatches("{upperPath}/{folderToSave}/{testFolder}/DistinguishablePatch_{folder}_{file}_simga{i}_GaussianWindowOnAWhole.csv".format(
		upperPath = upperPath,
		folderToSave = "GaussianWindowOnAWhole", 
		testFolder = test_folder_name +folder_suffix, 
		folder = test_folder_name, 
		file = test1_img_name[:test1_img_name.find(".")], 
		i = sigma))
	for i in range(0, len(listOfPatchMatches)):
		testPatches.append(listOfPatchMatches[i][0]) # just append the best match
	
	testPatchMatches = testDescriptorPerformance(test_folder_name, testPatches, test1_img_name,test2_img_name,"GaussianWindowOnAWhole",True,  folder_suffix, sigma)
	for i in range(0, len(testPatchMatches)):
		if(i%level == 0):
			matchesFound.append(testPatchMatches[i])

	# read test1_img_name to img
	# read test2_img_name to imgToMatch

	# checkHistogramOfTruthAndMatchesFound(testPatches, groundTruth, matchesFound, img, imgToMatch, "./{path}/hists".format(path = test_folder_name), True, True)

def main():
	# ---------------------------------TEST DESCRIPTOR -----------------------------------
	# folder_suffix = "_HOG_Jensen_Shannon_Divergence"
	# folder_suffix = "_HOG_subCircle_Jensen_Shannon_Divergence"
	# folder_suffix = "_HOG_16Bin_subAndSuperCircle_Jensen_Shannon_Divergence"
	# folder_suffix = "_HOG_16Bin_subCircle_Jensen_Shannon_Divergence"
	# folder_suffix = "_DistinguishablePatches_CornerResponse_Jensen_Shannon_Divergence"
	# folder_suffix = "_DistinguishablePatches_HSV_Descriptor_seperateHS_Jensen_Shannon_Divergence"
	# folder_suffix = "_DistinguishablePatches_HOG_Jensen_Shannon_Divergence"
	# folder_suffix = "_seperateHS_Jensen_Shannon_Divergence"
	# folder_suffix = "_seperateHSV_earthMover"
	# folder_suffix = "_seperateHS_earthMoverHueSpecial"
	start_time = time.time()
	print 'start matching:', start_time
	# populate_testset_illuminance1(folder_suffix)
	# populate_testset_illuminance2(folder_suffix)
	# populate_testset_rotation1(folder_suffix)
	# populate_testset_rotation2(folder_suffix)
	# populate_testset4(folder_suffix)
	# populate_testset7(folder_suffix)
	# findAndSaveDistinguishablePatches("testset_rotation1", "test1.jpg", folder_suffix)
	# populateFeatureMatchingTest("testset_rotation1", "test1.jpg", "test2.jpg","_DistinguishablePatches_HOG_Jensen_Shannon_Divergence")
	print 'finish matching; time spent:', time.time() - start_time

	raise ValueError("testDescriptorPerformance")

	# -------------------------FULL ALGORITHEM: Extract Distinguishable points + Match ---------------------
	folderName = "testset7"
	imgName = "test1.jpg"
	imgToMatchName = "test2.jpg"
	# imgName = "test2.jpg"
	# imgToMatchName = "test3.jpg"

	img = cv2.imread("images/{folder}/{name}".format(folder = folderName, name = imgName), 1)
	imgToMatch = cv2.imread("images/{folder}/{name}".format(folder = folderName, name = imgToMatchName), 1)
	sigma = 39


	# # for testset3
	# # testPatch1 = comparePatches.Patch(540, 65,sigma) # test1

	# # for testset4
	# # testPatch1 = comparePatches.Patch(400, 862, sigma) # test1
	# # testPatch1 = comparePatches.Patch(310, 575, sigma) # test2
	# # testPatch1 = comparePatches.Patch(556, 20, sigma)  # test3
	# testPatch1 = comparePatches.Patch(379, 424, sigma) # test4
	# # testPatch1 = comparePatches.Patch(475, 267, sigma) # test5
	# # testPatch1 = comparePatches.Patch(440, 645, sigma) # test6
	# # testPatch1 = comparePatches.Patch(592, 494, sigma) # test7
	# # testPatch1 = comparePatches.Patch(437, 645, sigma) # test8

	# testPatchIndex = 4

	# # testPatch1.computeRGBHistogram(img)
	# testPatch1.computeHSVHistogram(img)
	# comparePatches.drawPatchesOnImg(np.copy(img), testPatch1)
	# # cv2.imwrite("testPatchHSV/GaussianWindowSubPatches/new_testPatch{x}_{folder}_{file}_simga{i}.jpg".format(x = testPatchIndex, folder = folderName, file = imgName[0:imgName.find(".")], i = sigma), comparePatches.drawPatchesOnImg(np.copy(img), testPatch1))
	
	# 1. Find the most distinguishable patches in the test1 img
	distinguishablePatches = comparePatches.findDistinguishablePatches(img, sigma)
	distinguishablePatches = distinguishablePatches[0:10]
	imgToSave = comparePatches.drawPatchesOnImg(np.copy(img), distinguishablePatches, False)
	cv2.imwrite("matches/HSV_cornerWeight_0.5_MostDistinguishablePatch_{folder}_{file1}_{file2}_simga{i}_GaussianWindowOnAWhole.jpg".format(folder = folderName, file1 = imgName[0:imgName.find(".")], file2 = imgToMatchName[0:imgToMatchName.find(".")],  i = sigma), imgToSave)
	
	# 2. Match the distinguishablePatches to every patches in the test2 img
	# 2.0 Extract patches to match in test2 img with different windowScales
	patchStep = 0.5 # shift the patch by 1/4 patch len
	matchPatches_origin = comparePatches.extractPatches(imgToMatch, sigma, patchStep)
	print "length of matchPatches:", len(matchPatches_origin)
	scale = 8 # window pixel window for up and down scaling
	patchesArr = []
	patchesArr.append(matchPatches_origin)

	for level in range(-2, 3):
		if(level != 0):
			patchesArr.append(comparePatches.extractPatches(imgToMatch, sigma + scale * level, patchStep))
	for i in range(0, len(patchesArr)):
		print "len(patchesArr[{i}]):".format(i = i),len(patchesArr[i])

	# 2.1 Compute the Color Hist of the matchPatches
	for index in range(0, len(patchesArr)):
		matchPatches = patchesArr[index]
		for i in range(0, len(matchPatches)):
			# print "computeRGBHistogram for matchPatches[{i}]".format(i = i) 
			# matchPatches[i].computeRGBHistogram(imgToMatch)j
			print "computeHSVHistogram for matchPatches[{i}]".format(i = i) 
			matchPatches[i].computeHSVHistogram(imgToMatch)
	# imgToSave = comparePatches.drawPatchesOnImg(np.copy(imgToMatch), testFindOnePatchMatch(testPatch1, patchesArr))
	# cv2.imwrite("testPatchHSV/GaussianWindowSubPatches/new_testPatch{x}_GoodMatches_{folder}_{file}_simga{i}_shiftBy{step}_gaussianWindowOnAWhole.jpg".format(x = testPatchIndex, folder = folderName, file = imgName[0:imgName.find(".")], i = sigma, step = patchStep), imgToSave)
	# raise ValueError("purpose stop for test patch only")
	
	# 2.2 do the comparison of each distinguishable patch against the matchPatches(array)
	patchesFound = []
	k = 1 # 1 means looking at just the best match at each window size level only
	for i in range(0, len(distinguishablePatches)):
		patchToMatch = distinguishablePatches[i]
		goodMatchesArr = [] # total good matches array for good matches at different window size level

		for j in range(0, len(patchesArr)): # for each window size level, find the best matches
			goodMatches = findBestMatches(patchToMatch, patchesArr[j],k)
			goodMatchesArr.append(goodMatches) # append to the total good matches array

		overAllGoodMatches = [item for sublist in goodMatchesArr for item in sublist] # flatten the list of list -> list
		print "overAllGoodMatches len should be {x}:".format(x = k*len(patchesArr)), len(overAllGoodMatches)
		bests = findBestMatches(patchToMatch, overAllGoodMatches,1) # rank the best patches found at different scale level
		patchesFound.append(bests[0]) # pick the best one

	print "len of best patch matches found, should be 10:", len(patchesFound)

	# 2.3 plot out the found match patches
	imgToSave = comparePatches.drawPatchesOnImg(np.copy(imgToMatch), patchesFound, False)
	# imgToSave = comparePatches.drawPatchesOnImg(np.copy(imgToMatch), overAllGoodMatches)
	# imgToSave = comparePatches.drawPatchesOnImg(np.copy(imgToMatch), findBestMatches(patchToMatch, overAllGoodMatches))
	# cv2.imwrite("testPatch{x}_GoodMatches_{folder}_{file}_simga{i}.jpg".format(x = testPatchIndex, folder = folderName, file = imgName[0:imgName.find(".")], i = sigma), imgToSave)
	cv2.imwrite("matches/HSV_cornerWeight_0.5_MostDistinguishableMatch_{folder}_{file1}_{file2}_simga{i}_moveby{step}_scaleChange{scale}__GaussianWindowOnAWhole.jpg".format(folder = folderName, file1 = imgName[0:imgName.find(".")], file2 = imgToMatchName[0:imgToMatchName.find(".")],  i = sigma, step = patchStep, scale = scale), imgToSave)

	return

if __name__ == "__main__":
	main()