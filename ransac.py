import comparePatches
import plotStatistics
import saveLoadPatch
import numpy as np
import cv2
from cv2 import cv
from matplotlib import pyplot as plt
import math
import copy
import os
import time
import sys

def ransacPruning(test_patches, matches_found, n = 3, k = 20, t = 1.0, d = 4):
	"""
	n - the number of random points to pick every iteration in order to create the transform
	k - the number of iterations to run
	t - the threshold for the square distance for a point to be considered as a match
	d - the number of points that need to be matched for the transform to be valid
	"""
	return


def ransacPruningOpenCV(test_patches, matches_found, n = 3, k = 20, t = 1.0, d = 4):
	patch_key_points = []
	match_key_points = []
	for i in range(0, len(test_patches)):
		patch_key_points.append(cv2.KeyPoint(test_patches[i].y, test_patches[i].x, test_patches[i].size))
	for i in range(0, len(matches_found)):
		match_key_points.append(cv2.KeyPoint(matches_found[i].y, matches_found[i].x, matches_found[i].size))

	pts1 = np.array([keypoint.pt for keypoint in patch_key_points])
	pts2 = np.array([keypoint.pt for keypoint in match_key_points])

	for i in range(0, len(pts1)):
		print pts1[i], pts2[i]

	F, mask = cv2.findFundamentalMat(points1 = pts1, points2 = pts2, method = cv.CV_FM_RANSAC, param1 = 5, param2 = 0.5)


	"""If the F found is all zeros, which means not able to find Fundamental matrix"""
	if ((F == 0).all()):
		print "Can't find Homography need to mannually prune this image"
	else:
		print "Fundamental Matrix:", F
		print "mask:"
		for i in range(0, len(mask)):
			print "keypoint[", i, "]", "good match?", mask[i]

	return

def main():
	upperPath = "testAlgo3"

	"""Unit Test with testset7"""
	# test_folder_name = "testset7"
	# folder_suffix = "_full_algo_top20_unique_patches_descriptor_based_testset7_taylored"

	test_folder_name = "testset_orchid20"
	folder_suffix = "_integralImageHS_top20_unique_patches_descriptor_based_point_01_Harris_high_response_only_normalizedJS"
	test1_img_name = "test1.jpg"
	test2_img_name = "test3.jpg"
	sigma = 39
	
	list_of_test_patches = saveLoadPatch.loadUniquePatchesWithFeatureSet(\
		"{upperPath}/{folderToSave}/{testFolder}/DistinguishablePatch_{folder}_{file}_simga{i}_GaussianWindowOnAWhole.csv".format(
		upperPath = upperPath,
		folderToSave = "GaussianWindowOnAWhole", 
		testFolder = test_folder_name +folder_suffix, 
		folder = test_folder_name, 
		file = test1_img_name[:test1_img_name.find(".")], 
		i = sigma))
	test_patches = []
	for i in range(0, len(list_of_test_patches)):
		test_patches.append(list_of_test_patches[i])
		

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
	matches_found = []
	for i in range(0, len(test_patch_matches)):
		matches_found.append(test_patch_matches[i][0])

	img = cv2.imread("images/{test_folder_name}/{test1_img_name}".format(test_folder_name = test_folder_name, \
		test1_img_name = test1_img_name), 1)
	img_to_match = cv2.imread("images/{test_folder_name}/{test2_img_name}".format(test_folder_name = test_folder_name, \
		test2_img_name = test2_img_name), 1)
	ransacPruningOpenCV(test_patches, matches_found)
	comparePatches.drawMatchesOnImg(img, img_to_match, test_patches, matches_found)

	return


if __name__ == "__main__":
	main()