import cv2
from cv2 import cv
import numpy as np 
import drawMatches
import comparePatches
import saveLoadPatch
import utils

def filter_matches(matches, ratio=0.75):
		"""
		Filter out good matches from a list of matches.

		:param matches: The matches to be filtered.
		:param ratio: The threshold used in filtering, the smaller, the better.
		:return: The filtered matches.
		"""
		filtered_matches = []
		for m in matches:
			if len(m) == 2 and m[0].distance < m[1].distance * ratio:
				filtered_matches.append(m[0])

		return filtered_matches

def populateFeatureMatchingStatistics(test_folder_name, test1_img_name, test2_img_name):
	sigma = 39
	img, imgToMatch, test_patches, matches_found = runSIFT(test_folder_name, test1_img_name, test2_img_name)
	ground_truth = []
	listOfGroundTruth = saveLoadPatch.loadPatchMatches(\
		"{path}/GroundTruth_{folder}_{file1}_{file2}_simga{i}_GaussianWindowOnAWhole.csv".format(
		path = "testSIFT",
		folder = test_folder_name, 
		file1 = test1_img_name[:test1_img_name.find(".")], 
		file2 = test2_img_name[:test2_img_name.find(".")], 
		i = sigma))
	for i in range(0, len(listOfGroundTruth)):
		ground_truth.append(listOfGroundTruth[i][0])

	correct_color = (0,0,255)
	wrong_color = (255,0,0)
	custom_colors = []
	for i in range(0, len(ground_truth)):
		if (utils.isGoodMatch(matches_found[i], ground_truth[i])):
			custom_colors.append(correct_color)
		else:
			custom_colors.append(wrong_color)

	distinguished_match = comparePatches.drawMatchesOnImg(np.copy(img), np.copy(imgToMatch), test_patches, matches_found, \
		show = True, custom_colors = custom_colors)
	cv2.imwrite("testSIFT/{savefilename}.jpg".format(\
		savefilename = test_folder_name + test1_img_name[0:test1_img_name.find(".")] + \
		test2_img_name[0:test2_img_name.find(".")]), distinguished_match)

	# comparePatches.drawMatchesOnImg(np.copy(img), np.copy(imgToMatch), test_patches, ground_truth, \
	# 	show = True)


def runSIFT(test_folder_name, test1_img_name, test2_img_name):
	NUM_GOOD_MATCH = 20
	img = cv2.imread("images/{test_folder_name}/{test1_img_name}".format(test_folder_name = test_folder_name, test1_img_name = test1_img_name), 1)
	img_gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	imgToMatch = cv2.imread("images/{test_folder_name}/{test2_img_name}".format(test_folder_name = test_folder_name, test2_img_name = test2_img_name), 1)
	imgToMatch_gray = cv2.cvtColor(imgToMatch, cv2.COLOR_BGR2GRAY)
	
	# cv2.imshow("img",img)
	# cv2.waitKey(0)

	# cv2.imshow("imgToMatch", imgToMatch)
	# cv2.waitKey(0)

	sift = cv2.SIFT()

	features1, desc1 = sift.detectAndCompute(img_gray,None)
	features2, desc2 = sift.detectAndCompute(imgToMatch_gray,None)

	FLANN_INDEX_KDTREE = 1
	flann_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
	matcher = cv2.FlannBasedMatcher(flann_params, {})
	count_for_best_matches_in_knn = 2

	matches = matcher.knnMatch(desc1, trainDescriptors=desc2,
										k=count_for_best_matches_in_knn)
	matches = filter_matches(matches)

	matches = sorted(matches, key= lambda match: match.distance)
	print "len(matches)",len(matches)
	print "len(features1)",len(features1)
	print "len(features2)", len(features2)

	test_patches = []
	matches_found = []
	for i in range(0, NUM_GOOD_MATCH):
		img1_idx = matches[i].queryIdx
		img2_idx = matches[i].trainIdx
		# x is col, y is row
		(x1,y1) = features1[img1_idx].pt
		(x2,y2) = features2[img2_idx].pt

		size1 = features1[img1_idx].size
		size2 = features2[img2_idx].size

		this_test_patch = comparePatches.Patch(int(y1), int(x1), int(size1), initialize_features = False)
		test_patches.append(this_test_patch)

		this_match_found = comparePatches.Patch(int(y2), int(x2), int(size2), initialize_features = False)
		matches_found.append(this_match_found)

	img_with_test_patches = comparePatches.drawPatchesOnImg(np.copy(img), test_patches, mark_sequence = True)
	cv2.imwrite("testSIFT/test_patches_{savefilename}.jpg".format(\
		savefilename = test_folder_name + test1_img_name[0:test1_img_name.find(".")] + test2_img_name[0:test2_img_name.find(".")]), img_with_test_patches)
	
	match_img = drawMatches.drawMatches(np.copy(img),features1,np.copy(imgToMatch),features2,matches[:NUM_GOOD_MATCH], draw_size = True)
	cv2.imshow("match_img", match_img)
	cv2.waitKey(0)
	cv2.imwrite("testSIFT/{savefilename}.jpg".format(savefilename = test_folder_name + test1_img_name[0:test1_img_name.find(".")] + test2_img_name[0:test2_img_name.find(".")]), match_img)
	return img, imgToMatch, test_patches, matches_found


def main():
	# runSIFT("testset_illuminance1", "test1.jpg", "test2.jpg")
	# runSIFT("testset_illuminance2", "test1.jpg", "test2.jpg")
	# runSIFT("testset_rotation1", "test1.jpg", "test2.jpg")
	# runSIFT("testset_rotation2", "test1.jpg", "test2.jpg")
	# runSIFT("testset4", "test1.jpg", "test2.jpg")
	# runSIFT("testset8", "test1.jpg", "test2.jpg")
	populateFeatureMatchingStatistics("testset8", "test1.jpg", "test2.jpg")

if __name__ == "__main__":
	main()