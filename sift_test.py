import cv2
import numpy as np 
import drawMatches

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


def runSIFT(test_folder_name, test1_img_name, test2_img_name):
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

	match_img = drawMatches.drawMatches(img,features1,imgToMatch,features2,matches[:20])
	cv2.imshow("match_img", match_img)
	cv2.waitKey(0)
	cv2.imwrite("testSIFT/{savefilename}.jpg".format(savefilename = test_folder_name + test1_img_name[0:test1_img_name.find(".")] + test2_img_name[0:test2_img_name.find(".")]), match_img)


def main():
	# runSIFT("testset_illuminance1", "test1.jpg", "test2.jpg")
	# runSIFT("testset_illuminance2", "test1.jpg", "test2.jpg")
	# runSIFT("testset_rotation1", "test1.jpg", "test2.jpg")
	# runSIFT("testset_rotation2", "test1.jpg", "test2.jpg")
	# runSIFT("testset4", "test1.jpg", "test2.jpg")
	runSIFT("testset7", "test1.jpg", "test3.jpg")

if __name__ == "__main__":
	main()