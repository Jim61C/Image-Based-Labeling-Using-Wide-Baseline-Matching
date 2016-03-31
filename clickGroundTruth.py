import numpy as np
import cv2
import matchPatches
import saveLoadPatch
import comparePatches
import feature_modules
import saveLoadPatch
from feature_modules import utils
import os

class clickRecorder(object):
	sigma = 39
	groundTruth =[]
	testPatches = []
	imgToMatchPrev = []
	
	imgToMatchOrigin = None
	imgToMatch = None
	count = 0

	centre_feature_count = 0
	subsquare_paradigm_count = 0
	centre_hog_feature_count = 0
	# update centre_feature_count to be the next index
	for feature_pkl in os.listdir(utils.FEATURES_GENERATED_FOLDER):
		if (feature_pkl.find(utils.CENTRE_PARADIGM_FEATURE_PREFIX) != -1):
			idx = int(feature_pkl[feature_pkl.find(utils.CENTRE_PARADIGM_FEATURE_PREFIX)+ \
				len(utils.CENTRE_PARADIGM_FEATURE_PREFIX):feature_pkl.find(".")])
			if (idx >= centre_feature_count):
				centre_feature_count = idx + 1
		elif (feature_pkl.find(utils.SUBSQUARE_PARADIGM_FEATURE_PREFIX) != -1):
			idx = int(feature_pkl[feature_pkl.find(utils.SUBSQUARE_PARADIGM_FEATURE_PREFIX)+ \
				len(utils.SUBSQUARE_PARADIGM_FEATURE_PREFIX):feature_pkl.find(".")])
			if (idx >= subsquare_paradigm_count):
				subsquare_paradigm_count = idx + 1
		elif (feature_pkl.find(utils.CENTRE_HOG_PARADIGM_FEATURE_PREFIX) != -1):
			idx = int(feature_pkl[feature_pkl.find(utils.CENTRE_HOG_PARADIGM_FEATURE_PREFIX)+ \
				len(utils.CENTRE_HOG_PARADIGM_FEATURE_PREFIX):feature_pkl.find(".")])
			if (idx >= centre_hog_feature_count):
				centre_hog_feature_count = idx + 1

	print "centre_feature_count:", centre_feature_count
	path = None
	detect_shape = False

	def setDetectShape(self, detect_shape):
		self.detect_shape = detect_shape

	def msgBox(self, message, height = 200, width = 800):
		blank_image = np.zeros((int(height),int(width),3), np.uint8)
		blank_image[:,:] = (255,255,255)
		cv2.putText(blank_image, message, (50, height/2), cv2.FONT_HERSHEY_SIMPLEX, 0.5 ,(0,0,0),1)
		cv2.imshow("msgBox", blank_image)
		cv2.waitKey(0)
		cv2.destroyWindow("msgBox")

	def setPath(self, test_folder_name, folder_suffix, upperPath):
		path = upperPath + "/GaussianWindowOnAWhole/" + test_folder_name + folder_suffix
		print path
		self.path = path

	def plotBaseImgWithPatches(self, test_folder_name, folder_suffix, upperPath):	
		img = cv2.imread("{path}/DistinguishablePatch_{folder}_{file}_simga{i}_GaussianWindowOnAWhole.jpg".format( \
			path = self.path , \
			folder = test_folder_name, \
			file = "test1", \
			i = self.sigma), 1)
		
		print "DistinguishablePatch_{folder}_{file}_simga{i}_GaussianWindowOnAWhole.jpg".format( \
			folder = test_folder_name, \
			file = "test1", \
			i = self.sigma)
		print img.shape
		

		testPatches = []
		listOfTestPatches = saveLoadPatch.loadPatchMatches("{path}/DistinguishablePatch_{folder}_{file}_simga{i}_GaussianWindowOnAWhole.csv".format(
			path = self.path,
			folder = test_folder_name, 
			file = "test1", 
			i = self.sigma))
		for i in range(0, len(listOfTestPatches)):
			testPatches.append(listOfTestPatches[i][0])
		for i in range(0, len(testPatches)):
			cv2.circle(img,(testPatches[i].y, testPatches[i].x), 2, (0,0,255), -1)
			cv2.putText(img, "{i}".format(i =i), (testPatches[i].y, testPatches[i].x), cv2.FONT_HERSHEY_SIMPLEX, 0.5 ,(0,0,255),1)
		
		self.testPatches = testPatches
		
		cv2.imshow("Base Img:", img)
		cv2.waitKey(0)

	def mouseEventCallback(self, event, x, y, flags, user_data):
		if event == 1:
			print(x, y)
			new_patch = comparePatches.Patch(y,x,self.sigma)
			self.imgToMatchPrev.append(np.copy(self.imgToMatch)) # retain the last state of imgToMatch for matching purpose
			comparePatches.drawPatchesOnImg(self.imgToMatch, new_patch, show = False, gradiant = None, color = (0,0,255))
			cv2.circle(self.imgToMatch,(new_patch.y, new_patch.x), 2, (0,0,255), -1)
			cv2.putText(self.imgToMatch, "{i}".format(i =self.count), (new_patch.y, new_patch.x), cv2.FONT_HERSHEY_SIMPLEX, 0.5 ,(0,0,255),1)
			self.count += 1
			self.groundTruth.append(new_patch)
			cv2.imshow("targetImage", self.imgToMatch)

	def undoLoop(self):
		if cv2.waitKey(0) == ord('u'):
			print "pressd u!"
			if(len(self.imgToMatchPrev) > 0):
				print "undo last click!"
				self.imgToMatch = self.imgToMatchPrev.pop()
				self.count -= 1
				self.groundTruth.pop()
				cv2.imshow("targetImage", self.imgToMatch)
			self.undoLoop()
		else:
			print "pressed other keys, end process"

	def plotBaseImg(self, test_folder_name, image_db, base_img_name = "test1.jpg"):
		self.imgToMatch = cv2.imread("{image_db}/{test_folder_name}/{base_img_name}".format( \
			image_db = image_db, \
			test_folder_name = test_folder_name, \
			base_img_name = base_img_name), 1)
		self.imgToMatchOrigin = np.copy(self.imgToMatch)
		cv2.imshow("targetImage", self.imgToMatch)
		cv2.setMouseCallback("targetImage", self.mouseEventCallback)
		self.undoLoop()
		print "number of patches clicked on base image: ", len(self.groundTruth)

	def plotTargetImg(self, test_folder_name, image_db):
		self.imgToMatch = cv2.imread("{image_db}/{test_folder_name}/test2.jpg".format(image_db = image_db, test_folder_name = test_folder_name), 1)
		cv2.imshow("targetImage", self.imgToMatch)
		cv2.setMouseCallback("targetImage", self.mouseEventCallback)
		self.undoLoop()
		print "number of patches clicked on target image: ", len(self.groundTruth)

	def saveGroundTruth(self, test_folder_name):
		if(len(self.groundTruth) == len(self.testPatches)):
			print "path to save the groundTruth:",self.path
			saveLoadPatch.savePatchMatches(self.groundTruth, 1, \
			"{path}/GroundTruth_{folder}_{file1}_{file2}_simga{i}_GaussianWindowOnAWhole.csv".format( \
				path = self.path , \
				folder = test_folder_name, \
				file1 = "test1", \
				file2 = "test2", \
				i = self.sigma))
		else:
			self.msgBox("Please make sure number of clicks match the number of patches on base image!")
			cv2.imshow("targetImage", self.imgToMatch)
			self.undoLoop()
			self.saveGroundTruth(test_folder_name)

	def saveBaseImgUniquePatches(self, test_folder_name, folder_suffix, upperPath):
		path_unique_patches = matchPatches.createFolder(upperPath, "GaussianWindowOnAWhole", test_folder_name, folder_suffix)
		cv2.imwrite("{path}/{folder}_{file}_simga{i}_eyeballed_unique_patches.jpg".format(\
			path = path_unique_patches, \
			folder = test_folder_name, \
			file = "test1", \
			i = self.sigma) ,self.imgToMatch)
		saveLoadPatch.savePatchMatches(self.groundTruth, 1, \
		"{path}/DistinguishablePatch_{folder}_{file}_simga{i}_GaussianWindowOnAWhole.csv".format( \
			path = path_unique_patches , \
			folder = test_folder_name, \
			file = "test1", \
			i = self.sigma))

	def saveFeature(self, feature_obj, feature_id):
		saveLoadPatch.save_object(feature_obj, "{features_generated_folder}/{feature_id}.pkl".format(\
			features_generated_folder = utils.FEATURES_GENERATED_FOLDER, \
			feature_id = feature_id))

	def fitFeatures(self, test_folder_name, folder_suffix, upperPath):
		cv2.imshow("imgToMatch original",self.imgToMatchOrigin)
		cv2.waitKey(0)
		for i in range(0, len(self.groundTruth)):
			print "\nin clickRecorder, checking clicked patch:", i
			patch = self.groundTruth[i]
			"""centre_paradigm"""
			potential_centre_feature_id = "{centre_paradigm}{count}".format( \
				centre_paradigm = utils.CENTRE_PARADIGM_FEATURE_PREFIX, \
				count = self.centre_feature_count)
			potential_centre_feature = feature_modules.FeatureCentreParadigm(patch, potential_centre_feature_id)
			if(potential_centre_feature.fitParadigm(self.imgToMatchOrigin)):
				self.centre_feature_count = self.centre_feature_count + 1
				print "successfully constructed feature centre_paradigm for patch ", i, " clicked"
				self.saveFeature(potential_centre_feature, potential_centre_feature_id)

			"""subsquare_paradigm"""
			potential_subsquare_paradigm_id = "{subsquare_paradigm}{count}".format( \
				subsquare_paradigm = utils.SUBSQUARE_PARADIGM_FEATURE_PREFIX, \
				count = self.subsquare_paradigm_count)
			potential_subsquare_paradigm_feature = feature_modules.FeatureSubSquareParadigm(patch, potential_subsquare_paradigm_id)
			if(potential_subsquare_paradigm_feature.fitParadigm(self.imgToMatchOrigin)):
				self.subsquare_paradigm_count = self.subsquare_paradigm_count + 1
				print "successfully constructed FeatureSubSquareParadigm for patch ", i, " clicked"
				self.saveFeature(potential_subsquare_paradigm_feature, potential_subsquare_paradigm_id)

			if (self.detect_shape):
				"""centre_hog_paradigm"""
				potential_centre_hog_feature_id = "{centre_hog_paradigm}{count}".format( \
					centre_hog_paradigm = utils.CENTRE_HOG_PARADIGM_FEATURE_PREFIX, \
					count = self.centre_hog_feature_count)
				potential_centre_hog_feature = feature_modules.FeatureCentreHOGParadigm(patch, potential_centre_hog_feature_id)
				if(potential_centre_hog_feature.fitParadigm(self.imgToMatchOrigin)):
					self.centre_hog_feature_count = self.centre_hog_feature_count + 1
					print "successfully constructed feature centre_hog_paradigm for patch ", i, " clicked"
					self.saveFeature(potential_centre_hog_feature, potential_centre_hog_feature_id)


def main():
	"""For clicking on target image for groundTruth"""
	# test_folder_name = raw_input("Please input the testset name: ")
	# folder_suffix = "_UniqueAlgo3_Jensen_Shannon_Divergence"
	# upperPath = "testAlgo3"
	# image_db = "images"
	# my_click_recorder = clickRecorder()
	# # set path
	# my_click_recorder.setPath(test_folder_name, folder_suffix ,upperPath)
	# # plot the distinguishable patches on base image
	# my_click_recorder.plotBaseImgWithPatches(test_folder_name, folder_suffix ,upperPath)
	# # plot the target image for user to click
	# my_click_recorder.plotTargetImg(test_folder_name, image_db)
	# # save the groundTruth if validated
	# my_click_recorder.saveGroundTruth(test_folder_name)

	"""For clicking on base image for unique patches"""
	test_folder_name = raw_input("Please input the testset name: ")
	base_img_name = raw_input("Please input the testset image name: ")
	image_db = "images"
	upperPath = "testAlgo3"
	folder_suffix = "_eyeballed_unique_patches_feature_construction"
	my_click_recorder = clickRecorder()
	detect_shape = raw_input("detect shape? (y/n)")
	if (detect_shape == "y"):
		my_click_recorder.setDetectShape(True)
	else:
		my_click_recorder.setDetectShape(False)

	my_click_recorder.plotBaseImg(test_folder_name, image_db, base_img_name)
	my_click_recorder.saveBaseImgUniquePatches(test_folder_name, folder_suffix, upperPath)
	my_click_recorder.fitFeatures(test_folder_name, folder_suffix, upperPath)



	return

if __name__ == "__main__":
	main()