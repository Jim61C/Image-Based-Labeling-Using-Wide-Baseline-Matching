import numpy as np
import cv2
import matchPatches
import saveLoadPatch
import comparePatches


class clickRecorder(object):
	sigma = 39
	groundTruth =[]
	testPatches = []
	imgToMatchPrev = []
	
	imgToMatch = None
	count = 0

	path = None

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

	def plotBaseImg(self, test_folder_name, image_db):
		self.imgToMatch = cv2.imread("{image_db}/{test_folder_name}/test1.jpg".format(image_db = image_db, test_folder_name = test_folder_name), 1)
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


def main():
	"""For clicking on target image for groundTruth"""
	test_folder_name = raw_input("Please input the testset name: ")
	folder_suffix = "_UniqueAlgo3_Jensen_Shannon_Divergence"
	upperPath = "testAlgo3"
	image_db = "images"
	my_click_recorder = clickRecorder()
	# set path
	my_click_recorder.setPath(test_folder_name, folder_suffix ,upperPath)
	# plot the distinguishable patches on base image
	my_click_recorder.plotBaseImgWithPatches(test_folder_name, folder_suffix ,upperPath)
	# plot the target image for user to click
	my_click_recorder.plotTargetImg(test_folder_name, image_db)
	# save the groundTruth if validated
	my_click_recorder.saveGroundTruth(test_folder_name)

	"""For clicking on base image for unique patches"""
	# test_folder_name = raw_input("Please input the testset name: ")
	# image_db = "images"
	# upperPath = "testAlgo2"
	# folder_suffix = "_eyeballed_unique_patches"
	# my_click_recorder = clickRecorder()
	# my_click_recorder.plotBaseImg(test_folder_name, image_db)
	# my_click_recorder.saveBaseImgUniquePatches(test_folder_name, folder_suffix, upperPath)




	return

if __name__ == "__main__":
	main()