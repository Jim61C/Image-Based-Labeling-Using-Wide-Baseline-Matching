import numpy as np
import cv2
import matchPatches
import saveLoadPatch
import comparePatches


class clickRecorder(object):
	sigma = 39
	groundTruth =[]
	imgToMatchPrev = []
	imgToMatch = None
	count = 0

	path = None

	def plotBaseImg(self, test_folder_name, folder_suffix, upperPath):	
		path = upperPath + "/GaussianWindowOnAWhole/" + test_folder_name + folder_suffix
		print path
		self.path = path
		img = cv2.imread("{path}/DistinguishablePatch_{folder}_{file}_simga{i}_GaussianWindowOnAWhole.jpg".format( \
			path = path , \
			folder = test_folder_name, \
			file = "test1", \
			i = self.sigma), 1)
		print img.shape
		

		testPatches = []
		listOfTestPatches = saveLoadPatch.loadPatchMatches("{upperPath}/{folderToSave}/{testFolder}/DistinguishablePatch_{folder}_{file}_simga{i}_GaussianWindowOnAWhole.csv".format(
			upperPath = upperPath,
			folderToSave = "GaussianWindowOnAWhole", 
			testFolder = test_folder_name +folder_suffix, 
			folder = test_folder_name, 
			file = "test1", 
			i = self.sigma))
		for i in range(0, len(listOfTestPatches)):
			testPatches.append(listOfTestPatches[i][0])
		for i in range(0, len(testPatches)):
			cv2.circle(img,(testPatches[i].y, testPatches[i].x), 2, (0,0,255), -1)
			cv2.putText(img, "{i}".format(i =i), (testPatches[i].y, testPatches[i].x), cv2.FONT_HERSHEY_SIMPLEX, 0.5 ,(0,0,255),1)

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
	def plotTargetImg(self, test_folder_name, image_db):
		self.imgToMatch = cv2.imread("{image_db}/{test_folder_name}/test2.jpg".format(image_db = image_db, test_folder_name = test_folder_name), 1)
		cv2.imshow("targetImage", self.imgToMatch)
		cv2.setMouseCallback("targetImage", self.mouseEventCallback)
		self.undoLoop()
		print len(self.groundTruth)
		self.saveGroundTruth(test_folder_name)

	def saveGroundTruth(self, test_folder_name):
		print "path to save the groundTruth:",self.path
		saveLoadPatch.savePatchMatches(self.groundTruth, 1, \
		"{path}/GroudTruth_{folder}_{file1}_{file2}_simga{i}_GaussianWindowOnAWhole.csv".format( \
			path = self.path , \
			folder = test_folder_name, \
			file1 = "test1", \
			file2 = "test2", \
			i = self.sigma))

def main():
	test_folder_name = raw_input("Please input the testset name: ")
	folder_suffix = "_UniqueAlgo2_Jensen_Shannon_Divergence"
	upperPath = "testAlgo2"
	image_db = "images"
	my_click_recorder = clickRecorder()
	my_click_recorder.plotBaseImg(test_folder_name, folder_suffix ,upperPath)
	my_click_recorder.plotTargetImg(test_folder_name, image_db)
	return

if __name__ == "__main__":
	main()