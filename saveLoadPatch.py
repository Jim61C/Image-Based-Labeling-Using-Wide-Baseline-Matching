import numpy as np
import comparePatches
import pickle
import cv2
import csv

def load_data(filename):
    try:
        with open(filename) as f:
            patches = pickle.load(f)
    except:
        patches = []
    return patches

def save_data(data, filename):
    with open(filename, "wb") as f:
        pickle.dump(data, f)

# Input: patches is a flattened list of matchPatches
# store as a csv file with ['x', 'y', 'size', 'correspondingPatchIndex']
def savePatchMatches(patches, level, filename):
	"""
	patches: a flattened list of patches;
	level: number of patches corresponding to the same testPatch (Default is 5)
	"""
	with open(filename, 'w') as csvfile:
		fieldnames = ['x', 'y', 'size', 'correspondingPatchIndex', 'featureSet', 'LDAScore']
		writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
		
		writer.writeheader()
	
		n_test = len(patches)/level
		for i in range (0, n_test):
			for j in range(0, level):
				#Save information on patches[i*level + j]
				writer.writerow({ \
					'x': patches[i*level + j].x , \
					'y':patches[i*level + j].y , \
					'size':patches[i*level + j].size, \
					'correspondingPatchIndex':i, \
					'featureSet': patches[i*level + j].feature_to_use, \
					'LDAScore':patches[i*level + j].LDAFeatureScore})				
	return


# will load patches in terms of original corresponding patches
# return result is a list of list of patches: [[matches for patch0], [matches for patch1]...]
def loadPatchMatches(filename):
	patchMatches = []
	with open(filename) as csvfile:
		reader = csv.DictReader(csvfile)

		patches = []
		prePatchMatchIndex = 0
		for row in reader:
			newPatchMatchIndex = int(row['correspondingPatchIndex'])
			# print "prePatchMatchIndex:", prePatchMatchIndex
			# print "newPatchMatchIndex:", newPatchMatchIndex
			if(newPatchMatchIndex > prePatchMatchIndex):
				# print "need to append the list of patches to patchMatches", prePatchMatchIndex
				prePatchMatchIndex = newPatchMatchIndex
				patchMatches.append(patches)
				patches = []

			patches.append(comparePatches.Patch(int(row['x']),int(row['y']),int(row['size'])))

		if(len(patches) > 0):
			patchMatches.append(patches)

	return patchMatches



def main():
	# img = cv2.imread("images/testset7/test1.jpg", 1)
	# testPatches = []
	# sigma = 39
	# testPatches.append(comparePatches.Patch(433, 792, sigma)) # test0
	# testPatches.append(comparePatches.Patch(325, 899, sigma)) # test1
	# testPatches.append(comparePatches.Patch(700, 759, sigma)) # test2
	# testPatches.append(comparePatches.Patch(700, 530, sigma)) # test3
	# testPatches.append(comparePatches.Patch(484, 352, sigma)) # test4
	# testPatches.append(comparePatches.Patch(80, 722, sigma)) # test5
	# testPatches.append(comparePatches.Patch(162, 455, sigma)) # test6
	# savePatchMatches(testPatches, 1, "./testPatchHSV/temp.csv")

	# matchPatches = loadPatchMatches("./testPatchHSV/temp.csv")
	# print "len(matchPatches):", len(matchPatches)
	# print matchPatches
	# for i in range(0, len(matchPatches)):
	# 	patches = matchPatches[i]
	# 	for j in range(0, len(patches)):
	# 		print patches[j].x, ",", patches[j].y,  ",", patches[j].size

	return


if __name__ == "__main__":
	main()