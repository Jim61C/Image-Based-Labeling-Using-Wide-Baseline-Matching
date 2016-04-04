import numpy as np
from matplotlib import pyplot as plt
import comparePatches
from sklearn.preprocessing import normalize
import os

def plotColorHistogram(patch, img, path, fname, save = True, show = True, histToUse = "HSV", useGaussian = True):
	"""
	path: Exmaple: '/.../hists'
	"""
	fig = None
	if(histToUse == "HSV"):
		if(len(patch.HSVHistArr) == 0): 
			patch.computeHSVHistogram(cv2.cvtColor(img.astype(np.float32), cv2.COLOR_BGR2HSV), useGaussian, True)
		# fig = plt.plot(patch.HSVHist)
		# plt.xlabel(fname)
		# if(save):
		# 	plt.savefig(path+"/"+fname+".png")
		# if(show):
		# 	plt.show(fig)
		# plt.clf()
		fig = plt.figure(1)
		# plt.subplot(351)
		# plt.bar(np.arange(len(patch.HueHist)),patch.HueHist, width = 1.0,color = 'r', label = fname+"_Hue")
		# plt.xlim(0, len(patch.HueHist))
		# plt.legend()
		ax = fig.add_subplot(3,5,1)
		ax.bar(np.arange(len(patch.HueHist)),patch.HueHist, width = 1.0,color = 'r', label = fname+"_Hue")
		ax.set_xlim(0, len(patch.HueHist))
		ax.legend()

		# plt.subplot(356)
		# plt.bar(np.arange(len(patch.SaturationHist)),patch.SaturationHist, width = 1.0, color = 'g', label = fname+"_Saturation")
		# plt.xlim(0, len(patch.SaturationHist))
		# plt.legend()
		ax = fig.add_subplot(3,5,6)
		ax.bar(np.arange(len(patch.SaturationHist)),patch.SaturationHist, width = 1.0, color = 'g', label = fname+"_Saturation")
		ax.set_xlim(0, len(patch.SaturationHist))
		ax.legend()

		ax = fig.add_subplot(3,5,11)
		ax.bar(np.arange(len(patch.ValueHist)),patch.ValueHist, width = 1.0, color = 'b',label = fname + "_Value")
		ax.legend()
		ax.set_xlim(0, len(patch.ValueHist))
		ax.set_xlabel(fname+"_HSVSeperate")

		for i in range(1, 5):
			ax = fig.add_subplot(3,5,i+1)
			ax.bar(np.arange(len(patch.HueHistArr[i])),patch.HueHistArr[i], width = 1.0, color = 'r',label = fname + "_SubPatch[{i}]_Hue".format(i = i))
			ax.legend()
			ax.set_xlim(0, len(patch.HueHistArr[i]))

			ax = fig.add_subplot(3,5,(i+1)+5)
			ax.bar(np.arange(len(patch.SaturationHistArr[i])),patch.SaturationHistArr[i], width = 1.0, color = 'g',label = fname + "_SubPatch[{i}]_Saturation".format(i = i))
			ax.legend()
			ax.set_xlim(0, len(patch.SaturationHistArr[i]))

			ax = fig.add_subplot(3,5,(i+1)+10)
			ax.bar(np.arange(len(patch.ValueHistArr[i])),patch.ValueHistArr[i], width = 1.0, color = 'b',label = fname + "_SubPatch[{i}]_Value".format(i = i))
			ax.legend()
			ax.set_xlim(0, len(patch.ValueHistArr[i]))
			ax.set_xlabel(fname+"_HSVSeperate_subpatch_{i}".format(i = i))
		
		if(save):
			plt.savefig(path+"/"+fname+"_HSVSeperate.png")
		if(show):
			plt.show()
		plt.clf()

		# plt.plot(patch.HueHist)
		# plt.xlabel(fname+"_Hue")
		# if(save):
		# 	plt.savefig(path+"/"+fname+"_Hue.png")
		# plt.clf()

		# plt.plot(patch.SaturationHist)
		# plt.xlabel(fname+"_Saturation")
		# if(save):
		# 	plt.savefig(path+"/"+fname+"_Saturation.png")
		# plt.clf()

		# plt.plot(patch.ValueHist)
		# plt.xlabel(fname + "_Value")
		# if(save):
		# 	plt.savefig(path+"/"+fname+"_Value.png")
		# plt.clf()
	return

def plotGivenHSVHists(path, fname, HueHist, SaturationHist, ValueHist, save = True, show = True):
	fig = plt.figure(1)
	plt.subplot(311)
	plt.bar(np.arange(len(HueHist)),HueHist, width = 1.0,color = 'r', label = fname+"_Hue")
	plt.xlim(0, len(HueHist))
	plt.legend()

	plt.subplot(312)
	plt.bar(np.arange(len(SaturationHist)),SaturationHist, width = 1.0, color = 'g', label = fname+"_Saturation")
	plt.xlim(0, len(SaturationHist))
	plt.legend()

	plt.subplot(313)
	plt.bar(np.arange(len(ValueHist)),ValueHist, width = 1.0, color = 'b', label = fname+"_Value")
	plt.xlim(0, len(ValueHist))
	plt.legend()

	if(save):
		plt.savefig(path+"/"+fname+"_HSVSeperate.png")
	if(show):
		plt.show()
	plt.clf()

def plotOneGivenHist(path,fname, Hist, save = True, show = True):
	plt.bar(np.arange(len(Hist)),Hist, width = 1.0,color = 'r', label = fname+"_Hist")
	plt.xlim(0, len(Hist))
	plt.ylim(0, 1.0)
	plt.legend()
	plt.xlabel(fname+"_Hist")

	if(save):
		plt.savefig(path+"/"+fname+"_Hist.png")
	if(show):
		plt.show()
	plt.clf()


def plotHSVHistCmp(path, fname, testPatchesHSVHist,testPatchLabel, matchesFoundHSVHist, matchFoundLabel, groundTruthHSVHist, groundTruthLabel, saveHist = True, displayHist = True):
	plt.figure(1)
	plt.subplot(311)
	plt.plot(testPatchesHSVHist, label = testPatchLabel)
	plt.legend()
	plt.subplot(312)
	plt.plot(matchesFoundHSVHist, label = matchFoundLabel)
	plt.legend()
	plt.subplot(313)
	plt.plot(groundTruthHSVHist, label = groundTruthLabel)
	
	plt.xlabel(fname)
	plt.legend()
	if(saveHist):
		plt.savefig(path+"/"+fname+".png")
	if(displayHist):
		plt.show()
	plt.clf()
	return
def plotHOGHistCmp(path, fname, testPatchesHOGHist,testPatchLabel, matchesFoundHOGHist, matchFoundLabel, groundTruthHOGHist, groundTruthLabel, saveHist = True, displayHist = True):
	plt.figure(1)
	plt.subplot(311)
	plt.bar(np.arange(len(testPatchesHOGHist)),testPatchesHOGHist, width = 1.0,color = 'r', label = testPatchLabel+"_HOG")
	plt.xlim(0, len(testPatchesHOGHist))
	# plt.ylim(0,1.0)
	plt.legend()

	plt.subplot(312)
	plt.bar(np.arange(len(matchesFoundHOGHist)),matchesFoundHOGHist, width = 1.0,color = 'r', label = matchFoundLabel+"_HOG")
	plt.xlim(0, len(matchesFoundHOGHist))
	# plt.ylim(0,1.0)
	plt.legend()

	plt.subplot(313)
	plt.bar(np.arange(len(groundTruthHOGHist)),groundTruthHOGHist, width = 1.0,color = 'r', label = groundTruthLabel+"_HOG")
	plt.xlim(0, len(groundTruthHOGHist))
	# plt.ylim(0,1.0)
	plt.xlabel(fname)
	plt.legend()

	if(saveHist):
		plt.savefig(path+"/"+fname+".png")
	if(displayHist):
		plt.show()
	plt.clf()
	return

def plotHSVSeperateHistCmp(path, fname, \
	testPatchHSVSeperateHists, testPatchLabel, \
	matchesFoundHSVSeperateHists, matchFoundLabel, \
	groundTruthHSVSepreateHists, groundTruthLabel, \
	saveHist = True, displayHist =True ):
	f, ((test_patch_hue_ax, test_patch_saturation_ax, test_patch_value_ax),(match_found_hue_ax, match_found_saturation_ax, match_found_value_ax), (ground_truth_hue_ax, ground_truth_saturation_ax, ground_truth_value_ax)) = plt.subplots(3, 3, sharex='col', sharey='row')
	f.set_figheight(10)
	f.set_figwidth(15)
	# test Patch H, S, V
	test_patch_hue_ax.bar(np.arange(len(testPatchHSVSeperateHists[0])),testPatchHSVSeperateHists[0], width = 1.0,color = 'r', label = testPatchLabel+"_Hue")
	test_patch_hue_ax.set_xlim(0, len(testPatchHSVSeperateHists[0]))
	test_patch_hue_ax.set_ylim(0,1.0)
	test_patch_hue_ax.legend()

	test_patch_saturation_ax.bar(np.arange(len(testPatchHSVSeperateHists[1])),testPatchHSVSeperateHists[1], width = 1.0,color = 'g', label = testPatchLabel+"_Saturation")
	test_patch_saturation_ax.set_xlim(0, len(testPatchHSVSeperateHists[1]))
	test_patch_saturation_ax.set_ylim(0,1.0)
	test_patch_saturation_ax.legend()		
	
	test_patch_value_ax.bar(np.arange(len(testPatchHSVSeperateHists[2])),testPatchHSVSeperateHists[2], width = 1.0,color = 'b', label = testPatchLabel+"_Value")
	test_patch_value_ax.set_xlim(0, len(testPatchHSVSeperateHists[2]))
	test_patch_value_ax.set_ylim(0,1.0)
	test_patch_value_ax.legend()

	# match found H, S, V
	match_found_hue_ax.bar(np.arange(len(matchesFoundHSVSeperateHists[0])),matchesFoundHSVSeperateHists[0], width = 1.0,color = 'r', label = matchFoundLabel+"_Hue")
	match_found_hue_ax.set_xlim(0, len(matchesFoundHSVSeperateHists[0]))
	match_found_hue_ax.set_ylim(0,1.0)
	match_found_hue_ax.legend()	

	match_found_saturation_ax.bar(np.arange(len(matchesFoundHSVSeperateHists[1])),matchesFoundHSVSeperateHists[1], width = 1.0,color = 'g', label = matchFoundLabel+"_Saturation")
	match_found_saturation_ax.set_xlim(0, len(matchesFoundHSVSeperateHists[1]))
	match_found_saturation_ax.set_ylim(0,1.0)
	match_found_saturation_ax.legend()	

	match_found_value_ax.bar(np.arange(len(matchesFoundHSVSeperateHists[2])),matchesFoundHSVSeperateHists[2], width = 1.0,color = 'b', label = matchFoundLabel+"_Value")
	match_found_value_ax.set_xlim(0, len(matchesFoundHSVSeperateHists[2]))
	match_found_value_ax.set_ylim(0,1.0)
	match_found_value_ax.legend()	

	# ground truth H, S, V
	ground_truth_hue_ax.bar(np.arange(len(groundTruthHSVSepreateHists[0])),groundTruthHSVSepreateHists[0], width = 1.0,color = 'r', label = groundTruthLabel+"_Hue")
	ground_truth_hue_ax.set_xlim(0, len(groundTruthHSVSepreateHists[0]))
	ground_truth_hue_ax.set_ylim(0,1.0)
	ground_truth_hue_ax.legend()	
	
	ground_truth_saturation_ax.bar(np.arange(len(groundTruthHSVSepreateHists[1])),groundTruthHSVSepreateHists[1], width = 1.0,color = 'g', label = groundTruthLabel+"_Saturation")
	ground_truth_saturation_ax.set_xlim(0, len(groundTruthHSVSepreateHists[1]))
	ground_truth_saturation_ax.set_ylim(0,1.0)
	ground_truth_saturation_ax.legend()	

	ground_truth_value_ax.bar(np.arange(len(groundTruthHSVSepreateHists[2])),groundTruthHSVSepreateHists[2], width = 1.0,color = 'b', label = groundTruthLabel+"_Value")
	ground_truth_value_ax.set_xlim(0, len(groundTruthHSVSepreateHists[2]))
	ground_truth_value_ax.set_ylim(0,1.0)
	ground_truth_value_ax.legend()	

	if(saveHist):
		plt.savefig(path+"/"+fname+".png")
	if(displayHist):
		plt.show()
	plt.clf()
	return

def plotUniquenessDistribution(path, fname, patches, metric, normalize_approach = "", displayHist = False, saveHist = True):
	if(metric == "HSV"):
		distribution = [patch.HSVScore for patch in patches]
	elif(metric == "HOG"):
		distribution = [patch.HOGScore for patch in patches]
	if(normalize_approach != ""):
		distribution = normalize(distribution, norm=normalize_approach)[0]
	print "in plotUniquenessDistribution, distribution:", distribution
	plt.plot(distribution, label = fname)

	if(saveHist):
		plt.savefig(path+"/"+fname+".png")
	if(displayHist):
		plt.show()
	plt.clf()

def plotResponseDistribution(path, this_feature_set, testPatchIndex, test_patch_response, random_patches_response, displayHist = False, saveHist = True):
	# file path
	if(not os.path.isdir(path)):
		os.makedirs(path)
	# file name
	fname = "LDA_Distribution_testPatch[{i}]".format(i = testPatchIndex)
	for i in range(0, len(this_feature_set)):
		fname += "_{feature}".format(feature = this_feature_set[i])

	# print random_patches_response
	# print [test_patch_response]
	bin = 50.0
	# plot histogram for test_patch_response and random_patches_response
	# plt.hist(random_patches_response, bin, normed=1, facecolor='green', alpha=0.75)
	plt.hist(random_patches_response, bin, facecolor='green', alpha=0.75)
	# plt.hist([test_patch_response], 1, normed=1, facecolor='red', alpha=0.75)
	plt.bar([test_patch_response],[1], width = (np.max(random_patches_response) - np.min(random_patches_response))/bin, color = 'r')

	if(saveHist):
		plt.savefig(path+"/"+fname+".png")
	if(displayHist):
		plt.show()
	plt.clf()

	