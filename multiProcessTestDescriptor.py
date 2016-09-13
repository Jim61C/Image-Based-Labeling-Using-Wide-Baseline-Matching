from multiprocessing import Pool, cpu_count
import matchPatches
import comparePatches
import os
import time
from feature_modules import utils

def dispatch_match_test(testName):
	start_time = time.time()
	print 'start matching for ', testName ,": ",  start_time
	folder_suffix = "_seperateHS_Jensen_Shannon_Divergence_pyramid"
	print testName
	if(testName == "populate_testset_illuminance1"):
	    matchPatches.populate_testset_illuminance1(folder_suffix)
	elif(testName == "populate_testset_illuminance2"):
	    matchPatches.populate_testset_illuminance2(folder_suffix)
	elif(testName == "populate_testset_rotation1"):
	    matchPatches.populate_testset_rotation1(folder_suffix)
	elif(testName == "populate_testset_rotation2"):
	    matchPatches.populate_testset_rotation2(folder_suffix)
	elif(testName == "populate_testset4"):
	    matchPatches.populate_testset4(folder_suffix)
	elif(testName == "populate_testset7"):
	    matchPatches.populate_testset7(folder_suffix)
	print 'finish matching for ',testName,' ; time spent:', time.time() - start_time, " secs"

def driver_full_algorithm(args):
	if(len(args) == 5):
		dispatch_full_algorithm_from_two_folder(args)
	elif(len(args) == 2):
		dispatch_full_algorithm(args)
	return

def dispatch_full_algorithm_from_two_folder(args):
	test_folder_name1, test_folder_name2, image1, image2, image_db = args
	folder_suffix = "_integralImageHS_descriptor_based_point_01_Harris_from_two_folder_high_response_only_normalizedJS"
	matchPatches.findDistinguishablePatchesAndExecuteMatchingFromTwoFolders(\
		image_db, test_folder_name1, test_folder_name2, \
		image1, image2, \
		folder_suffix, upperPath = "testLabeling", initialize_features = False)

def dispatch_full_algorithm(args):
	test_folder_name, image_db = args
	folder_suffix = "_integralImageHS_top20_unique_patches_descriptor_based_point_01_Harris_high_response_only_normalizedJS"
	if (test_folder_name == "testset7"):
		matchPatches.findDistinguishablePatchesAndExecuteMatching(\
			image_db, test_folder_name, "test1.jpg", "test3.jpg", folder_suffix, upperPath = "testAlgo3", initialize_features = False)	
	else:
		matchPatches.findDistinguishablePatchesAndExecuteMatching(\
			image_db, test_folder_name, "test1.jpg", "test3.jpg", folder_suffix, upperPath = "testAlgo3", initialize_features = False)

def dispatch_matching_given_test_patches(args):
	test_folder_name, image_db = args
	folder_suffix = "_integralImageHS_top20_unique_patches_descriptor_based_point_01_Harris_high_response_only_normalizedJS"
	if (test_folder_name == "testset7"):
		matchPatches.executeMatchingGivenDinstinguishablePatches(\
			image_db, test_folder_name, "test1.jpg", "test3.jpg", folder_suffix, upperPath = "testAlgo3", initialize_features = False)	
	else:
		matchPatches.executeMatchingGivenDinstinguishablePatches(\
			image_db, test_folder_name, "test1.jpg", "test2.jpg", folder_suffix, upperPath = "testAlgo3", initialize_features = False)

def dispatch_matching_given_test_patches_test_from_two_folder(args):
	"""
	Given already detected patches from test image (test_folder_name1), 
	no need to go through feature detection phase again, straight to matching
	"""
	test_folder_name1, test_folder_name2, image1, image2, image_db = args
	folder_suffix = "_integralImageHS_descriptor_based_point_01_Harris_from_two_folder_high_response_only_normalizedJS"
	matchPatches.executeMatchingGivenDinstinguishablePatchesFromTwoFolders(image_db, test_folder_name1, test_folder_name2, \
	image1, image2, folder_suffix, upperPath = "testLabeling", initialize_features = False)

def dispatch_feature_detection(args):
	test_folder_name, image_db = args
	folder_suffix = "__DistinguishablePatches_HSFlattened_0.7_Corner_0.3_Descriptor_seperateHS_Jensen_Shannon_Divergence"
	matchPatches.findAndSaveDistinguishablePatches(image_db, test_folder_name, "test1.jpg", folder_suffix, sigma = 39, upperPath = "testMatches")

def dispatch_feature_matching(args):
	test_folder_name, image_db = args
	folder_suffix = "_DistinguishablePatches_HS_0.3_Corner_0.4_HOG_0.3_Descriptor_seperateHS_Jensen_Shannon_Divergence"
	matchPatches.populateFeatureMatchingStatistics(image_db, test_folder_name, "test1.jpg", "test2.jpg",folder_suffix, upperPath = "testMatches")
	
def dispatch_feature_detection_algo3(args):
	test_folder_name, custom_features_set, img_name = args
	comparePatches.populateTestFindDistinguishablePatchesAlgo3(\
		test_folder_name = test_folder_name, \
		img_name = img_name, \
		sigma = 39, \
		image_db = "images", \
		custom_feature_sets = custom_features_set)

def dispatch_test_labeling_num_matches(args):
	plot_folder_name, tight_criteria = args
	matchPatches.populateCheckTestLabelingNumMatches(plot_folder_name, tight_criteria, save = False, show = True)

def extract_all_testfoldernames(image_db):
	folders = [(name, image_db) for name in os.listdir(image_db) \
	if os.path.isdir(os.path.join(image_db, name))]
	return folders


def main():
	"""Necessary step for dispatch_feature_detection_algo3"""
	utils.loadGeneratedFeatureParadigm()
	image_db = "images"

	"""driver_full_algorithm"""
	test_folder_args = []
	num_orchid_tests = 20
	for i in range (1, num_orchid_tests + 1):
		"""For full algo from single folder"""
		test_folder_args.append(("testset_orchid{i}".format(i = i), image_db))


	pool = Pool(cpu_count())
	pool.map(driver_full_algorithm, test_folder_args)
	pool.close()
	pool.join()

	"""
	driver_full_algorithm from two folder: 
	testing image with viewpoint test2.jpg to compare with test3.jpg in the database(of same scene)
	"""
	test_folder_args = []
	num_orchid_tests = 20
	for i in range (1, num_orchid_tests + 1):
		"""For full algo from single folder"""
		# test_folder_args.append(("testset_orchid{i}".format(i = i), image_db))
		"""For full algo from two folder"""
		test_folder_args.append(("testset_orchid{i}".format(i = i), "testset_orchid{i}".format(i = i), \
			"test2.jpg", "test3.jpg", image_db))


	pool = Pool(cpu_count())
	pool.map(driver_full_algorithm, test_folder_args)
	pool.close()
	pool.join()

	"""
	dispatch_matching_given_test_patches_test_from_two_folder:
	testing image with viewpoint test2.jpg to compare with test3.jpg in the database(of different scene)
	"""
	test_folder_args = []
	num_orchid_tests = 20
	for i in range (1, num_orchid_tests + 1):
		for j in range(1, num_orchid_tests + 1):
			if (i != j):
				test_folder_args.append(("testset_orchid{i}".format(i = i), "testset_orchid{j}".format(j = j), \
					"test2.jpg", "test3.jpg", image_db))

	pool = Pool(cpu_count())
	pool.map(dispatch_matching_given_test_patches_test_from_two_folder, test_folder_args)
	pool.close()
	pool.join()

	return


if __name__ =="__main__":
	main()