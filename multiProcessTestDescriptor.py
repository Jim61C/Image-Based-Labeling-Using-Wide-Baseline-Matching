from multiprocessing import Pool, cpu_count
import matchPatches
import comparePatches
import os
import time
from feature_modules import utils

def dispatch_match_test(testName):
	start_time = time.time()
	print 'start matching for ', testName ,": ",  start_time
	# folder_suffix = "_HOG_subCircle_Jensen_Shannon_Divergence"
	# folder_suffix = "_HOG_16Bin_subAndSuperCircle_Jensen_Shannon_Divergence"
	folder_suffix = "_seperateHS_Jensen_Shannon_Divergence_pyramid"
	# folder_suffix = "_HOG_Ori_Assignment_Jensen_Shannon_Divergence"
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

def dispatch_full_algorithm(args):
	test_folder_name, image_db = args
	# folder_suffix = "_DistinguishablePatches_HSAndCorner_Descriptor_seperateHS_Jensen_Shannon_Divergence"
	# folder_suffix = "_UniqueAlgo2_Force_HSV_Jensen_Shannon_Divergence"
	folder_suffix = "_full_algo_top20_unique_patches_response_based_no_neighbourblue"
	if (test_folder_name == "testset7"):
		matchPatches.findDistinguishablePatchesAndExecuteMatching(image_db, test_folder_name, "test1.jpg", "test3.jpg", folder_suffix, upperPath = "testAlgo3")	
	else:
		matchPatches.findDistinguishablePatchesAndExecuteMatching(image_db, test_folder_name, "test1.jpg", "test2.jpg", folder_suffix, upperPath = "testAlgo3")

def dispatch_feature_detection(args):
	test_folder_name, image_db = args
	# folder_suffix = "_DistinguishablePatches_HS_0.3_Corner_0.4_HOG_0.3_Descriptor_seperateHS_Jensen_Shannon_Divergence"
	# folder_suffix = "_DistinguishablePatches_HS_0.5_Corner_0.5_Descriptor_seperateHS_Jensen_Shannon_Divergence"
	# folder_suffix = "_DistinguishablePatches_HS_0.7_Corner_0.3_Descriptor_seperateHS_Jensen_Shannon_Divergence"
	# folder_suffix = "_DistinguishablePatches_HS_1.0_Descriptor_seperateHS_Jensen_Shannon_Divergence"
	# folder_suffix = "_DistinguishablePatches_Corner_1.0_Descriptor_seperateHS_Jensen_Shannon_Divergence"
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

def extract_all_testfoldernames(image_db):
	folders = [(name, image_db) for name in os.listdir(image_db) \
	if os.path.isdir(os.path.join(image_db, name))]
	return folders


def main():
	"""Necessary step for dispatch_feature_detection_algo3"""
	utils.loadGeneratedFeatureParadigm()
	image_db = "images"

	# test_folder_args = extract_all_testfoldernames(image_db)
	# testNames = ["populate_testset_illuminance1", "populate_testset_illuminance2", "populate_testset_rotation1","populate_testset_rotation2","populate_testset4"]
	# testNames = ["populate_testset_rotation1","populate_testset_rotation2", "populate_testset7"]
	# test_folder_names = ["testset_illuminance1", "testset_illuminance2", "testset_rotation1","testset_rotation2","testset4","testset7"]
	# test_folder_names = ["testset2", "testset3", "testset5"]
	
	# test_folder_args = [\
	# ("testset_illuminance1", image_db), \
	# ("testset_illuminance2", image_db), \
	# ("testset_rotation1", image_db), \
	# ("testset_rotation2", image_db), \
	# ("testset4", image_db), \
	# ("testset7", image_db), \
	# ("testset1", image_db), \
	# ("testset8", image_db), \
	# ]
	
	# test_folder_args = [ \
	# 	("testset_illuminance1", [utils.BOTTOM_RIGHT_GREEN_FEATURE_ID]), \
	# 	("testset_illuminance1", [utils.TOP_LEFT_PURPLE_FEATURE_ID]), \
	# 	("testset_illuminance1", [utils.BOTTOM_RIGHT_YELLOW_FEATURE_ID]), \
	# 	("testset_illuminance1", [utils.BOTTOM_RIGHT_NEIGHBOUR_BLUE_FEATURE_ID]), \
	# 	("testset_illuminance1", [utils.DONUT_SHAPE_FEATURE_ID]), \
	# 	("testset_illuminance1", [utils.DONUT_SHAPE_FEATURE_ID, utils.BOTTOM_RIGHT_NEIGHBOUR_BLUE_FEATURE_ID]) \
	# ]


	test_folder_args = [ \
		("testset7", [utils.GENERATED_FEATURE_IDS[0]], "test1.jpg"), \
		("testset7", [utils.GENERATED_FEATURE_IDS[0]], "test3.jpg"), \
		# ("testset7", [utils.GENERATED_FEATURE_IDS[1]], "test1.jpg"), \
		# ("testset7", [utils.GENERATED_FEATURE_IDS[1]], "test3.jpg"), \
		# ("testset7", [utils.GENERATED_FEATURE_IDS[2]], "test1.jpg"), \
		# ("testset7", [utils.GENERATED_FEATURE_IDS[2]], "test3.jpg"), \
		# ("testset7", [utils.GENERATED_FEATURE_IDS[3]], "test1.jpg"), \
		# ("testset7", [utils.GENERATED_FEATURE_IDS[3]], "test3.jpg") \
	]

	start_time = time.time()
	pool = Pool(cpu_count())
	# pool.map(dispatch_match_test, testNames)
	# pool.map(dispatch_feature_detection, test_folder_args)
	# pool.map(dispatch_feature_matching, test_folder_args)
	# pool.map(dispatch_full_algorithm, test_folder_args)
	pool.map(dispatch_feature_detection_algo3, test_folder_args)

	pool.close()
	pool.join()

	print "End Of Multiprocessing; time spent: ", time.time() - start_time
	return


if __name__ =="__main__":
	main()