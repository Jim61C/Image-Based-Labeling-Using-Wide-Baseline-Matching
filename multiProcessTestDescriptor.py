from multiprocessing import Pool, cpu_count
import matchPatches
import time

def dispatch_match_test(testName):
	start_time = time.time()
	print 'start matching for ', testName ,": ",  start_time
	# folder_suffix = "_HOG_subCircle_Jensen_Shannon_Divergence"
	# folder_suffix = "_HOG_16Bin_subAndSuperCircle_Jensen_Shannon_Divergence"
	# folder_suffix = "_seperateHS_Jensen_Shannon_Divergence"
	folder_suffix = "_HOG_36Bin_Ori_Jensen_Shannon_Divergence"
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

def dispatch_full_algorithm(test_folder_name):
	folder_suffix = "_DistinguishablePatches_HSAndCorner_Descriptor_seperateHS_Jensen_Shannon_Divergence"
	matchPatches.findDistinguishablePatchesAndExecuteMatching(test_folder_name, "test1.jpg", "test2.jpg", folder_suffix)

def dispatch_feature_detection(test_folder_name):
	# folder_suffix = "_DistinguishablePatches_HS_0.3_Corner_0.4_HOG_0.3_Descriptor_seperateHS_Jensen_Shannon_Divergence"
	# folder_suffix = "_DistinguishablePatches_HS_0.5_Corner_0.5_Descriptor_seperateHS_Jensen_Shannon_Divergence"
	# folder_suffix = "_DistinguishablePatches_HS_0.7_Corner_0.3_Descriptor_seperateHS_Jensen_Shannon_Divergence"
	# folder_suffix = "_DistinguishablePatches_HS_1.0_Descriptor_seperateHS_Jensen_Shannon_Divergence"
	# folder_suffix = "_DistinguishablePatches_Corner_1.0_Descriptor_seperateHS_Jensen_Shannon_Divergence"
	folder_suffix = "__DistinguishablePatches_HSFlattened_0.7_Corner_0.3_Descriptor_seperateHS_Jensen_Shannon_Divergence"
	matchPatches.findAndSaveDistinguishablePatches(test_folder_name, "test1.jpg", folder_suffix, sigma = 39, upperPath = "testMatches")

def dispatch_feature_matching(test_folder_name):
	folder_suffix = "_DistinguishablePatches_HS_0.3_Corner_0.4_HOG_0.3_Descriptor_seperateHS_Jensen_Shannon_Divergence"
	matchPatches.populateFeatureMatchingTest(test_folder_name, "test1.jpg", "test2.jpg",folder_suffix, upperPath = "testMatches")

def main():
	print "cpu_count():",cpu_count()
	# testNames = ["populate_testset_illuminance1", "populate_testset_illuminance2", "populate_testset_rotation1","populate_testset_rotation2","populate_testset4","populate_testset7"]
	testNames = ["populate_testset_rotation1","populate_testset_rotation2"]
	# test_folder_names = ["testset_illuminance1", "testset_illuminance2", "testset_rotation1","testset_rotation2","testset4","testset7"]
	# test_folder_names = ["testset2", "testset3", "testset5"]

	pool = Pool(cpu_count())
	pool.map(dispatch_match_test, testNames)
	# pool.map(dispatch_full_algorithm, test_folder_names)
	# pool.map(dispatch_feature_detection, test_folder_names)
	# listOfArguments = []
	# pool.map(matchPatches.populateNewTest, listOfArguments)
	pool.close()
	pool.join()
	# for partition in partition_list(data_files, 4):
		# res = pool.map(process_file_callable, partition)
		# print res
	return


if __name__ =="__main__":
	main()