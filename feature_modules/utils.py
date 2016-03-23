import numpy as np
import math
import os
import pickle

MIN_RAW_EUCLIDEAN_SCORE = 1/(1+ math.sqrt(2.0))
MAX_RAW_EUCLIDEAN_SCORE = 1.0

FEATURES_GENERATED_FOLDER = "features_generated"

BOTTOM_RIGHT_GREEN_FEATURE_ID = "BOTTOM_RIGHT_GREEN"
BOTTOM_RIGHT_NEIGHBOUR_BLUE_FEATURE_ID = "BOTTOM_RIGHT_NEIGHBOUR_BLUE"
BOTTOM_RIGHT_YELLOW_FEATURE_ID = "BOTTOM_RIGHT_YELLOW"
DONUT_SHAPE_FEATURE_ID = "DONUT_SHAPE"
TOP_LEFT_PURPLE_FEATURE_ID = "TOP_LEFT_PURPLE"
TOP_RIGHT_YELLOW_FEATURE_ID = "TOP_RIGHT_YELLOW"
CORNERNESS_FEATURE_ID = "CORNERNESS"
SHARP_HOG_FEATURE_ID = "SHARP_HOG"
BORDER_GREEN_FEATURE_ID = "BORDER_GREEN"
CENTRE_YELLOW_FEATURE_ID = "CENTRE_YELLOW"
CENTRE_BLUE_FEATURE_ID = "CENTRE_BLUE"
GREEN_PATCH_BOTTOM_LEFT_BLUE_FEATURE_ID = "GREEN_PATCH_BOTTOM_LEFT_BLUE"

CENTRE_PARADIGM_FEATURE_PREFIX = "centre_paradigm_"
SUBSQUARE_PARADIGM_FEATURE_PREFIX = "subsquare_paradigm_"
# GENERATED_FEATURE_IDS = ["centre_paradigm_0", "centre_paradigm_1", "centre_paradigm_2", "subsquare_paradigm_0"]
GENERATED_FEATURE_IDS = []

GENERATED_FEATURE_PARADIGMS = []

ALL_FEATURE_IDS = []

def converScaleTo01(value, min, max):
	"""
	conver the value in min, max range scale to [0,1] range
	"""
	return float(value - min) / float(max - min)


def loadGeneratedFeatureIds():
	global GENERATED_FEATURE_IDS
	GENERATED_FEATURE_IDS = []
	for feature_obj_pkl in os.listdir(utils.FEATURES_GENERATED_FOLDER):
		GENERATED_FEATURE_IDS.append(feature_obj_pkl[:feature_obj_pkl.find(".")])

def loadGeneratedFeatureParadigm():
	"""load in the auto generated feature paradigms"""
	global GENERATED_FEATURE_PARADIGMS
	global GENERATED_FEATURE_IDS
	GENERATED_FEATURE_PARADIGMS = []
	GENERATED_FEATURE_IDS = []
	for feature_obj_pkl in os.listdir(FEATURES_GENERATED_FOLDER):
		if (feature_obj_pkl.find(".pkl") != -1 ):
			with open(FEATURES_GENERATED_FOLDER + "/" + feature_obj_pkl, 'rb') as input:
				this_generated_feature = pickle.load(input)
				GENERATED_FEATURE_PARADIGMS.append(this_generated_feature)
				GENERATED_FEATURE_IDS.append(this_generated_feature.id)
	
	print "len(GENERATED_FEATURE_PARADIGMS):", len(GENERATED_FEATURE_PARADIGMS)
	print "GENERATED_FEATURE_IDS:", GENERATED_FEATURE_IDS

	"""put all feature id into a big array"""
	loadAllFeatureIds()

def loadAllFeatureIds():
	global ALL_FEATURE_IDS
	# auto generated feature from paradigms
	for feature_id in GENERATED_FEATURE_IDS:
		ALL_FEATURE_IDS.append(feature_id)
	# customized feature
	ALL_FEATURE_IDS.append(BOTTOM_RIGHT_GREEN_FEATURE_ID)
	ALL_FEATURE_IDS.append(BOTTOM_RIGHT_NEIGHBOUR_BLUE_FEATURE_ID)
	ALL_FEATURE_IDS.append(BOTTOM_RIGHT_YELLOW_FEATURE_ID)
	ALL_FEATURE_IDS.append(DONUT_SHAPE_FEATURE_ID)
	ALL_FEATURE_IDS.append(TOP_LEFT_PURPLE_FEATURE_ID)
	ALL_FEATURE_IDS.append(TOP_RIGHT_YELLOW_FEATURE_ID)
	ALL_FEATURE_IDS.append(CORNERNESS_FEATURE_ID)
	ALL_FEATURE_IDS.append(SHARP_HOG_FEATURE_ID)
	ALL_FEATURE_IDS.append(BORDER_GREEN_FEATURE_ID)
	ALL_FEATURE_IDS.append(CENTRE_YELLOW_FEATURE_ID)
	ALL_FEATURE_IDS.append(CENTRE_BLUE_FEATURE_ID)
	ALL_FEATURE_IDS.append(GREEN_PATCH_BOTTOM_LEFT_BLUE_FEATURE_ID)
	print "ALL_FEATURE_IDS:", ALL_FEATURE_IDS

