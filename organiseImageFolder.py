import numpy as np
import math
import sys
import operator
import itertools
import os
from shutil import copyfile

def queryPath(path):
	"""
	checks if the given path exisits, if not existing, create and return it; else, just echo back it
	"""
	if(not os.path.isdir("./{path}".format(
		path = path))):
		os.makedirs("./{path}".format(
			path = path))
	return path

def main():
	images_db = "images"
	raw_folder = "orchid_raw"
	test_folder_name_prefix = "testset_orchid"
	test_img_name_prefix = "test"
	count = 0
	for file_name in os.listdir("{images_db}/{raw_folder}/".format(images_db = images_db, raw_folder = raw_folder)):
		if (file_name.find(".jpg") != -1):

			"""create test folder"""
			queryPath("{images_db}/{test_folder_name}/".format(\
				images_db = images_db, \
				test_folder_name = test_folder_name_prefix + str(int(count/4) + 1)))
			
			"""copy over images"""
			if (count % 4 == 0):
				# label image
				copyfile('{images_db}/{raw_folder}/{file_name}'.format(images_db = images_db, \
					raw_folder = raw_folder, \
					file_name = file_name), \
					"{images_db}/{test_folder_name}/{file_name}".format( images_db= images_db, \
						test_folder_name = test_folder_name_prefix + str(int(count/4) + 1), \
						file_name = "label.jpg" ))
			else:
				copyfile('{images_db}/{raw_folder}/{file_name}'.format(images_db = images_db, \
					raw_folder = raw_folder, \
					file_name = file_name), \
					"{images_db}/{test_folder_name}/{file_name}".format( images_db= images_db, \
						test_folder_name = test_folder_name_prefix + str(int(count/4) + 1), \
						file_name = test_img_name_prefix + str(int(count%4)) + ".jpg" ))

			count += 1

	return

if __name__ == "__main__":
	main()