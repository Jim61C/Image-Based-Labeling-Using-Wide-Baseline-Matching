# Unique Features For Image Labeling:
A wide base line algorithm based on the concept of detecting unique features of the scene
Application in an image labeling system

# Usage:
Running the algorithm
- clickGroundTruth.py - Automatic Unique Feature Construction & Ground Truth Click; Constructed unique features will be saved in ./features_generated
- comparePatches.py - Unique Feature Detection
- matchPatches.py - Unique Feature Matching & full algorithm starter
- multiProcessTestDescriptor.py - Driver program for muti-process full algorithms

Image Labeling Database Folder Structure Required:
./images
├── testset_flower1
│   ├── test1.jpg
│   ├── test2.jpg
│   └── test3.jpg
...

- test1.jpg and test3.jpg are the super wide baseline image sets used for construct the unique feature patches database for the scene
- test2.jpg is the testing image of the scene
- Results will be placed in ./{upperPath}/GaussianWindowOnAWhole/{testFolderName}_{folderSuffix}/
- {upperPath} is the folder contained in the root folder,
- {testFolderName} is the testset name, for example, "testset_flower1"
- {folderSuffix} is the user specified suffix for marking purpose appended at the end of the folder storing the results.
- These three string parameters will be asked for input in the starter in matchPatches.py
