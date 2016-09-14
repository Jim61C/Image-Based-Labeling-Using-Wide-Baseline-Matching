# Unique Features For Image Labeling:
- Author: Xing Yifan <xingyifan@u.nus.edu>
- A wide base line algorithm based on the concept of detecting unique features of the scene 
- Application in an image labeling system

# Usage:

- Setup Module Requirements:
  - Python
  - Numpy
  
    ```
    pip install numpy
    ```
  - Scipy
  
    ```
    pip install scipy
    ```
  - [Scikit Learn](http://scikit-learn.org/stable/install.html)
  
    ```
    pip install -U scikit-learn    
    ```
  - matplotlib
  
    ```
    pip install matplotlib
    ```
  - earth mover distance package
  
    ```
    pip install pyemd
    ```

- clickGroundTruth.py

  - Follow on screen insruction for Automatic Unique Feature Construction / Ground Truth Click
  - 1. click for mannual groundTruth labelling
  - 2. click for automatic feature construction
  - Constructed unique features will be saved in ./features_generated
  - click on the image for the patch to be processed/recorded
  - press 'u' for undo
  - press any other key for finish

- comparePatches.py
  - Unique Feature Detection

- matchPatches.py
  - Follow on screen menu for different option
  - 1. Unique Feature Matching & full algorithm starter
  - 2. Manual Pruning procedure for good image matches database contruction
  - 3. Check testing image number of matches against different images in the database for labeling purpose

- multiProcessTestDescriptor.py
  - Driver program for muti-process full algorithms

Image Labeling Database Folder Structure Required:

./{image_db}

└── {testFolderName}

          └── test1.jpg

          └── test2.jpg

          └── test3.jpg
...

- {image_db} is the folder located at the root level of the repository containing image folders of many scenes, for example, "./images" used in the current folder hierarchy
- {testFolderName} is the folder containing the three images of three different view points, for example, "testset_orchid1"
- test1.jpg and test3.jpg are the super wide baseline image sets used for construct the unique feature patches database for the scene
- test2.jpg is the testing image of the scene (with a relatively shorter baseline)
- Results will be placed in ./{upperPath}/GaussianWindowOnAWhole/{testFolderName}_{folderSuffix}/
  - {upperPath} is a user specified folder located at the root level of the repository, default is "testAlgo3" for database construction and "testLabeling" for testing image matching against database images
  - {testFolderName} is the testset name, for example, "testset_flower1", "testset_orchid1", etc
  - {folderSuffix} is the user specified suffix for marking purpose appended at the end of the folder storing the results.
  - These three string parameters will be asked for input in the starter in matchPatches.py
  - These three string parameters are predefined in multiProcessTestDescriptor.py in a predefined routine for orchids testsets
