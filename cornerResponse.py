import numpy as np
import cv2
from matplotlib import pyplot as plt

# get a guassian kernel of size * size
def gauss_kernels(size,sigma=1.0):
	if size<3:
		size = 3
	m = size/2
	x, y = np.mgrid[-m:m+1, -m:m+1]
	x = x.astype(float)
	y = y.astype(float)
	# print x*x + y*y
	# print (x*x + y*y)/(2*sigma*sigma)
	# print -(x*x + y*y)/(2*sigma*sigma)
	kernel = np.exp(-(x*x + y*y)/(2*sigma*sigma))
	# print kernel
	kernel_sum = kernel.sum()
	if not sum==0:
		kernel = kernel/kernel_sum 
	return kernel


# assume ConvMatrix is of size n*n
def convoleImg(img, ConvMatrix, _step = 1):
	response = np.zeros((img.shape[0], img.shape[1]))
	convsize = ConvMatrix.shape[0]
	step = convsize/2 * _step
	print "convolve step:", step
	for i in np.arange (0+convsize/2, response.shape[0]-convsize/2, step):
		for j in np.arange(0+convsize/2, response.shape[1]-convsize/2, step):
			imgi = i - convsize/2
			imgj = j - convsize/2
			pixelResponse = 0
			for k in range(0, convsize):
				for l in range(0, convsize):
					pixelResponse += img[imgi+k][imgj+l]*ConvMatrix[k][l]
			response[i][j] = pixelResponse
	return response


def sobelConvolution(img):
	verticalConv = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
	horizontalConv = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
	gy = convoleImg(img, verticalConv)
	gx = convoleImg(img, horizontalConv)
	return gx, gy

def getHarrisCornerResponse(img, windowSize, _step = 1):
	response = np.zeros(shape = (img.shape[0], img.shape[1]))
	gx, gy = sobelConvolution(img)
	Ixx = np.multiply(gx,gx)
	Ixy = np.multiply(gx,gy)
	Iyy = np.multiply(gy,gy)

	gaussianKernel = gauss_kernels(windowSize)
	print "gaussianKernel:", gaussianKernel
	Wxx = convoleImg(Ixx, gaussianKernel, _step)
	Wxy = convoleImg(Ixy, gaussianKernel, _step)
	Wyy = convoleImg(Iyy, gaussianKernel, _step)
	print "Wxx amax:", np.amax(Wxx)
	print "Wxy amax:", np.amax(Wxy)
	print "Wyy amax:", np.amax(Wyy)

	k = 0.06
	maxResponse = 0
	for i in np.arange(windowSize/2, response.shape[0] - windowSize/2, windowSize/2 * _step):
		for j in np.arange(windowSize/2, response.shape[1] - windowSize/2, windowSize/2 * _step):
			W = np.array([[Wxx[i][j], Wxy[i][j]], [Wxy[i][j], Wyy[i][j]]])
			detW = np.linalg.det(W)
			# print "detW:", detW, "< 0?:", detW < 0
			traceW = np.trace(W)
			# print "trace of W:", traceW
			response[i][j] = detW - k * traceW * traceW
			# print "response[{i}][{j}]".format(i = i,  j = j), response[i][j]
			if(response[i][j] > maxResponse):
				maxResponse = response[i][j] # update responseMax

	return maxResponse, response

def filter_patches(patches, thresh_pass, responseMatrix, maxResponse):
	thresh = thresh_pass * maxResponse
	fitlered_patches = []
	for i in range(0, len(patches)):
		if(responseMatrix[patches[i].x][patches[i].y] >= thresh):
			fitlered_patches.append(patches[i])
	return fitlered_patches


def main():
	# x1 = np.arange(9.0).reshape((3, 3))
	# print np.trace(x1)
	# x2 = np.array([[1,2,1],[3,0,1],[5,2,2]], dtype = np.float32)
	# print x2.dtype
	# print x1
	# print x2
	# print np.multiply(x1, x2)
	# print np.multiply(x1, x1)
	# print np.multiply(x2, x2)
	imgName = "test1.jpg"
	folderName = "testset4"
	# folderName = "testset_rotation1"
	filenames = ["images/{folder}/{name}".format(folder = folderName,  name = imgName)]
	windowSize = 39
	step = 1
	thresh_pass = 0.001
	for filenameIndex in range(0, len(filenames)):
		filename = filenames[filenameIndex]
		img = cv2.imread(filename,0)
		print "img.shape:", img.shape
		
		maxResponse, responseMatrix = getHarrisCornerResponse(img, windowSize, step)
		print "responseMatrix.shape:", responseMatrix.shape
		print "maxResponse:", maxResponse
		print "most negative Response:", np.amin(responseMatrix)
		thresh = thresh_pass * maxResponse
		
		imgColor = cv2.imread(filename, 1)
		for i in np.arange(windowSize/2, responseMatrix.shape[0] - windowSize/2, windowSize/2 * step):
			for j in np.arange(windowSize/2, responseMatrix.shape[1] - windowSize/2, windowSize/2 * step):
				if(responseMatrix[i][j] >= thresh):
					cv2.rectangle(imgColor,(j-windowSize/2,i-windowSize/2),(j+windowSize/2,i+windowSize/2),(0,0,255),2)
		# cv2.imwrite("corners/{folder}_{filename}_HarrisEdges_windowSize{windowSize}.jpg".format(windowSize = windowSize, folder =folderName,  filename = imgName[0:imgName.find(".")]), imgColor)
		cv2.imwrite("corners/{folder}_{filename}_HarrisCorners_windowSize{windowSize}_threshPass_{thresh_pass}.jpg".format(windowSize = windowSize, folder =folderName,  filename = imgName[0:imgName.find(".")], thresh_pass = thresh_pass), imgColor)
		cv2.imshow("corners", imgColor)
		cv2.waitKey(0)
	return 

if __name__ == "__main__":
	main()