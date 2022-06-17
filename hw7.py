"""
Author:   Nicholas Lutrzykowski
Course:   CSCI 6270
Homework:  4
Problem: 1
File:    p1_svm.py

Purpose: A script that takes descriptors and applies svm

Questions: 
	- What does the confusion matrix mean? It is giving me confusion... 
	- Is the offset bias the same thing as the svc.intercept_ attribute? 
	- How exactly does the validation work? 
		Train over and over using the same data but with different training param? 
		Is the only parameter that changes c? 

"""

import cv2
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
from scipy.linalg import null_space
import random
from sklearn.cluster import AgglomerativeClustering
import copy

import os
import math
import random
import sys
import copy
import glob
import math

def read_images(img1_name, img2_name):
	in_dir = os.getcwd()
	new_dir = os.path.join(in_dir, 'hw7data')
	img1_path, img2_path = os.path.join(new_dir, img1_name), os.path.join(new_dir, img2_name)
	img1, img2 = cv2.imread(img1_path), cv2.imread(img2_path)
	img1_g, img2_g = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE), cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

	return img1, img2, img1_g, img2_g

def get_sift_keypoints(img1, img2):
	sift_alg = cv2.SIFT_create() 
	kp1, des1  = sift_alg.detectAndCompute(img1.astype(np.uint8),None)
	kp2, des2  = sift_alg.detectAndCompute(img2.astype(np.uint8),None)

	num_keypoints1, num_keypoints2 = len(kp1), len(kp2)

	bf = cv2.BFMatcher()
	matches = bf.knnMatch(des1, des2, k=2)

	# Added ratio test 
	good_matches = [] 
	good_matches2 = []
	list_kp1 = [] 
	list_kp2 = [] 
	for m, n in matches: 
		if m.distance < 0.8*n.distance:
			good_matches.append([m])
			good_matches2.append(m)
			list_kp1.append(kp1[m.queryIdx].pt)
			list_kp2.append(kp2[m.trainIdx].pt)

	list_kp1 = np.array(list_kp1)
	list_kp2 = np.array(list_kp2)

	img = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good_matches, None, flags=2)
	cv2.imwrite("output.png", img)

	return list_kp1, list_kp2

def get_sift(img):
	sift_alg = cv2.SIFT_create()
	kp1, des1  = sift_alg.detectAndCompute(img.astype(np.uint8),None)

	points = [] 
	for p in kp1: 
		points.append(p.pt)

	points = np.array(points).astype(np.float32)
	#print("POINTS")
	#print(points.shape)
	#print(points)

	return points

def get_features(img1, img2, img1_g, img2_g):
	feature_params = dict( maxCorners = 100, qualityLevel = 0.5, minDistance = 7, blockSize = 7 )

	lk_params = dict( winSize = (30, 30), maxLevel = 10, criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 0.03))

	
	p0 = cv2.goodFeaturesToTrack(img1_g, mask = None, **feature_params)

	p0 = get_sift(img1_g.astype(np.uint8))

	# Create a mask image for drawing purposes
	mask = np.zeros_like(img1)

	# Compare to other image 
	p1, st, err = cv2.calcOpticalFlowPyrLK(img1_g, img2_g, p0, None, **lk_params)
	#p1, st, err = cv2.calcOpticalFlowPyrLK(img1_g, img2_g, p0, None)

	# Select valid points 
	p1_temp = p1.reshape(p1.shape[0], 1, 2)
	p0_temp = p0.reshape(p0.shape[0], 1, 2)
	
	good_new = p1_temp[st == 1]
	good_old = p0_temp[st == 1]

	color = np.random.randint(0, 255, (len(good_new), 3))
	
	# draw the tracks
	count = 1
	lines = []
	keypoints = [] 
	vectors = [] 
	for i, (new, old) in enumerate(zip(good_new, good_old)):
		a, b = int(new[0]), int(new[1])
		c, d = int(old[0]), int(old[1])
		
		
		'''
		vect = np.array([a - c, b - d])
		u = -vect[1] / math.sqrt(vect[0]**2+vect[1]**2)
		v = vect[0] / math.sqrt(vect[0]**2+vect[1]**2)
		a_, b_, c_ = u, v, u*a+v*b
		'''
		vect = np.array([a - c, b - d])

		if(math.sqrt(vect[0]**2+vect[1]**2) < 1e-15): continue

		a_ = vect[0] / math.sqrt(vect[0]**2+vect[1]**2)
		b_ = vect[1] / math.sqrt(vect[0]**2+vect[1]**2)
		c_ = -math.sqrt(vect[0]**2+vect[1]**2)
		
		c_ = -((a_*a)+(b_*b))

		lines.append([a_,b_,c_])
		
		if(math.sqrt(vect[0]**2+vect[1]**2) > 10):
			keypoints.append([a, b, c, d])
			vectors.append(vect)
		'''
		print("EQUATION TEST")
		print(a_*a+b_*b-c_)
		'''
		mask = cv2.arrowedLine(mask, (c, d), (a, b), color[i].tolist(), 2)
		img1 = cv2.circle(img1, (a, b), 2, color[i].tolist(), -1)

		img = cv2.add(img1, mask)

		count += 1

	#cv2.imwrite("output.png", img)

	return np.array(lines), np.array(keypoints), np.array(vectors), img

def ransac(lines, img, r = 6):
	num_iter = int(lines.shape[0]**2 * 0.05)

	xfoe, yfoe, k = 0, 0, 0

	for sample in range(num_iter):
		i, j = random.sample(range(lines.shape[0]), 2)

		if i == j: continue 
		M = np.array([lines[i], lines[j]])
		ns = null_space(M)
		
		if (ns.shape[1] > 1):
			ns = ns[:, 0]

		if (ns[2] < 1e-15): continue

		x = ns[0]/ns[2]
		y = ns[1]/ns[2]
		
		dist = lines[:,0]*x + lines[:,1]*y + lines[:,2]

		res = np.sum(dist**2 < r**2)
		
		if res > k:
			k = res 
			xfoe = int(x)	
			yfoe = int(y)

	moving = False
	if k < 70:
		print("There are not enough motion lines within the distance tolerance")
		print("The camera is not moving")
		cv2.imwrite("output1.png", img)
	else: 
		print("The camera is moving")
		if(xfoe < 0 or xfoe > img.shape[1] or yfoe < 0 or yfoe > img.shape[0]): 
			img = edit_image(img, xfoe, yfoe)
			
		else:
			img = cv2.circle(img, (xfoe, yfoe), 10, [255,0,0], -1)
		
		cv2.imwrite("output1.png", img)
		moving = True

	return xfoe, yfoe, moving

def edit_image(img, xfoe, yfoe):
	x_size, y_size = img.shape[1], img.shape[0]
	img_x, img_y = 0, 0
	if(xfoe < 0): 
		x_size = img.shape[1]-xfoe
		img_x = -xfoe
	if(yfoe < 0): 
		y_size = img.shape[0]-yfoe
		img_y = -yfoe
	if(xfoe > img.shape[1]): x_size = xfoe
	if(yfoe > img.shape[0]): y_size = yfoe
	
	img_new = np.zeros((y_size+20, x_size+20, 3), dtype = np.int8)
	
	img_new[img_y:img_y+img.shape[0], img_x:img_x+img.shape[1], :] = img

	img_new = cv2.circle(img_new, (xfoe, yfoe), 10, [255,0,0], -1)

	return img_new

def clustering(vectors, kp, img1, moving): 
	# Start with n clusters (number of points)
	# Create nxn matrix 
	# Create list of clusters 
	# End condition: When min distance between any 2 clusters > threshold 
	# Calculating the distance value: 
	points = copy.copy(kp[:, :2])
	points[:, 0] = points[:, 0] / img1.shape[1]
	points[:, 1] = points[:, 1] / img1.shape[0]

	desc = np.concatenate((vectors, points), axis=1)

	clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=5).fit(desc)
	color = np.random.randint(0, 255, (clustering.n_clusters_, 3))

	top_left = np.zeros((clustering.n_clusters_, 2), dtype = int)
	bot_right = np.zeros((clustering.n_clusters_, 2), dtype = int)
	top_left[:,0] = img1.shape[1]
	top_left[:,1] = img1.shape[0]


	counts = np.bincount(clustering.labels_)
	foe = np.argmax(counts)

	'''
	for i in range(clustering.n_clusters_):
		print(np.count_nonzero(clustering.labels_ == i))
	'''
	mask = np.zeros_like(img1)
	for i in range(kp.shape[0]):
		if np.count_nonzero(clustering.labels_ == clustering.labels_[i]) < 10: continue 
		if foe == clustering.labels_[i] and moving: continue

		a, b, c, d = kp[i]
		mask = cv2.arrowedLine(mask, (c, d), (a, b), color[clustering.labels_[i]].tolist(), 2)
		img1 = cv2.circle(img1, (a, b), 2, color[clustering.labels_[i]].tolist(), -1)
		img = cv2.add(img1, mask)
		cluster = clustering.labels_[i]

		if (top_left[cluster, 0] > a): top_left[cluster, 0] = a 
		if (top_left[cluster, 1] > b): top_left[cluster, 1] = b
		if (top_left[cluster, 0] > c): top_left[cluster, 0] = c 
		if (top_left[cluster, 1] > d): top_left[cluster, 1] = d
		if (bot_right[cluster, 0] < c): bot_right[cluster, 0] = c
		if (bot_right[cluster, 1] < d): bot_right[cluster, 1] = d

	count = 0
	for i in range(clustering.n_clusters_):
		if top_left[i,0] == 0 and bot_right[i, 0] == 0 and top_left[i,1] == 0 and bot_right[i, 1] == 0:
			continue
		if top_left[i,0] == 0: top_left[i,0] = 20
		if top_left[i,1] == 0: top_left[i,1] = 20
		if bot_right[i,0] == 0: bot_right[i,0] = img.shape[0]-20
		if bot_right[i,1] == 0: bot_right[i,1] = img.shape[1]-20

		img = cv2.rectangle(img, top_left[i], bot_right[i], color[i].tolist(), 2)
		count += 1

	if count > 0:
		cv2.imwrite("output2.png", img)



if __name__ == "__main__":

	# Command Line Arguments (python hw7.py img1, img2)
	if len(sys.argv) != 3: 
		print("Enter in format: <img1> <img2>")
		sys.exit()

	img1_name, img2_name = sys.argv[1], sys.argv[2]
	img1, img2, img1_g, img2_g = read_images(img1_name, img2_name)

	#kp1, kp2 = get_sift_keypoints(img1_g, img2_g)
	lines, kp, vectors, img = get_features(img1, img2, img1_g.astype(np.uint8), img2_g.astype(np.uint8))

	xfoe, yfoe, moving = ransac(lines, img)

	clustering(vectors, kp, img1, moving)







