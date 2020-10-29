#!/bin/usr/env python3

### STAT 27700 PSET 2 ###

import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt

def solver(X, y):
	X_T = np.transpose(X)
	X_TX = np.matmul(X_T, X)
	inverse = np.linalg.inv(X_TX)
	pseudo = np.matmul(inverse, X_T)
	w = np.dot(pseudo, y)
	return w

def compute_error_average(face_X, face_y):
	size = 16
	errors = []
	for i in range(1, 9):
		start = size*(i-1)
		end = size*i
		X = np.delete(face_X, np.s_[start:end], 0)
		y = np.delete(face_y, np.s_[start:end], 0)
		w = solver(X, y)
		yhat = predictor(face_X[start:end], w)
		errs = error(face_y[start:end], yhat)
		errors.append(errs)
	return sum(errors)/len(errors)

def predictor(x, w):
	y_tildes = np.dot(x, w)
	y = np.zeros((16, 1))
	for i in range(0, 16):
		if y_tildes[i][0] > 0:
			y[i][0] = 1
		else:
			y[i][0] = -1
	return y

def error(y, yhat):
	errs = 0
	diff = yhat - y
	for i in range(0, 16):
		if diff[i][0] != 0:
			errs += 1
	return errs

def graph(face_X, face_y, w):
	x_0 = face_X[:, 0]
	x_1 = face_X[:, 1]
	fig = plt.figure()
	plt.scatter(x_0, x_1, c=face_y)
	x = x_1*(-w[1][0]/(w[0][0]))
	plt.plot(x, x_1)
	plt.title('Question 3 graph: 2 features')
	fig.savefig('q3graph.pdf')

if __name__=='__main__':
	face_emotion_data = loadmat('face_emotion_data.mat')
	face_y = face_emotion_data['y']
	face_X = face_emotion_data['X']
	face_w = solver(face_X, face_y)
	errs_avg_all_features = compute_error_average(face_X, face_y)
	face_X_small = np.zeros((128, 3))
	keep = [0, 2, 3]
	for col in keep:
		face_X_small[:, keep.index(col)] = face_X[:, col]
	errs_avg_3_features = compute_error_average(face_X_small, face_y)
	face_X_smaller = np.zeros((128, 2))
	keep = [0, 3]
	for col in keep:
		face_X_smaller[:, keep.index(col)] = face_X[:, col]
	w_small = solver(face_X_smaller, face_y)
	graph(face_X_smaller, face_y, w_small)

