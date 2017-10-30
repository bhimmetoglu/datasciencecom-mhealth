# Author: Burak Himmetoglu (burakhmmtgl@gmail.com)
# Date  : 10-25-2017
# 
# -- Utilities for mHealth dataset -- #

import numpy as np
import pandas as pd
import os
import requests
import zipfile
from sklearn.model_selection import train_test_split

## Download and extract data files
def download_and_extract():
	url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00319/MHEALTHDATASET.zip"
	print("Downloading..")
	r = requests.get(url)

	# Write into file
	open('MHEALTHDATASET.zip','wb').write(r.content)

	# Extract
	print('Extracting...')
	zip_h = zipfile.ZipFile('MHEALTHDATASET.zip','r')
	zip_h.extractall()
	zip_h.close()

	# Rename and remove zip
	os.rename('MHEALTHDATASET', 'data')
	os.remove('MHEALTHDATASET.zip')

## Read data per subject
def read_subject(subject):
	""" Read measurements from a given subject """
	file_name = 'mHealth_subject' + str(subject) + '.log'
	file_path = os.path.join('data',file_name)


	# Read file
	try:
		df = pd.read_csv(file_path, delim_whitespace = True, header = None)
	except IOError:
		print("Data file does not exist!")

	# Remove data with null class (=0)
	df = df[df[23] != 0]

	return df

## Rewrite a sequence for a given subject in tensor format
def split_by_blocks(df, block_size=100):
	""" Split data from each subject into blocks of shorter length """

	# Channels
	n_channels = df.shape[1]-1

	# Group by labels 
	grps = df.groupby(23)
	
	# Create a list for concatenating
	X_ = []
	Y_ = []

	# Loop over groups (labels), reshape to tensor and concatenate
	for ig in range(1,len(grps)+1,1):
		df_ = grps.get_group(ig)

		# Data and targets
		y = pd.unique(df_[23].values)
		x = df_.drop(23, axis=1).as_matrix()
		
		n_blocks = len(x) // block_size
		x = x[:n_blocks*block_size]
		y = y[:n_blocks*block_size]

		x_tensor = x.reshape(-1, block_size, n_channels)

		# Append
		X_.append(x_tensor)
		Y_.append(np.array([y]*len(x_tensor), dtype=int).squeeze())

	# Concatenate and return
	X = np.concatenate(X_, axis=0)
	Y = np.concatenate(Y_, axis=0)

	return X, Y

## Merge all the subjects and save into file
def collect_save_data(subject_count = 10, block_size=100):
	""" Collects all the data from all the subjects and writes in file """

	# Initiate lists
	X_ = []
	Y_ = []
	for s in range(1,subject_count+1):

		# Read the data
		df = read_subject(s)

		# Split into blocks
		x,y = split_by_blocks(df, block_size)

		# Add to list
		X_.append(x)
		Y_.append(y)

	# Concatenate and save
	X = np.concatenate(X_, axis=0)
	Y = np.concatenate(Y_, axis=0)

	# Save 
	np.save(os.path.join('data','dataX.npy'), X)
	np.save(os.path.join('data','dataY.npy'), Y)


## One-hot encoding
def one_hot(labels, n_class = 12):
	""" One-hot encoding """
	expansion = np.eye(n_class)
	y = expansion[:,labels-1].T

	return y

## Standardize
def standardize(X):
	""" Standardize by mean and std for each measurement channel"""
	return (X - np.mean(X, axis=0)[None,:,:]) / np.std(X, axis=0)[None,:,:]

## Get batches
def get_batches(X, y, batch_size = 100):
	""" Yield batches ffrom data """
	n_batches = len(X) // batch_size
	X, y = X[:n_batches*batch_size], y[:n_batches*batch_size]

	# Loop over batches and yield
	for b in range(0, len(X), batch_size):
		yield X[b:b+batch_size], y[b:b+batch_size]