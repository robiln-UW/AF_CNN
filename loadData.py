import os
import glob
import csv
import matplotlib.pyplot as plt
import numpy as np
import random
import math
from scipy.io import loadmat


labelMapper = {'N':0,'A':1,'O':2,'~':3}
max_data_length = 18286

def read_data(path):
    
    # create dictionary of labels
    with open(path + '/REFERENCE.csv',mode='r') as infile:
        reader = csv.reader(infile)
        labelDict = {rows[0]:labelMapper[rows[1]] for rows in reader}


    labelList = []
    dataList = []
    for filename in glob.glob(os.path.join(path, '*.mat')):

        # get sample name from filename
        _,t = os.path.split(filename)
        sampleName = t[:-4]

        # find correct label for datapoint
        label = labelDict[sampleName]

        # read ECG data from file
        ECGData = loadmat(filename)['val'][0,:]
        ECGData = np.pad(ECGData,(0,max_data_length-len(ECGData)),'wrap')
        dataList.append(ECGData)
        labelList.append(label)

    
    data = np.asarray(dataList)
    labels = np.array(labelList)
    return data, labels


class DataSet(object):
    def __init__(self,data, labels):
        assert data.shape[0] == labels.shape[0], ('data.shape {0} labels.shape {1}'.format(data.shape, labels.shape))

        self.num_samples = data.shape[0]
        self._data = data
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def data(self):
        return self._data

    @property
    def labels(self):
        return self._labels

    def next_batch(self, batch_size, shuffle=True):
        start = self._index_in_epoch
        if start == 0 and shuffle: # start over
            indices = np.arange(self.num_samples)
            np.random.shuffle(indices) # randomize indices
            self._data = self.data[indices]
            self._labels = self.labels[indices]

        if start + batch_size > self.num_samples:
            self._epochs_completed += 1
            self._index_in_epoch = 0
            end = self.num_samples
        else:
            end = start + batch_size
            self._index_in_epoch += batch_size
        return self._data[start:end], self._labels[start:end]

