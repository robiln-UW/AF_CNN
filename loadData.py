import os
import glob
import csv
import matplotlib.pyplot as plt
import numpy as np
import random
import math
from scipy.io import loadmat


labelMapper = {'N':0,'A':1,'O':2,'~':3}


def read_data(path,config):

    data_size = config.data_size 
    # create dictionary of labels
    with open(path + '/REFERENCE.csv',mode='r') as infile:
        reader = csv.reader(infile)
        labelDict = {rows[0]:labelMapper[rows[1]] for rows in reader}



    # load data from file

    train_labelList = []
    train_dataList = []

    valid_labelList = []
    valid_dataList = []

    test_labelList = []
    test_dataList = []
    
    num_files = 8528
    end_valid = math.floor(num_files*0.75)
    end_train = math.floor(end_valid*0.75)
    files_read = 0
    
    for filename in glob.glob(os.path.join(path, '*.mat')):
        files_read += 1
        # get sample name from filename
        _,t = os.path.split(filename)
        sampleName = t[:-4]

        # find correct label for datapoint
        label = labelDict[sampleName]

        # read ECG data from file
        ECGData = loadmat(filename)['val'][0,:]

        # split data into data_size sized chunks
        for i in range(len(ECGData)//(data_size+1)):
            subData = ECGData[(data_size+1)*i:(data_size+1)*(i+1)-1]

            if files_read < end_train:  # belongs in train set
                train_dataList.append(subData)
                train_labelList.append(label)
                # add more AF data points
                if (label == labelMapper['A']):
                    for _ in range(3):
                        train_dataList.append(subData)
                        train_labelList.append(label)
                # add reversed noisy data
                if (label == labelMapper['~']):
                    train_dataList.append(np.flip(subData,axis=0))
                    train_labelList.append(label)
            elif files_read < end_valid: # belongs in validation set
                valid_dataList.append(subData)
                valid_labelList.append(label)
            else:  # belongs in test set
                test_dataList.append(subData)
                test_labelList.append(label)
    
    train_data = np.asarray(train_dataList)
    train_labels = np.array(train_labelList)
    valid_data = np.asarray(valid_dataList)
    valid_labels = np.asarray(valid_labelList)
    test_data = np.asarray(test_dataList)
    test_labels = np.asarray(test_labelList)
    
    return train_data, train_labels, valid_data, valid_labels, test_data, test_labels


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

