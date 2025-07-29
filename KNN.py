__authors__ = ['1667663', '1667150']
__group__ = ''

import numpy as np
import math
import operator
from scipy.spatial.distance import cdist


class KNN:
    def __init__(self, train_data, labels):
        self._init_train(train_data)
        self.labels = np.array(labels)
        #############################################################
        ##  THIS FUNCTION CAN BE MODIFIED FROM THIS POINT, if needed
        #############################################################

    def _init_train(self, train_data):
        """
        initializes the train data
        :param train_data: PxMxNx3 matrix corresponding to P color images
        :return: assigns the train set to the matrix self.train_data shaped as PxD (P points in a D dimensional space)
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        train_data = np.reshape(train_data, (train_data.shape[0],train_data.shape[1]*train_data.shape[2]))
        self.train_data = train_data
        self.train_data.astype(float)
                                
    def get_k_neighbours(self, test_data, k):
        """
        given a test_data matrix calculates de k nearest neighbours at each point (row) of test_data on self.neighbors
        :param test_data: array that has to be shaped to a NxD matrix (N points in a D dimensional space)
        :param k: the number of neighbors to look at
        :return: the matrix self.neighbors is created (NxK)
                 the ij-th entry is the j-th nearest train point to the i-th test point
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        test_data = np.reshape(test_data, (test_data.shape[0],test_data.shape[1]*test_data.shape[2]))
        self.neighbors = self.labels[np.argsort(cdist(test_data, self.train_data, 'euclidean'), axis=1)[:, :k]]

    def get_class(self):
        """
        Get the class by maximum voting
        :return: 1 array of Nx1 elements. For each of the rows in self.neighbors gets the most voted value
                (i.e. the class at which that row belongs)
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        arrclass = np.empty((self.neighbors.shape[0],), dtype=object)
        for i in range(self.neighbors.shape[0]):
            max = ['', 0]
            for j in self.neighbors[i]:
                contador = list(self.neighbors[i]).count(j)
                if contador > max[1]:
                    max = [j, list(self.neighbors[i]).count(j)]
            arrclass[i] = max[0]
        return arrclass
        
                
            

    def predict(self, test_data, k):
        """
        predicts the class at which each element in test_data belongs to
        :param test_data: array that has to be shaped to a NxD matrix (N points in a D dimensional space)
        :param k: the number of neighbors to look at
        :return: the output form get_class a Nx1 vector with the predicted shape for each test image
        """
        self.get_k_neighbours(test_data, k)
        return self.get_class()
