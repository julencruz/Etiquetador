__authors__ = ['1667663', '1667150']
__group__ = ''

import numpy as np
import utils
 

class KMeans:

    def __init__(self, X, K=1, options=None):
        """
         Constructor of KMeans class
             Args:
                 K (int): Number of cluster
                 options (dict): dictionary with options
            """
        self.num_iter = 0
        self.K = K
        self._init_X(X)
        self._init_options(options)  # DICT options
        
    #############################################################
    ##  THIS FUNCTION CAN BE MODIFIED FROM THIS POINT, if needed
    #############################################################
        
    def _init_X(self, X):
        """Initialization of all pixels, sets X as an array of data in vector form (PxD)
            Args:
                X (list or np.array): list(matrix) of all pixel values
                    if matrix has more than 2 dimensions, the dimensionality of the sample space is the length of
                    the last dimension
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        
        X = X.astype(float)
        dim = np.shape(X)
        if len(dim) <= 2:
            X = np.reshape(X, (-1,3))
        elif len(dim) == 3 and dim[2] == 3:
            X = np.reshape(X, (-1,3))
        elif len(dim) > 3:
            X = np.reshape(X, (dim[0]*dim[1],dim[-1]))

        self.X = X

    def _init_options(self, options=None):
        """
        Initialization of options in case some fields are left undefined
        Args:
            options (dict): dictionary with options
        """
        if options is None:
            options = {}
        if 'km_init' not in options:
            options['km_init'] = 'first'
        if 'verbose' not in options:
            options['verbose'] = False
        if 'tolerance' not in options:
            options['tolerance'] = 0.0001
        if 'max_iter' not in options:
            options['max_iter'] = 1000
        if 'fitting' not in options:
            options['fitting'] = 'ICD'  # within class distance.

        # If your methods need any other parameter you can add it to the options dictionary
        self.options = options

        #############################################################
        ##  THIS FUNCTION CAN BE MODIFIED FROM THIS POINT, if needed
        #############################################################

    def _init_centroids(self):
        """
        Initialization of centroids
        """

        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        # if self.options['km_init'].lower() == 'first':
        #     self.centroids = np.random.rand(self.K, self.X.shape[1])
        #     self.old_centroids = np.random.rand(self.K, self.X.shape[1])
        # else:
        #     self.centroids = np.random.rand(self.K, self.X.shape[1])
        #     self.old_centroids = np.random.rand(self.K, self.X.shape[1])
        
        self.centroids = np.empty((self.K, self.X.shape[1]), dtype=float)
        self.old_centroids = np.copy(self.centroids)
        
        if self.options['km_init'].lower() == 'first':
            indices = np.sort(np.unique(self.X, return_index=True, axis=0)[1])[:self.K]
            self.centroids = self.X[indices]

        elif self.options['km_init'].lower() == 'random':
            indices = np.unique(self.X, return_index=True, axis=0)[1]
            random = np.random.randint(low=0, high=len(indices), size=self.K)
            self.centroids = self.X[indices[random]]
            
        #La opción custom es coger las filas únicas más grandes por orden RGB
        elif self.options['km_init'].lower() == 'custom':
            indices = np.flip(np.sort(np.unique(self.X, return_index=True, axis=0)[1]))[:self.K]
            self.centroids = self.X[indices]

    def get_labels(self):
        """
        Calculates the closest centroid of all points in X and assigns each point to the closest centroid
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        dist  = distance(self.X, self.centroids)
        self.labels = np.argmin(dist, axis = 1)
        

    def get_centroids(self):
        """
        Calculates coordinates of centroids based on the coordinates of all the points assigned to the centroid
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        self.old_centroids = np.copy(self.centroids)
        for i in range(0, self.K):
                self.centroids[i] = np.mean((self.X[self.labels == i]), axis=0)
            

    def converges(self):
        """
        Checks if there is a difference between current and old centroids
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        tolerance = np.isclose(self.centroids, self.old_centroids, rtol=self.options['tolerance'] , atol=0)
        if tolerance.all() or self.num_iter >= self.options['max_iter']:
            return True
        else: 
            return False
            

    def fit(self):
        """
        Runs K-Means algorithm until it converges or until the number of iterations is smaller
        than the maximum number of iterations.
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        
        stop = False
        self._init_centroids()
        self.num_iter = 0
        while not stop:
            self.get_labels()
            self.get_centroids()
            self.num_iter += 1
            stop = self.converges()
            
    def withinClassDistance(self):
        """
         returns the within class distance of the current clustering
        """

        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        WCD = 0
        distancies = distance(self.X, self.old_centroids)
        for i in range(0, self.labels.shape[0]):
            WCD = WCD + (distancies[i, self.labels[i]] ** 2 )
        WCD = WCD / self.labels.shape[0]
        return WCD
            
    def find_bestK(self, max_K):
        """
         sets the best k analysing the results up to 'max_K' clusters
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        DEC = np.empty((max_K-2), dtype=float)
        for i in range(2, max_K):
            self.K = i
            self.fit()
            WCD = self.withinClassDistance()
            DEC[i-2] = WCD
            
        self.K = max_K
        for i in range(1, len(DEC)):
            percDec = 1-(DEC[i]/DEC[i-1])
            if percDec <= 0.2:
                self.K = i+1
                self.WCD = DEC[i-1]
                break
        
            
            
        
def distance(X, C):
    """
    Calculates the distance between each pixel and each centroid
    Args:
        X (numpy array): PxD 1st set of data points (usually data points)
        C (numpy array): KxD 2nd set of data points (usually cluster centroids points)

    Returns:
        dist: PxK numpy array position ij is the distance between the
        i-th point of the first set an the j-th point of the second set
    """

    #########################################################
    ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
    ##  AND CHANGE FOR YOUR OWN CODE
    #########################################################
    distance = np.empty((X.shape[0], C.shape[0]), dtype=float)
    
    # for i in range (X.shape[0]):
    #     for j in range(C.shape[0]):
    #         distance[i,j] = (np.linalg.norm(X[i].astype(np.longdouble) - C[j].astype(np.longdouble)))
    
    X_rep = np.repeat(X[:, np.newaxis, :], C.shape[0], axis=1)
    C_rep = np.repeat(C[np.newaxis, :, :], X.shape[0], axis=0)
    distance = np.linalg.norm(X_rep - C_rep, axis=2)
    
    return distance

def get_colors(centroids):
    """
    for each row of the numpy matrix 'centroids' returns the color label following the 11 basic colors as a LIST
    Args:
        centroids (numpy array): KxD 1st set of data points (usually centroid points)

    Returns:
        labels: list of K labels corresponding to one of the 11 basic colors
    """


    #########################################################
    ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
    ##  AND CHANGE FOR YOUR OWN CODE
    #########################################################
    
    indices = np.argmax(utils.get_color_prob(centroids), axis = 1)
    return utils.colors[indices]   
