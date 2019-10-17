# ps1_functions.py
# Skeleton file by Chris Harshaw, Yale University, Fall 2017
# Adapted by Jay Stanley, Yale University, Fall 2018
# Adapted by Scott Gigante, Yale University, Fall 2019
# Skeleton code defined and created by Mark Torres, Yale University, Fall 2019
# CPSC 553 -- Problem Set 1
#
# This script contains uncompleted functions for implementing diffusion maps.
#
# NOTE: please keep the variable names that I have put here, as it makes grading easier.

# import required libraries
import numpy as np
import codecs, json
import math
import scipy
from scipy import linalg


##############################
# Predefined functions
##############################

def load_json_files(file_path):
    '''
    Loads data from a json file

    Inputs:
        file_path   the path of the .json file that you want to read in

    Outputs:
        json_data    this is a numpy array if data is numeric, it's a list if it's a string

    '''

    #  load data from json file
    with codecs.open(file_path, 'r', encoding='utf-8') as handle:
        json_data = json.loads(handle.read())

    # if a string, then returns list of strings
    if not isinstance(json_data[0], str):
        # otherwise, it's assumed to be numeric and returns numpy array
        json_data = np.array(json_data)

    return json_data


##############################
# Skeleton code (fill these in)
##############################

def compute_distances(X):
    '''
    Constructs a distance matrix from data set, assumes Euclidean distance

    Inputs:
        X       a numpy array of size n x p holding the data set (n observations, p features)

    Outputs:
        D       a numpy array of size n x n containing the euclidean distances between points

    '''

    # initialize # of observations (n)

    n = len(X)

    # initialize distance array:
    D = np.empty(shape = (n, n))

    # impute values for distance array:
    for i in range(n):
        for j in range(n):

            # initialize X, Y values
            x = X[i].copy()
            y = X[j].copy()

            # calculate distance
            distance = np.linalg.norm(x - y)

            # set that value to D
            D[i, j] = distance
            
    # return distance matrix
    return D


def compute_affinity_matrix(D, kernel_type, sigma=None, k=None):
    '''
    Construct an affinity matrix from a distance matrix via gaussian kernel.

    Inputs:
        D               a numpy array of size n x n containing the distances between points
        kernel_type     a string, either "gaussian" or "adaptive".
                            If kernel_type = "gaussian", then sigma must be a positive number
                            If kernel_type = "adaptive", then k must be a positive integer
        sigma           the non-adaptive gaussian kernel parameter
        k               the adaptive kernel parameter

    Outputs:
        W       a numpy array of size n x n that is the affinity matrix

    '''

    # construct affinity matrix if kernel_type == "gaussian"
    if kernel_type == "gaussian":

        # make sure that sigma is positive
        assert sigma > 0 

        # initialize W
        numRows = len(D)
        numCols = len(D[0])

        W = np.empty(shape = (numRows, numCols))

        # create kernel

        # iterate through rows and columns
        for i in range(numRows):
           for j in range(numCols):

                W[i, j] = np.exp((-1 * (D[i, j] **2)) / sigma ** 2)

    # construct affinity matrix if kernel_type == "adaptive"
    elif kernel_type == "adaptive":
        
        # make sure that k is a positive integer
        assert k > 0 and isinstance(k, int)


        # define the knn kernel
        def knn_kernel(distance, sigma_x, sigma_y):

            '''
                Calculates knn kernel value

                Inputs:
                    distance     distance between x and y    
                    sigma_x      the distance from x to its kth neighbor
                    sigma_y      the distance from y to its kth neighbor

                Outputs:
                    kernel       output of the kernel function

            '''

            # calculate exponential values
            exp_x = np.exp((-1 * (distance**2)) / sigma_x ** 2)
            exp_y = np.exp((-1 * (distance**2)) / sigma_y ** 2)

            # calculate kernel
            kernel = 0.5 * (exp_x + exp_y)

            # return kernel
            return kernel

        # define sigma, the distance to point X's kth nearest neighbor
        def sigma(row, k):

            '''
                Calculates the distance to point X's kth nearest neighbor

                Inputs:
                    row          The row of the distance matrix
                    k            The kth neighbor

                Outputs:
                    kth_distance The distance of the kth neighbor

            '''

            # order the distances, in descending order
            ordered_distances = np.sort(row)[::-1]

            # select the k-th entry
            kth_distance = ordered_distances[k - 1]

            # return the kth distance
            return kth_distance

        

        ### iterate process for each point:

        # initialize empty matrix, with the same shape as D (so, n x n)
        numRows = len(D)
        numCols = len(D[0])

        W = np.empty(shape = (numRows, numCols))

        # iterate through rows and columns
        for i in range(numRows):
           for j in range(numCols):

                # set array of Xs
                xArray = D[i]

                # set correct value for y, as well as array of Ys
                yArray = D[j]

                # calculate the correct sigma for each
                sigma_x = sigma(xArray, k)
                sigma_y = sigma(yArray, k)

                # get distance between the two points
                distance = D[i, j]

                # calculate the kernel value
                kernel = knn_kernel(distance, sigma_x, sigma_y)

                # add to W
                W[i, j] = kernel

    # return the affinity matrix
    return W


def diff_map_info(W):
    '''
    Construct the information necessary to easily construct diffusion map for any t

    Inputs:
        W           a numpy array of size n x n containing the affinities between points

    Outputs:

        diff_vec    a numpy array of size n x n-1 containing the n-1 nontrivial eigenvectors of Markov matrix as columns
        diff_eig    a numpy array of size n-1 containing the n-1 nontrivial eigenvalues of Markov matrix

        We assume the convention that the coordinates in the diffusion vectors are in descending order
        according to eigenvalues.
    '''

    # define a diagonal matrix D, where the diagonal entries are the row sums of W
    rowSums = np.sum(W, axis = 1)
    D = np.diag(rowSums)

    # compute Markov matrix by D^.5 * W * W^-.5
    sqrtD = scipy.linalg.fractional_matrix_power(D, 0.5)
    negativeSqrtD = scipy.linalg.fractional_matrix_power(D, -0.5)
    M = np.matmul(sqrtD, np.matmul(W, negativeSqrtD))

    # Normalize
    M = M / np.sum(M, axis = 1, keepdims = True)


    # get eigenvalues, eigenvectors of Markov matrix, where eigenvecs[:, i] is the normalized eigenvector corresponding to eigenvals[i]
    eigenvals, eigenvecs = np.linalg.eigh(M)

    # flip the eigenvals and eigenvecs so they're in descending order
    diff_eig_desc = eigenvals[::-1]
    diff_vec = np.flip(eigenvecs, axis = 0)

    # drop last column of diff_eig and diff_vec (to get nontrivial eigenvectors and eigenvalues)
    diff_eig = diff_eig_desc[:-1].copy() # drop last value
    diff_vec = np.delete(diff_vec, -1, axis = 1) # drop last column

    # return the info for diffusion maps
    return diff_vec, diff_eig

def get_diff_map(diff_vec, diff_eig, t):
    '''
    Construct a diffusion map at t from eigenvalues and eigenvectors of Markov matrix

    Inputs:
        diff_vec    a numpy array of size n x n-1 containing the n-1 nontrivial eigenvectors of Markov matrix as columns
        diff_eig    a numpy array of size n-1 containing the n-1 nontrivial eigenvalues of Markov matrix
        t           diffusion time parameter t

    Outputs:
        diff_map    a numpy array of size n x n-1, the diffusion map defined for t
    '''

    # raise the eigenvalues to the t-th power

    eigVals_raised = np.power(diff_eig, [t])

    # multiply eigenvalues to eigenvectors
    diff_map = eigVals_raised * diff_vec 

    # return diffusion map
    return diff_map





