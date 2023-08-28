import numpy as np
from math import comb
from collections import deque


def fixed_weight_iterator(weight, end=None):
    '''Iterator for positive binary numbers of a given Hamming weight, in sorted order

    Parameters:
    weight: Hamming weight, a positive integer
    end: only numbers up to end-1 are output (optional, default is None)

    Returns:
    An iterator object for the sequence
    '''
    assert(weight > 0)
    x = (1<<weight) - 1 # smallest integer with the given Hamming weight
    while (end is None) or (x < end):
        yield x
        xlsb = x & -x # xlsb = least significant bit of x
        y = x + xlsb # replace righmost block of 1s in x with a 1 immediately to its left
        ylsb = y & -y # ylsb = least significant bit of y
        x = y + ylsb//(2*xlsb) - 1 # update x to the next integer with the same Hamming weight


def bit_iterator(x):
    '''Iterate over 1s in binary representation of x, from least to most significant
    
    Parameters:
    x: a positive integer
    
    Returns:
    an iterator object for the sequence
    '''
    while x>0:
        lsb = x & -x # lsb = least significant bit of x
        yield lsb
        x ^= lsb # clear least significant bit of x


def power_2_iterator(end = None):
    '''Iterate over non-negative powers of 2 in increasing order
    
    Parameters:
    end: only numbers up to end-1 are output (optional, default is None)
    
    Returns:
    an iterator object for the sequence
    '''
    x = 1
    while (end is None) or (x < end):
        yield x
        x *= 2


def smooth_binary_mechanism_noise(T, rho = 1., dimensions = 1, noise_generator = np.random.normal, neutral_element = 0.):
    '''Compute additive noise vector for private counting under continual observation
    
    This method implements the "Smooth Binary Mechanism".

    Adding the noise vector to the prefix sum vector of x in {0,1}^T will make
    it differentially private (specifically, rho-zCDP) with respect to the
    neighboring relation that changes a single bit of x.
    
    The space usage of the iterator is proportional to log T, it uses constant time
    on average for generating each value, uses less noise than binary mechanism, and
    has the same (Gaussian) noise distribution in every step. 
    See https://arxiv.org/abs/2306.09666 for details.
    
    Parameters:
    T: number of time steps
    rho: privacy parameter (optional, default is 1.)
    dimensions: number of noise values output per time step (optional, default is 1)
    noise_generator: generator for Gaussian noise (optional, default is Numpy's)
    neutral_element: zero noise value (optional, default is floating point zero)
    
    Returns:
    an iterator object for a sequence of T noise vectors
    '''
    
    # Initialization
    depth = 0
    while comb(depth, depth//2) <= length: # find smallest tree with > T balanced leaves
        depth += 2
    variance = depth/(4*rho) # variance per node to achieve rho-zCDP
    noise = {} # dictionary mapping node depths to noise
    leaves = fixed_weight_iterator(depth//2, 1<<depth)
    
    # Iteration
    n = neutral_element # Current noise value, always equal to sum of values in noise dict
    l1, l2 = None, 0 # Two adjacent leaves currently considered, paths encoded in binary
    for _ in range(length):
        l1, l2 = l2, next(leaves)
        for b in power_2_iterator(l1 ^ l2): # Iterate over bit positions after longest common prefix
            if b & l1 > 0: # Remove nodes from path to previous leaf l1
                n -= noise[b]
                del noise[b]
            if b & l2 > 0: # Add nodes from path to next leaf l2
                if b not in noise:
                    noise[b] = noise_generator(0, variance, size=dimensions)
                n += noise[b]
        yield n


def binary_mechanism_noise(T, rho = 1., dimensions = 1, noise_generator = np.random.normal, neutral_element = 0.):
    '''Compute additive noise vector for private counting under continual observation
    
    This method implements an efficient variant of the classical "Binary Mechanism".

    Adding the noise vector to the prefix sum vector of x in {0,1}^T will make
    it differentially private (specifically, rho-zCDP) with respect to the
    neighboring relation that changes a single bit of x.
    
    The space usage of the iterator is proportional to log T, it uses constant time
    on average for generating each value. See https://arxiv.org/abs/2306.09666 for details.
    
    Parameters:
    T: number of time steps
    rho: privacy parameter (optional, default is 1.)
    dimensions: number of noise values output per time step (optional, default is 1)
    noise_generator: generator for Gaussian noise (optional, default is Numpy's)
    neutral_element: zero noise value (optional, default is floating point zero)
    
    Returns:
    an iterator object for a sequence of T noise vectors
    '''
    
    # Initialization
    depth = 0
    while 2**depth <= length: # find smallest tree with > T leaves
        depth += 1
    variance = depth/(2*rho) # variance per node to achieve rho-zCDP
    noise = {} # dictionary mapping node depths to noise
    leaves = iter(range(1, length+1))
    
    # Iteration
    n = neutral_element # Current noise value, always equal to sum of values in noise dict
    l1, l2 = None, 0 # Two adjacent leaves currently considered, paths encoded in binary
    for _ in range(length): # Invariant: 1s in l2 have stored noise, summing to n
        l1, l2 = l2, next(leaves)
        for b in power_2_iterator(l1 ^ l2): # Iterate over bit positions after longest common prefix
            if b & l1 > 0: # Remove nodes from path to previous leaf
                n -= noise[b]
                del noise[b]
            if b & l2 > 0: # Add nodes from path to next leaf
                if b not in noise:
                    noise[b] = noise_generator(0, variance, size=dimensions)
                n += noise[b]
        yield n
