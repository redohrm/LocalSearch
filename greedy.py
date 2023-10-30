#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# PROGRAMMER: Ruth Dohrmann
# PROGRAM: greedy.py
# Description: This program uses greedy local search (gradient ascent) to attempt to obtain the maximum
# value of a Sum of Gaussians function.

import SumofGaussians as SG
import numpy as np, sys

seed = int(sys.argv[1])
dims = int(sys.argv[2])
ncenters = int(sys.argv[3])

# set up the random number generator
rng = np.random.default_rng(seed)
# Set up the Sum of Gaussians function
sog = SG.SumofGaussians(dims,ncenters,rng)

epsilon = 1e-8

def main():
    # This function performs the greedy search and prints out the necessary output
    greedy_search()


# This function attempts to find the maximum value of the SoG function 
# by performing a greedy search.
def greedy_search():

    # The program starts in a random location in the [0,10] dims-cube, 
    # where x is a dims-dimensional vector.
    x = rng.uniform(size=dims) * 10.0

    iter_count = 0
    # 100,000: max number of iterations
    while iter_count < 100000:
        x_eval=sog.Evaluate(np.array(x))
        # print x and the value of the SoG function
        for j in range(len(x)):
            print(x[j], end=' ')
        print(x_eval)
        arr=sog.Gradient(np.array(x))
        newX = x.copy()
        # use a step size of (0.01 * gradient) to perform gradient ascent
        for i in range(len(arr)):
            newX[i] += arr[i]*.01
            # stay in bounds
            if newX[i] > 10 or newX[i] < 0:
                newX[i] = x[i]
        newX_eval = sog.Evaluate(np.array(newX))
        # terminate when the value of the function no longer increases 
        # (within 1e-8 tolerance)
        if (newX_eval - x_eval) < epsilon:
            return
        x = newX
        iter_count += 1
    return


main()
