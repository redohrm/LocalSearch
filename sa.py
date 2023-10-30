#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# PROGRAMMER: Ruth Dohrmann
# PROGRAM: sa.py
# Description: This program uses simulated annealing to attempt to obtain the maximum value of the
# Sum of Gaussians (SoG) function.

import SumofGaussians as SG
import numpy as np, sys
import math

seed = int(sys.argv[1])
dims = int(sys.argv[2])
ncenters = int(sys.argv[3])

# set up the random number generator
rng = np.random.default_rng(seed)
# Set up the Sum of Gaussians function
sog = SG.SumofGaussians(dims,ncenters,rng)

epsilon = 1e-8

def main():

    # This function performs the simulated annealing search and prints out the necessary output    
    simulated_annealing()

# This function attempts to find the local maximum by performing a simulated annealing search. 
def simulated_annealing():
    # The program starts in a random location in the [0,10] dims-cube, 
    # where x is a dims-dimensional vector.
    x = [(rng.uniform() * 10.0) for i in range(dims)]

    # elements of temperature calculation
    temp_max = 1000000000  # starting temperature
    temp_limit = 250

    # set up iteration count
    iter_count = 0
    max_iter = 100000
    # 100,000: max number of iterations
    while iter_count < max_iter:
        # set temperature
        temp = temp_max * (1-(iter_count/temp_limit))
        x_eval=sog.Evaluate(np.array(x))
        # print x and the value of the SoG function
        for j in range(len(x)):
            print(x[j], end=' ')
        print(x_eval)
        rand_move = [(rng.uniform(-0.05, 0.05)) for i in range(dims)]
        newX = x.copy()
        # calculate random move that may or may not be performed
        for i in range(len(rand_move)):
            newX[i] += rand_move[i]
            # do not move outside of bounds
            if newX[i] > 10 or newX[i] < 0:
                newX[i] = x[i]
        newX_eval = sog.Evaluate(np.array(newX))
        # if the move is beneficial, accept it
        if newX_eval > x_eval:
            x = newX
            # terminate when the value of the function no longer increases 
            # (within 1e-8 tolerance) if the temperature is at or below zero
            if (newX_eval - x_eval) < epsilon and temp <= 0:
                return
        # if the move is not beneficial, calculate the probability of
        # accepting the move
        elif temp > 0:
            rand_num = rng.uniform()
            threshold = math.exp((newX_eval-x_eval)/temp)
            if rand_num < threshold:
                x = newX
        iter_count += 1
    return


main()
