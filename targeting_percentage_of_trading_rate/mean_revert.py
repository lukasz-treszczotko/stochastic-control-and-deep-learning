#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 09:18:34 2018

An example of stochastic mean-reverting trading rate with exponential jumps

@author: lukasz
"""

import numpy as np
from numba import jit


@jit
def helper(timestep, poisson_path, kappa=1., mu_0=0.3, T=1, exp_rate=0.1):
        dt = T/float(len(poisson_path))
        #print("dt:", dt)
        #num_jumps = poisson_path[timestep]
        mask = poisson_path - np.roll(poisson_path, 1)
        temp = (mask>0).astype(int)
        jump_indices = np.argwhere(temp>0)
        #print(jump_indices)
        result = 0.
        for jump_time in jump_indices:
            if jump_time <= timestep:
                result += np.exp(-kappa*(timestep*dt-jump_time*dt))*np.random.exponential(exp_rate)
        result += np.exp(-kappa*timestep*dt)*mu_0
        #print(result)
        return result

    

@jit
def helper_2(poisson_path, kappa=1., exp_rate=0.1):
    
    result = np.zeros_like(poisson_path, dtype=np.float32)
    for timestep in range(len(poisson_path)):
        result[timestep] = float(helper(timestep, poisson_path, kappa=kappa, exp_rate=exp_rate))
        #print(result[timestep])
    return result
        
                
        
@jit
def mu_process(sample_length, kappa, lambda_poisson, exp_rate, 
               T=1., num_samples=1000):
    #mu_0 = 1.
    dt = T/sample_length
    #timesteps = np.linspace(0, T-dt, sample_length)
    poisson_rates = np.random.poisson(dt * lambda_poisson, 
                                      (num_samples, sample_length))
    
    poisson_processes = np.cumsum(poisson_rates, axis=1)
    result = np.zeros_like(poisson_processes, dtype=np.float32)
    for j in range(poisson_processes.shape[0]):
        #np.random.seed(j)
        result[j]= helper_2(poisson_processes[j], kappa=kappa, exp_rate=exp_rate)
        #print(result[j])
    return result

ex_data = mu_process(200, 
                     0.5, 
                     7., 
                     0.5, 
                     num_samples=10000)

import h5py
hf = h5py.File('ex_data.h5', 'w')
hf.create_dataset('ex_data', data=ex_data)
hf.close()
    
        
    
    
