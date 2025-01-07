# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 16:25:23 2022

@author: helber
"""

# Decomposition of N-station flow line
from TwoMachineLineReliable2023 import TwoReliableMachines
import numpy as np
import matplotlib.pyplot as plt

from IPython.display import set_matplotlib_formats
set_matplotlib_formats('svg')
from matplotlib import cm


class N_MachineLineReliable:
    def __init__(self, number_of_stations, mu_list, C_list):
        self.number_of_stations = number_of_stations
        self.C = C_list
        
        self.mu = mu_list
        self.mu_up= mu_list[0:len(mu_list)-1]
        self.mu_dn= mu_list[1:len(mu_list)]
        
    def determineThroughputAndInventory(self):   
        self.iterationCounter = 0
        self.ps = [0]*(self.number_of_stations - 1) # Starving probabilities
        self.pb = [0]*(self.number_of_stations - 1) # Blocking probabilities
        self.TP = [0]*(self.number_of_stations - 1) # Throughput
        self.nb = [0]*(self.number_of_stations - 1) # Average inventory
        
        # Initialize virtual lines wrt KPIs
        for i in range(self.number_of_stations - 1):
            currentTwoMachineLine = TwoReliableMachines("", 
                                                        self.mu_up[i],  
                                                        self.mu_dn[i], 
                                                        self.C[i])
            
            dummy, self.TP[i], self.ps[i], self.pb[i], self.nb[i] = currentTwoMachineLine.determineKPIs()
        
        NotReady = True
        while NotReady:
            self.iterationCounter = self.iterationCounter + 1
            # Remember offset, virtual line 1 is at position 0 !!
            # Forward pass
            for i in range(1, self.number_of_stations - 1):
                k_up = 1 / self.TP[i-1] + 1 / (self.mu[i]) - 1 / (self.mu_dn[i-1])
                  
                self.mu_up[i] = 1 / k_up 
                
                # Now update performance measures for the virtual two-machine line
                currentTwoMachineLine = TwoReliableMachines("", self.mu_up[i],  self.mu_dn[i], self.C[ i ])
                dummy, self.TP[i], self.ps[i], self.pb[i], self.nb[i] = currentTwoMachineLine.determineKPIs()
            
            # Backward pass
            for i in range(self.number_of_stations - 3, -1, -1):
                k_down = 1 /  self.TP[i+1] + 1 /(  self.mu[i+1]) - 1 / ( self.mu_up[i+1])
                
                self.mu_dn[i] = 1 / k_down 
    
                # Now update performance measures for the virtual two-machine line
                currentTwoMachineLine = TwoReliableMachines("", self.mu_up[i], self.mu_dn[i], self.C[ i ])
                dummy, self.TP[i], self.ps[i], self.pb[i], self.nb[i] = currentTwoMachineLine.determineKPIs()
            
            # Check for convergence
            # when throughput is the same for all virtual lines
            if abs(self.TP[0] - self.TP[self.number_of_stations - 2]) / self.TP[0] < 0.000001:
                NotReady = False
      
if __name__ == "__main__":               
    myLongLine = N_MachineLineReliable(number_of_stations=4, 
                                    mu_list=[10, 10, 10, 10],
                                    C_list=[1000,1000,1000])

    myLongLine.determineThroughputAndInventory()

    print("\nThroughput: ", myLongLine.TP)
    # print("Average inventory: ", myLongLine.nb)
    # print("Blocking probabilities: ", myLongLine.pb)
    # print("Starving probabilities: ", myLongLine.ps)
    print("Number of iterations required: ", myLongLine.iterationCounter)
