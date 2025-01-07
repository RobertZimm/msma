# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 14:55:48 2022

@author: helber
"""

import numpy as np
import numpy.linalg as la




class TwoReliableMachines:
    def __init__(self, name, mu1, mu2, C):
        self.name = name
        self.mu1 = mu1
        self.mu2 = mu2
        self.C   = C
        self.N   = C + 2
        self.NumberOfStates = C + 3        
        self.Q = np.zeros(( self.NumberOfStates, self.NumberOfStates ))
        self.Qmod = np.zeros(( self.NumberOfStates, self.NumberOfStates ))
        self.nmod = np.zeros((1, self.NumberOfStates ))
        self.pi = np.zeros((1, self.NumberOfStates ))
        self.StN = np.arange(self.NumberOfStates + 1 )
        counter = -1
        for n in range( self.NumberOfStates ):
            counter = counter + 1
            self.StN[ n ] = counter # so states are numbered from 0
            
                  
            
                      
            
            
        
    def initializeGeneratorMatrix (self):
        
        for i in range( self.NumberOfStates):
            self.Q[i][i] = 0
                        
            
        for n in range( self.NumberOfStates ):
            if n < self.N:      # first machine is not blocked 
                self.Q[ self.StN[ n ], self.StN[ n + 1 ]] = self.mu1
            if n > 0:           # second machine is not starving
                self.Q[ self.StN[ n ], self.StN[ n - 1 ]] = self.mu2
        
        for i in range( self.NumberOfStates ):
            for j in range( self.NumberOfStates ):
                if i != j:
                    self.Q[i][i] = self.Q[i][i] - self.Q[i][j]
                
        
        
    def determineStateProbabilities (self):
        
        self.initializeGeneratorMatrix()
        
        self.Qmod = self.Q.copy()
        
        size = self.Qmod.shape[0]
        
        for i in range(size):
            self.Qmod[i][size-1] = 1
        
        self.nmod[0][size-1] = 1
        
        # print("Qmod is \n", self.Qmod, "\nnmod is \n", self.nmod)
        
        self.pi = self.nmod.dot(la.inv( self.Qmod )) 
        
        # print("Vector of state probabilities is ", self.pi)       
        
       

        
    def determineThroughput(self ):
        self.determineStateProbabilities()
        print()
        print("Throughput via Machine 1 is ", self.mu1*( 1 - self.pi[0][ self.StN[ self.N ]]))
        print("Throughput via Machine 2 is ", self.mu2*( 1 - self.pi[0][ self.StN[ 0 ]]))

        
     
    def determineKPIs( self ):
        self.determineStateProbabilities()
        TP1 = self.mu1*( 1 - self.pi[0][ self.StN[ self.N ]] )
        TP2 = self.mu2*( 1 - self.pi[0][ self.StN[ 0 ] ] )
        
        ps = self.pi[0][ self.StN[ 0      ]] 
        pb = self.pi[0][ self.StN[ self.N ]]
        
        nb = 0
        for n in range( self.N + 1 ):
            nb = nb + n * ( self.pi[0][ self.StN[ n]]  )
            
        return TP1, TP2, ps, pb, nb
      

         
         
if __name__ == "__main__":     
    myTwoMachineLine = TwoReliableMachines("X12", 0.2, 0.3, 3)

    TP1, TP2, ps, pb, nb = myTwoMachineLine.determineKPIs()

    print( TP1, TP2, ps, pb, nb)
