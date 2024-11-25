#   Numerical solution of a CTMC moded of a limited buffer, two-machine flow line.

#   Author: Stefan Helber, Date: November 20, 2024

import numpy as np
import numpy.linalg as la


class TwoMachineLineBothReliable:
    def __init__(self, name, mu1, mu2, C):
        self.name = name
        self.mu1 = mu1
        self.mu2 = mu2
        self.N = C + 2  # extended buffer size
        self.NumberOfStates = C + 3
        self.Q = np.zeros((self.NumberOfStates, self.NumberOfStates))
        self.Qmod = np.zeros((self.NumberOfStates, self.NumberOfStates))
        self.nmod = np.zeros((1, self.NumberOfStates))
        self.pi = np.zeros((1, self.NumberOfStates))  # states prob.
        self.StN = np.arange(self.NumberOfStates)
        counter = -1
        for n in range(self.NumberOfStates):
            counter += 1
            self.StN[n] = counter  # so states are numbered from 0
            
        self.determineSteadyStateProbabilities()

    def initializeGeneratorMatrix(self):
        for n in range(self.NumberOfStates):
            if n < self.N:  # first machine is not blocked
                self.Q[self.StN[n], self.StN[n + 1]] = self.mu1
            if n > 0:  # second macine is not starved
                self.Q[self.StN[n], self.StN[n - 1]] = self.mu2

        for i in range(self.NumberOfStates):
            for j in range(self.NumberOfStates):
                if i != j:
                    self.Q[i][i] = self.Q[i][i] - self.Q[i][j]

    def determineSteadyStateProbabilities(self):
        self.initializeGeneratorMatrix()
        self.Qmod = self.Q.copy()
        numberOfRowsOfQ = self.Qmod.shape[0]
        for i in range(numberOfRowsOfQ):
            self.Qmod[i][numberOfRowsOfQ - 1] = 1  # Qmod is quadratic
        # right hand side row vector with last element as a 1
        self.nmod[0][numberOfRowsOfQ - 1] = 1

        self.pi = self.nmod.dot(la.inv(self.Qmod))
        
    
    def calc_TH1(self):
        return self.mu1 * (1 - self.pi[0][self.StN[self.N]])
        
    
    def calc_TH2(self):
        return self.mu2 * (1 - self.pi[0][self.StN[0]])
        
    def calc_n_bar(self):
        return sum([i * self.pi[0][i] for i in range(self.NumberOfStates)])


# We now create an object of the class
myTwoMachineLine = TwoMachineLineBothReliable("StefansLine", 10, 8, 2)

print("Vector of state probablities is:", myTwoMachineLine.pi)

print("Throughput via Machine 1 is:",
    myTwoMachineLine.calc_TH1(),)

print("Throughput via Machine 2 is:",
    myTwoMachineLine.calc_TH2(),)

print("Average parts in the system is:", 
      myTwoMachineLine.calc_n_bar())
