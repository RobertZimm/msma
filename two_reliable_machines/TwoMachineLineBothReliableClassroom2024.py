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


# We now create an object of the class

myTwoMachineLine = TwoMachineLineBothReliable("StefansLine", 10, 8, 2)

myTwoMachineLine.initializeGeneratorMatrix()

print(myTwoMachineLine.Q)
