import numpy as np
import numpy.linalg as la


class TwoMachineLineFirstUnreliable:
    def __init__(self, name, mu1, mu2, p1, r1, C):
        self.name = name
        self.mu1 = mu1
        self.mu2 = mu2
        self.p1 = p1
        self.r1 = r1
        self.N = C + 2  # extended buffer size
        self.num_states = 2*(C + 3)
        self.Q = np.zeros((self.num_states, self.num_states))
        self.Qmod = np.zeros((self.num_states, self.num_states))
        self.nmod = np.zeros((1, self.num_states))
        self.pi = np.zeros((1, self.num_states))  # states prob.
        self.num_func = np.zeros((self.N+1, 2)).astype((int))
        
        count = 0
        for n in range(self.N+1):
            for alpha1 in [0, 1]:
                # so states are numbered from 0
                self.num_func[n, alpha1] = count  
                count += 1
            
        self.determineSteadyStateProbabilities()

    def initializeGeneratorMatrix(self):
        for n in range(self.N+1):
            for alpha1 in [0, 1]:
                # second machine is not starved
                if n > 0: 
                    self.Q[self.num_func[n,alpha1], self.num_func[n - 1,alpha1]] = self.mu2

                # first machine is not blocked
                if n < self.N and alpha1 == 1:
                    self.Q[self.num_func[n,1], self.num_func[n + 1,alpha1]] = self.mu1
                    self.Q[self.num_func[n,1], self.num_func[n,0]] = self.p1

                elif alpha1 == 0:
                    self.Q[self.num_func[n,0], self.num_func[n,1]] = self.r1

        for i in range(self.num_states):
            for j in range(self.num_states):
                if i != j:
                    self.Q[i, i] -= self.Q[i, j]

    def determineSteadyStateProbabilities(self):
        self.initializeGeneratorMatrix()
        self.Qmod = self.Q.copy()
        numberOfRowsOfQ = self.Qmod.shape[0]
        
        for i in range(numberOfRowsOfQ):
            self.Qmod[i, numberOfRowsOfQ - 1] = 1  # Qmod is quadratic
            
        # right hand side row vector with last element as a 1
        self.nmod[0][numberOfRowsOfQ - 1] = 1

        self.pi = self.nmod.dot(la.inv(self.Qmod))
        
    
    def calc_TH1(self):
        return self.mu1 * sum([self.pi[0, self.num_func[n, 1]] 
                               for n in range(self.N)])
     
    
    def calc_TH2(self):
        return self.mu2 * (1 - sum([self.pi[0, self.num_func[0, alpha1]] 
                                    for alpha1 in [0, 1]]))
        
    def calc_n_bar(self):
        return sum([n*self.pi[0, self.num_func[n, alpha1]] 
                    for n in range(self.N+1) 
                    for alpha1 in [0, 1]])


# We now create an object of the class
myTwoMachineLine = TwoMachineLineFirstUnreliable("StefansLine", 
                                                  mu1=1, 
                                                  mu2=8, 
                                                  p1=0.1, 
                                                  r1=0.2, 
                                                  C=200)

print("Vector of state probablities is:", myTwoMachineLine.pi)

print("Throughput via Machine 1 is:",
    myTwoMachineLine.calc_TH1(),)

print("Throughput via Machine 2 is:",
    myTwoMachineLine.calc_TH2(),)

print("Average parts in the system is:", 
      myTwoMachineLine.calc_n_bar())
