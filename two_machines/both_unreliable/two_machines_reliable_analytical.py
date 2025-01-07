import numpy as np
import numpy.linalg as la


class TwoMachineLineBothUnreliable:
    def __init__(self, name: str, 
                 mu1: float, mu2: float,
                 p1: float, p2: float, 
                 r1: float, r2: float, 
                 C: int):
        self.name = name

        self.mu1 = mu1
        self.p1 = p1
        self.r1 = r1

        self.mu2 = mu2
        self.p2 = p2
        self.r2 = r2

        self.N = C + 2  # extended buffer size
        self.num_states = 4*(C + 3)
        self.Q = np.zeros((self.num_states, self.num_states))
        self.Qmod = np.zeros((self.num_states, self.num_states))
        self.nmod = np.zeros((1, self.num_states))
        self.pi = np.zeros((1, self.num_states))  # states prob.
        self.num_func = np.zeros((self.N+1, 2, 2)).astype((int)) # (n, alpha1, alpha2)
        
        count = 0
        for n in range(self.N+1):
            for alpha1 in [0, 1]:
                for alpha2 in [0, 1]:
                    # so states are numbered from 0
                    self.num_func[n, alpha1, alpha2] = count  
                    count += 1
            
        self.determineSteadyStateProbabilities()


    def initializeGeneratorMatrix(self):
        for n in range(self.N+1):
            for alpha1 in [0, 1]:
                for alpha2 in [0, 1]:
                    # first machine is not blocked; 
                    # can only fail if it can produce since it is ODF
                    if n < self.N and alpha1 == 1:  
                        self.Q[self.num_func[n, 1, alpha2], self.num_func[n+1, 1, alpha2]] = self.mu1
                        self.Q[self.num_func[n, 1, alpha2], self.num_func[n, 0, alpha2]] = self.p1

                    # second machine is not starved; 
                    # can only fail if it can produce since it is ODF
                    if n > 0 and alpha2 == 1:  
                        self.Q[self.num_func[n, alpha1, 1], self.num_func[n-1, alpha1, 1]] = self.mu2
                        self.Q[self.num_func[n, alpha1, 1], self.num_func[n, alpha1, 0]] = self.p2

                    # first machine down -> can be repaired
                    if alpha1 == 0:
                        self.Q[self.num_func[n, 0, alpha2], self.num_func[n, 1, alpha2]] = self.r1

                    # second machine down -> can be repaired
                    if alpha2 == 0:
                        self.Q[self.num_func[n, alpha1, 0], self.num_func[n, alpha1, 1]] = self.r2

        for i in range(self.num_states):
            for j in range(self.num_states):
                if i != j:
                    self.Q[i, i] -= self.Q[i, j]


    def determineSteadyStateProbabilities(self):
        self.initializeGeneratorMatrix()
        self.Qmod = self.Q.copy()
        numberOfRowsOfQ = self.Qmod.shape[0]
        
        for i in range(numberOfRowsOfQ):
            self.Qmod[i, numberOfRowsOfQ-1] = 1  # Qmod is quadratic
            
        # right hand side row vector with last element as a 1
        self.nmod[0][numberOfRowsOfQ-1] = 1

        self.pi = self.nmod.dot(la.inv(self.Qmod))
        
    
    def calc_TH1(self):
        return self.mu1 * sum([self.pi[0, self.num_func[n, 1, alpha2]] 
                               for n in range(self.N)
                               for alpha2 in [0, 1]])
     
    
    def calc_TH2(self):
        return self.mu2 * sum([self.pi[0, self.num_func[n, alpha1, 1]] 
                               for n in range(1, self.N+1)
                               for alpha1 in [0, 1]])
        
    def calc_n_bar(self):
        return sum([n*self.pi[0, self.num_func[n, alpha1, alpha2]]
                    for n in range(1, self.N+1)
                    for alpha1 in [0, 1]
                    for alpha2 in [0, 1]])


if __name__ == "__main__":
    # We now create an object of the class
    myTwoMachineLine = TwoMachineLineBothUnreliable("RobertsLine", 
                                                    mu1=1, mu2=1.1, 
                                                    p1=0.1, p2=0.2, 
                                                    r1=0.2, r2=0.4,
                                                    C=3)

    print("Vector of state probablities is:", myTwoMachineLine.pi)

    print("Throughput via Machine 1 is:",
        myTwoMachineLine.calc_TH1(),)

    print("Throughput via Machine 2 is:",
        myTwoMachineLine.calc_TH2(),)

    print("Average parts in the system is:", 
        myTwoMachineLine.calc_n_bar())
