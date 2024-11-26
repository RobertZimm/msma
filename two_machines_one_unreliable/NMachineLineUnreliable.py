import numpy as np
import random


class UnreliableProductionLine:
    # Definition of codes for events
    PROCESS_STEP_COMPLETED = 1
    MACHINE_FAILURE = 2
    MACHINE_REPAIR = 3

    # Machine states
    MACHINE_UP = 1
    MACHINE_DOWN = 0


    def __init__(self, 
                 mu: np.ndarray, 
                 r: np.ndarray, 
                 p: np.ndarray, 
                 C: np.ndarray) -> None:
        self.mu = mu
        self.r = r
        self.p = p
        self.C = C
        self.num_machines = len(self.mu)


    def simulate(self, 
                 sim_duration: int, 
                 seed: int = 4711):
        # Initialize the random number generator with a given seed.
        rng = np.random.default_rng(seed)

        if len(self.C) != self.num_machines - 1:
            raise ValueError("Sizes of machine array and buffer array don't fit!")

        trans_time = sim_duration/10

        # Initialize the state variables
        self.parts_processed = np.zeros(self.num_machines, 
                                dtype=int)
        self.ext_buffer_level = np.zeros(self.num_machines - 1, 
                                    dtype=int)
        self.time_until_next_part = np.zeros(self.num_machines, 
                                        dtype=float)
        
        self.machine_states = np.ones(self.num_machines, 
                                dtype=int)
        
        # If machine up: time until failure
        # If machine down: time until repair
        self.time_until_state_change = np.zeros(self.num_machines, 
                                            dtype=float)

        # Processing time of the first workpiece on the first machine ( offset 0 )
        self.time_until_next_part[0] = rng.exponential(1/mu[0])

        # Initialize the other machines with plus infinity
        self.time_until_next_part[1:] = np.inf
        
        self.time_until_state_change[0] = np.random.exponential(1/p[0]) 
        self.time_until_state_change[1:] = np.inf

        # Start simulation of this Markovian system
        sim_clock = -trans_time

        while sim_clock < sim_duration:
            time_until_next_event = np.inf
            next_machine = np.inf

            # Choose initial event which does not exist
            next_event_type = 0

            # Get production ready machine 
            # with smallest time until next part
            # and smallest time until state change
            for n in range(self.num_machines):
                if (self.is_prod_ready(n) 
                    and self.time_until_next_part[n] < time_until_next_event):
                    time_until_next_event = self.time_until_next_part[n]
                    next_machine = n 
                    next_event_type = self.PROCESS_STEP_COMPLETED

                if self.time_until_state_change[n] < time_until_next_event:
                    time_until_next_event = self.time_until_state_change[n]
                    next_machine = n

                    if self.machine_states[n] == self.MACHINE_UP:
                        next_event_type = self.MACHINE_FAILURE     
                    else:
                        next_event_type = self.MACHINE_REPAIR

            # Advance in time
            sim_clock += time_until_next_event

            # Execute the next event
            if next_event_type == self.PROCESS_STEP_COMPLETED:
                if next_machine == 0:
                    self.ext_buffer_level[next_machine] += 1

                elif next_machine == self.num_machines - 1:
                    self.ext_buffer_level[next_machine - 1] -= 1
                    
                else:
                    self.ext_buffer_level[next_machine - 1] -= 1
                    self.ext_buffer_level[next_machine] += 1

                # Transient phase is over, we begin to count the processed parts
                if sim_clock > 0:
                    self.parts_processed[next_machine] += 1

            elif next_event_type == self.MACHINE_FAILURE:
                self.machine_states[next_machine] = self.MACHINE_DOWN

            elif next_event_type == self.MACHINE_REPAIR:
                self.machine_states[next_machine] = self.MACHINE_UP

            # Given the new state, and USING THE MEMORYLESSNESS PROPERTY, we update
            # the times until the next events. Since it is a CTMC, we do not need an
            # event calender.
            for n in range(self.num_machines):
                # This is for machines that are neither blocked nor starved
                if self.is_prod_ready(n):
                    self.time_until_next_part[n] = rng.exponential(1/mu[n])
                    self.time_until_state_change[n] = rng.exponential(1/p[n])
                
                # This is for machines that are blocked or starved
                elif self.machine_states[n] == self.MACHINE_DOWN:
                    self.time_until_next_part[n] = np.inf
                    self.time_until_state_change[n] = rng.exponential(1/r[n])

                else:
                    self.time_until_next_part[n] = np.inf
                    self.time_until_state_change[n] = np.inf

        # Calculate throughput 
        self.th = self.parts_processed / sim_duration

        return self.th, self.parts_processed


    def is_prod_ready(self, 
                      machine_num: int):
        if self.machine_states[machine_num] == self.MACHINE_DOWN:
            return False
        
        elif machine_num == 0:
            return self.ext_buffer_level[machine_num] < self.C[machine_num] + 2
        
        elif machine_num == self.num_machines - 1:
            return self.ext_buffer_level[machine_num - 1] > 0
        
        else:
            return (
                self.ext_buffer_level[machine_num - 1] > 0
                and self.ext_buffer_level[machine_num] < self.C[machine_num] + 2
            )


if __name__ == "__main__":
    mu = np.array([ 10,   11,   12, 13,   14]) # processing rates
    p  = np.array([ 0.01, 0.01, 0.01, 0.01, 0.01]) # failure rates
    r  = np.array([ 0.1,  0.1,  0.1,  0.1,  0.1]) # repair rates
        
    # Size of buffers between machines
    C = np.array([4, 3, 2, 1])

    # Time to be simulated
    sim_duration = 10000
    seed = random.randint(0, 1000)

    # Initialize the production line
    prod_line = UnreliableProductionLine(mu=mu, 
                                         r=r, 
                                         p=p, 
                                         C=C)

    # Simulate
    th, parts_processed = prod_line.simulate(sim_duration=sim_duration, 
                                             seed=seed)
    print("Seed:", seed)
    print("Throughput:", th)
    print("Parts processed in total:", parts_processed)
