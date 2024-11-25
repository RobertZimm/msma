import numpy as np



# Definition of codes (numbers) for events
PROCESS_STEP_COMPLETED = 1
MACHINE_FAILURE = 2
MACHINE_REPAIR = 3


def num_analysis(mu, r, p, C, sim_duration):
    # Initialize the random number generator with a given seed.
    rng = np.random.default_rng(4711)

    num_machines = len(mu)

    if len(C) != num_machines - 1:
        raise ValueError("Sizes of machine array and buffer array don't fit!")

    trans_time = sim_duration/10

    # Initialize the state variables
    parts_processed = np.zeros(num_machines, 
                             dtype=int)
    ext_buffer_level = np.zeros(num_machines - 1, 
                                dtype=int)
    time_until_next_part = np.zeros(num_machines, 
                                    dtype=float)
    
    machine_state = np.ones(num_machines, 
                             dtype=int)
    
    # If machine up: time until failure
    # If machine down: time until repair
    time_until_state_change = np.zeros(num_machines, 
                                 dtype=float)

    # Processing time of the first workpiece on the first machine ( offset 0 )
    time_until_next_part[0] = rng.exponential(1/mu[0])

    # Initialize the other machines with plus infinity
    time_until_next_part[1:] = np.inf

    for i in range(num_machines):
        time_until_state_change[i] = rng.exponential(1/p[i])

    # Start simulation of this Markovian system
    sim_clock = -trans_time

    while sim_clock < sim_duration:
        time_until_next_event = np.inf
        next_machine = np.inf

        # Choose initial event which does not exist
        next_event_type = 0

        # Get production ready machine 
        # with smallest time until next part
        for n in range(num_machines):
            if (is_prod_ready(n, ext_buffer_level, 
                             C, num_machines,
                             machine_state)
                and time_until_next_part[n] < time_until_next_event):
                time_until_next_event = time_until_next_part[n]
                next_machine = n 
                next_event_type = PROCESS_STEP_COMPLETED

            if time_until_next_part[n] < time_until_next_event:
                time_until_next_event = time_until_state_change[n]
                next_machine = n

                if machine_state[n] == 1:
                    next_event_type = MACHINE_FAILURE     
                else:
                    next_event_type = MACHINE_REPAIR

        # Advance in time
        sim_clock += time_until_next_event

        # Execute the next event
        if next_event_type == PROCESS_STEP_COMPLETED:
            if next_machine == 0:
                ext_buffer_level[next_machine] += 1

            elif next_machine == num_machines - 1:
                ext_buffer_level[next_machine - 1] -= 1
                
            else:
                ext_buffer_level[next_machine - 1] -= 1
                ext_buffer_level[next_machine] += 1

            # Transient phase is over, we begin to count the processed parts
            if sim_clock > 0:
                parts_processed[next_machine] += 1

        elif next_event_type == MACHINE_FAILURE:
            machine_state[next_machine] = 0

        elif next_event_type == MACHINE_REPAIR:
            machine_state[next_machine] = 1

        # Given the new state, and USING THE MEMORYLESSNESS PROPERTY, we update
        # the times until the next events. Since it is a CTMC, we do not need an
        # event calender.
        for n in range(num_machines):
            # This is for machines that are neither blocked nor starved
            if is_prod_ready(n, ext_buffer_level, 
                             C, num_machines,
                             machine_state):
                time_until_next_part[n] = rng.exponential(1/mu[n])
            
            # This is for machines that are blocked or starved
            else:
                time_until_next_part[n] = np.inf

            if machine_state[n] == 1:
                time_until_state_change[n] = rng.exponential(1/p[n])
            else:
                time_until_state_change[n] = rng.exponential(1/r[n])

    # Calculate throughput 
    th = parts_processed / sim_duration

    return th, parts_processed


def is_prod_ready(machine_num, ext_buffer_level, C, num_machines, machine_state):
    if machine_state[machine_num] == 0:
        return False
    
    return (machine_num == 0 and ext_buffer_level[machine_num] < C[machine_num] + 2
                or 
                machine_num == num_machines - 1 and ext_buffer_level[machine_num - 1] > 0
                or 0 < machine_num < num_machines - 1
                and ext_buffer_level[machine_num - 1] > 0
                and ext_buffer_level[machine_num] < C[machine_num] + 2)


if __name__ == "__main__":
    r = np.array([0.3, 0.3, 0.3]) # rate of repair
    p = np.array([0.1, 0.1, 0.1]) # rate of failure
    mu = np.array([10, 8, 8]) # rate of completion of machines
    C = np.array([1000, 1000]) # Size of buffers between machines
    sim_duration = 10000 # time to be simulated

    th, parts_processed = num_analysis(mu, r, p, C, sim_duration)
    print(th)
    print(parts_processed)
