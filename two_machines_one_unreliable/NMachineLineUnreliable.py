import numpy as np


def num_analysis(mu, C, sim_duration):
    # Definition of codes (numbers) for events
    process_step_completed = 1
    machine_failure = 2
    machine_repair = 3

    num_machines = len(mu)

    if len(C) != num_machines - 1:
        raise ValueError("Sizes of machine array and buffer array don't fit!")

    trans_time = sim_duration/10

    parts_processed = np.zeros(num_machines, 
                             dtype=int)
    ext_buffer_level = np.zeros(num_machines - 1, 
                                   dtype=int)
    time_until_next_part = np.zeros(num_machines, 
                                                dtype=float)

    # First machine starts with a workpiece, all other machines idle and all buffers
    # are empty
    # Initialize the random number generator with a given seed.
    rng = np.random.default_rng(4711)

    # Processing time of the first workpiece on the first machine ( offset 0 )
    time_until_next_part[0] = rng.exponential(1 / mu[0])

    # Initialize the other machines with plus infinity
    time_until_next_part[1:] = np.inf

    # Start simulation of this Markovian system
    sim_clock = -trans_time

    while sim_clock < sim_duration:
        time_until_next_event = np.inf
        next_machine = np.inf
        next_event_type = 0  # no such event exists

        for machine_number in range(num_machines):
            if isProductionReady(machine_number, 
                                 ext_buffer_level, 
                                 C, 
                                 num_machines):
                if time_until_next_part[
                    machine_number] < time_until_next_event:
                    time_until_next_event = time_until_next_part[
                                                                machine_number]
                    next_machine = machine_number  # next event at this current machine i
                    next_event_type = process_step_completed

        # Advance in time
        sim_clock += time_until_next_event

        # Execute the next event
        if next_event_type == process_step_completed:
            if next_machine == 0:
                ext_buffer_level[next_machine] += 1

            elif next_machine == num_machines - 1:
                ext_buffer_level[next_machine - 1] -= 1
                
            else:
                ext_buffer_level[next_machine - 1] -= 1
                ext_buffer_level[next_machine] += 1

            if sim_clock > 0:
                # Transient phase is over, we begin to count the processed parts
                parts_processed[next_machine] += 1

        # Given the new state, and USING THE MEMORYLESSNESS PROPERTY, we update
        # the times until the next events. Since it is a CTMC, we do not need an
        # event calender.
        for machine_number in range(num_machines):
            if isProductionReady(machine_number, 
                                 ext_buffer_level, 
                                 C, 
                                 num_machines):
                # This is for machines that are neither blocked nor starved
                time_until_next_part[
                    machine_number] = rng.exponential(1/mu[machine_number])
                
            else:
                time_until_next_part[machine_number] = np.inf

    th = parts_processed / sim_duration

    return th, parts_processed


def isProductionReady(machine_num, ext_buffer_level, C, num_machines):
    return (machine_num == 0 and ext_buffer_level[machine_num] < C[machine_num] + 2
                or 
                machine_num == num_machines - 1 and ext_buffer_level[machine_num - 1] > 0
                or 0 < machine_num < num_machines - 1
                and ext_buffer_level[machine_num - 1] > 0
                and ext_buffer_level[machine_num] < C[machine_num] + 2)



if __name__ == "__main__":
    mu = np.array([10, 8, 8]) # rate of completion of machines
    C = np.array([1000, 1000]) # Size of buffers between machines
    sim_duration = 10000 # time to be simulated

    th, parts_processed = num_analysis(mu, C, sim_duration)
    print(th)
    print(parts_processed)
