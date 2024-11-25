import numpy as np


# Definition of codes (numbers) for events
ProcessStepCompletion = 1
MachineFailure = 2
MachineRepair = 3

mu = np.array([10, 8])  # mu_1 = is 10, so is mu_2
C = np.array([10000])  # size of the buffer(s)

NumberOfMachines = len(mu)

if len(C) != NumberOfMachines - 1:
    raise ValueError("Sizes of machine array and buffer array don't fit!")

# Alternative way to test this

assert len(C) == NumberOfMachines - 1

TimeToBeSimulated = 100000
TransientTimeLength = 10000

PartProcessed = np.zeros(NumberOfMachines, dtype=int)
ExtendedBufferLevel = np.zeros(NumberOfMachines - 1, dtype=int)
TimeUntilNextWorkpieceCompletion = np.zeros(NumberOfMachines, dtype=float)

# First machine starts with a workpiece, all other machines idle and all buffers
# are empty

# Initialize the random number generator with a given seed.
rng = np.random.default_rng(4711)

# Processing time of the first workpiece on the first machine ( offset 0 )
TimeUntilNextWorkpieceCompletion[0] = rng.exponential(1 / mu[0])

# Initialize the other machines with plus infinity
for i in range(1, NumberOfMachines):
    TimeUntilNextWorkpieceCompletion[i] = np.inf

# Start simulation of this Markovian (!!) system
SimClock = -TransientTimeLength

while SimClock < TimeToBeSimulated:
    TimeUntilNextEvent = np.inf
    MachineWithNextEvent = np.inf
    TypeOfNextEvent = 0  # no such event exists
    for i in range(NumberOfMachines):
        if (
            i == 0
            and ExtendedBufferLevel[i] < C[i] + 2
            or i == NumberOfMachines - 1
            and ExtendedBufferLevel[i - 1] > 0
            or i > 0
            and i < NumberOfMachines - 1
            and ExtendedBufferLevel[i - 1] > 0
            and ExtendedBufferLevel[i] < C[i] + 2
        ):
            if TimeUntilNextWorkpieceCompletion[i] < TimeUntilNextEvent:
                TimeUntilNextEvent = TimeUntilNextWorkpieceCompletion[i]
                MachineWithNextEvent = i  # next event at this current machine i
                TypeOfNextEvent = ProcessStepCompletion

    # Advance in time
    SimClock = SimClock + TimeUntilNextEvent

    # Execute the next event
    if TypeOfNextEvent == ProcessStepCompletion:
        if MachineWithNextEvent == 0:
            ExtendedBufferLevel[MachineWithNextEvent] = (
                ExtendedBufferLevel[MachineWithNextEvent] + 1
            )
        else:
            if MachineWithNextEvent == NumberOfMachines - 1:
                ExtendedBufferLevel[MachineWithNextEvent - 1] -= 1
            else:
                ExtendedBufferLevel[MachineWithNextEvent - 1] -= 1
                ExtendedBufferLevel[MachineWithNextEvent] += 1

        if SimClock > 0:
            # Transient phase is over, we begin to count the processed parts
            PartProcessed[MachineWithNextEvent] += 1

    # Given the new state, and USING THE MEMORYLESSNESS PROPERTY, we update
    # the times until the next events. Since it is a CTMC, we do not need an
    # event calender.
    for i in range(NumberOfMachines):
        if (
            i == 0
            and ExtendedBufferLevel[i] < C[i] + 2
            or i == NumberOfMachines - 1
            and ExtendedBufferLevel[i - 1] > 0
            or i > 0
            and i < NumberOfMachines - 1
            and ExtendedBufferLevel[i - 1] > 0
            and ExtendedBufferLevel[i] < C[i] + 2
        ):
            # This is for machines that are neither blocked nor starved
            TimeUntilNextWorkpieceCompletion[i] = rng.exponential(1 / mu[i])
        else:
            TimeUntilNextWorkpieceCompletion[i] = np.inf

Throughput = PartProcessed / TimeToBeSimulated

print(Throughput)
print(PartProcessed)
