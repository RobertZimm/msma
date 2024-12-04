# -*- coding: utf-8 -*-
"""
Created on Sun Jan 15 15:50:03 2023

@author: stefa
"""

import numpy as np
import sys

# This is a quick & dirty version of a discrete-event simulation of
# a flow line with limited buffer capacities and exp. distr. processing times.
# We assume that machines are unreliable, failures are operation-dependent,
# and blocking occurs after service.

ProcessStepCompletion = 1
MachineFailure        = 2
MachineRepair         = 3

MachineStateUp        = 1
MachineStateDown      = 0



#mu = np.array([ 5, 9, 11.0, 12, 2]) # processing rates
mu = np.array([ 10,   10,   10.0, 10,   10]) # processing rates
p  = 1000*np.array([ 0.01, 0.01, 0.01, 0.01, 0.01]) # failure rates
r  = 100*np.array([ 0.1,  0.1,  0.1,  0.1,  0.1]) # repair rates

NumberOfMachines=len( mu )

C = 1000*np.array([ 1, 1, 1, 1], dtype=int)

if len( C ) != NumberOfMachines - 1:
    sys.exit("Size of array of buffer sizes does not fit.")
    
TransientTimeLength = 1000
TimeToBeSimulated = 10000
PartsProcessed = np.zeros(NumberOfMachines, dtype=int)
ExtendedBufferLevel = np.zeros(NumberOfMachines - 1, dtype=int)
TimeUntilNextWorkpieceCompletion = np.zeros( NumberOfMachines, dtype=float)


TimeUntilNextMachineFailure = np.zeros( NumberOfMachines, dtype=float)
TimeUntilNextMachineRepair  = np.zeros( NumberOfMachines, dtype=float)
CurrentStateOfMachine  = np.zeros( NumberOfMachines, dtype=int)

for i in range(NumberOfMachines ):
    TimeUntilNextWorkpieceCompletion[ i ] = np.inf
    CurrentStateOfMachine[ i ] = MachineStateUp
    
    


# First Machines starts with a first workpiece, all other machines idle and
# all buffers are initially empty, all machines are initially up

# We draw processing time of the first workpiece on the first machine

rng = np.random.default_rng( 47110 )

TimeUntilNextWorkpieceCompletion[0] = rng.exponential( 1/mu[0] )

# We draw failure time of the first machine 

TimeUntilNextMachineFailure[0] = rng.exponential( 1/p[0] )


SimClock = - TransientTimeLength

while SimClock < TimeToBeSimulated:
    TimeUntilNextEvent = np.inf
    MachineWithNextEvent = np.inf # machine numbering starts at 0
    TypeOfNextEvent = 0  # we could have different types, but none has type 0
    
        
    for i in range( NumberOfMachines ):
        if CurrentStateOfMachine[ i ] == MachineStateUp:
            if  i==0 and ExtendedBufferLevel[ i ]< C[i] + 2 or \
                i==NumberOfMachines - 1 and ExtendedBufferLevel[ i-1 ] > 0 or \
                i>0 and i<NumberOfMachines - 1 and ExtendedBufferLevel[ i-1 ] > 0 and ExtendedBufferLevel[ i ] < C[ i ] + 2:
                
                # Consider completion of a workpiece on that machine
                if TimeUntilNextWorkpieceCompletion[ i ] < TimeUntilNextEvent:
                    TimeUntilNextEvent = TimeUntilNextWorkpieceCompletion[ i ]
                    MachineWithNextEvent = i
                    TypeOfNextEvent = ProcessStepCompletion
                
                # Consider failure of that machine
                if TimeUntilNextMachineFailure[ i ] < TimeUntilNextEvent:
                    TimeUntilNextEvent = TimeUntilNextMachineFailure[ i ]
                    MachineWithNextEvent = i
                    TypeOfNextEvent = MachineFailure
        else: # as the machine is not up, it must be down, cannot be blocked or starved
            assert i==0 and ExtendedBufferLevel[ i ]< C[i] + 2 or \
                i==NumberOfMachines - 1 and ExtendedBufferLevel[ i-1 ] > 0 or \
                i>0 and i<NumberOfMachines - 1 and ExtendedBufferLevel[ i-1 ] > 0 and ExtendedBufferLevel[ i ] < C[ i ] + 2
                
            # Consider repair of that machine
            if TimeUntilNextMachineRepair[ i ] < TimeUntilNextEvent:
                TimeUntilNextEvent = TimeUntilNextMachineRepair[ i ]
                MachineWithNextEvent = i
                TypeOfNextEvent = MachineRepair
            
            
                    
        
    SimClock = SimClock + TimeUntilNextEvent
    
    if TypeOfNextEvent == ProcessStepCompletion:
        if MachineWithNextEvent == 0:
            ExtendedBufferLevel[ MachineWithNextEvent ] = ExtendedBufferLevel[ MachineWithNextEvent ] + 1
        else:
            if MachineWithNextEvent == NumberOfMachines - 1:
                ExtendedBufferLevel[ MachineWithNextEvent-1 ] = ExtendedBufferLevel[ MachineWithNextEvent-1 ] - 1
            else:
                ExtendedBufferLevel[ MachineWithNextEvent-1 ] = ExtendedBufferLevel[ MachineWithNextEvent-1 ] - 1
                ExtendedBufferLevel[ MachineWithNextEvent ] = ExtendedBufferLevel[ MachineWithNextEvent ] + 1
            
                    
        if SimClock > 0:
            # Transient phase is over, we are now logging
            PartsProcessed[ MachineWithNextEvent ] = PartsProcessed[ MachineWithNextEvent ] + 1
    
    else:
        if TypeOfNextEvent == MachineFailure:
            assert CurrentStateOfMachine[ MachineWithNextEvent ] == MachineStateUp
            CurrentStateOfMachine[ MachineWithNextEvent ] = MachineStateDown
            
        else: # must be a repair
            assert TypeOfNextEvent == MachineRepair
            assert CurrentStateOfMachine[ MachineWithNextEvent ] == MachineStateDown
            CurrentStateOfMachine[ MachineWithNextEvent ] = MachineStateUp
            
    
    
            
    # Given the new system state, and USING THE MEMORYLESSNESS PROPERTY (!), update
    # the times until the next event
    
    for i in range( NumberOfMachines ):
        if CurrentStateOfMachine[ i ] == MachineStateUp:
            # machine could complete an operation or fail
            if  i==0 and ExtendedBufferLevel[ i ]< C[i] + 2 or \
                i==NumberOfMachines - 1 and ExtendedBufferLevel[ i-1 ] > 0 or \
                i>0 and i<NumberOfMachines - 1 and ExtendedBufferLevel[ i-1 ] > 0 and ExtendedBufferLevel[ i ] < C[ i ] + 2:
                
                TimeUntilNextWorkpieceCompletion[ i ] = rng.exponential( 1/mu[ i ])
                TimeUntilNextMachineFailure[ i ] = rng.exponential( 1/p[ i ])
                TimeUntilNextMachineRepair[ i ]  = np.inf
                
            else:
                TimeUntilNextWorkpieceCompletion[ i ] = np.inf
                TimeUntilNextMachineFailure[ i ] = np.inf
                TimeUntilNextMachineRepair[ i ]  = np.inf
        else:
            assert CurrentStateOfMachine[ i ] == MachineStateDown
            assert i==0 and ExtendedBufferLevel[ i ]< C[i] + 2 or \
                i==NumberOfMachines - 1 and ExtendedBufferLevel[ i-1 ] > 0 or \
                i>0 and i<NumberOfMachines - 1 and ExtendedBufferLevel[ i-1 ] > 0 and ExtendedBufferLevel[ i ] < C[ i ] + 2
            
            TimeUntilNextWorkpieceCompletion[ i ] = np.inf
            TimeUntilNextMachineFailure[ i ] = np.inf
            
            TimeUntilNextMachineRepair[ i ] = rng.exponential( 1/r[ i ])
        

Throughput = PartsProcessed/TimeToBeSimulated

print(Throughput)