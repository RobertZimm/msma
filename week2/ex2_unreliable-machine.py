import numpy as np
from sklearn.metrics import mean_squared_error


def single_unreliable_machine(p: float, 
                              r: float, 
                              runtime: float = 1e3, 
                              init_state: str = "up", 
                              mode: str = "numerical") -> tuple:
    """
    Simulate an unreliable machine with two states using discrete-event simulation
    or compute availability analytically.

    Parameters
    ----------
    p : float
        Rate of failure when the machine is in the "up" state.
    r : float
        Rate of repair when the machine is in the "down" state.
    runtime : float
        Total time to simulate the machine.
    init_state : str, optional
        Initial state of the machine, either "up" or "down". Default is "up".
    mode : str, optional
        The mode of operation, either "numerical" for simulation or "analytical"
        for computing availability using the analytical formula. Default is "numerical".

    Returns
    -------
    tuple
        A tuple containing the fraction of time the machine is in the "up" state
        and the "down" state.
    
    Raises
    ------
    ValueError
        If mode is not 'numerical' or 'analytical'.
    """
    if mode == "numerical":
        current_state = init_state
        up_time = 0
        down_time = 0
        t = 0

        while t <= runtime:
            if current_state == "up": # p
                random_draw = np.random.exponential(1/p)
                up_time += random_draw
                t += random_draw
                current_state = "down"

            elif current_state == "down": # r
                random_draw = np.random.exponential(1/r)
                down_time += random_draw
                t += random_draw
                current_state = "up"

        return np.array([up_time/t, down_time/t])
    
    elif mode == "analytical":
        return np.array([r/(p+r), p/(p+r)])
    
    else:
        raise ValueError("mode must be 'numerical' or 'analytical'")
    

def compute_accuracy(p: float, 
                     r: float, 
                     runtime: float, 
                     init_state: str = "up") -> float:
    """
    Compute the accuracy of the numerical simulation of an unreliable machine
    by comparing it to the analytical results.

    Parameters
    ----------
    p : float
        Probability of failure.
    r : float
        Repair rate.
    runtime : float
        Simulation time.
    init_state : str, optional
        Initial state of the machine, either "up" or "down". Default is "up".

    Returns
    -------
    float
        The absolute difference between the numerical and analytical results,
        representing the accuracy of the numerical simulation.
    """
    analytical = single_unreliable_machine(p, r,  
                                           init_state=init_state, 
                                           mode="analytical")
    numerical = single_unreliable_machine(p, r, 
                                          runtime, 
                                          init_state=init_state, 
                                          mode="numerical")

    return mean_squared_error(analytical, numerical)
        

if __name__ == "__main__":
    for i in range(1, 10):
        print(compute_accuracy(0.1, 0.05, 10**i, init_state="up"))
