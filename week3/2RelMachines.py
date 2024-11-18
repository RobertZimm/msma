import numpy as np


def two_rel_machines(mu_1: float, mu_2: float, C: int, runtime: int):
    """
    Simulate a two-machine reliable system with exponential distributions for time between failures and time to repair.

    Parameters
    ----------
    mu_1 : float
        The rate parameter of the exponential distribution for the time between failures for the first machine.
    mu_2 : float
        The rate parameter of the exponential distribution for the time between failures for the second machine.
    C : int
        The capacity of the buffer.
    runtime : int
        The total time to run the simulation.

    Returns
    -------
    n_bar : float
        The average buffer size.
    pi_hat : list
        The estimated stationary distribution of the buffer size.
    TH_i : list
        The estimated throughput of the first and second machines.
    """
    state_times = [0 for _ in range(C+3)]
    t = 0
    current_state = 0

    while t < runtime:
        if current_state == 0:
            draw_T1 = np.random.exponential(1/mu_1)
            t += draw_T1
            state_times[current_state] += draw_T1
            current_state += 1
        
        elif current_state == C+2:
            draw_T2 = np.random.exponential(1/mu_2)
            t += draw_T2
            state_times[current_state] += draw_T2
            current_state -= 1

        else:
            draw_T1 = np.random.exponential(1/mu_1)
            draw_T2 = np.random.exponential(1/mu_2)

            if draw_T1 < draw_T2:
                t += draw_T1
                state_times[current_state] += draw_T1
                current_state += 1

            else:
                t += draw_T2
                state_times[current_state] += draw_T2
                current_state -= 1

    pi_hat = [0 for _ in range(len(state_times))]
    for i in range(len(state_times)):
        pi_hat[i] = state_times[i] / t

    n_bar = sum([i * state_times[i] for i in range(len(state_times))])
    TH_i = [calc_TH1(C, mu_1, mu_2, pi_hat), calc_TH2(C, mu_2, pi_hat)]

    return n_bar, pi_hat, TH_i


def calc_Q(C: int, mu_1: float, mu_2: float):
    """
    Calculate the transition rate matrix Q for the two-machine system.

    Parameters
    ----------
    C : int
        The capacity of the buffer.
    mu_1 : float
        The rate of completion of the first machine.
    mu_2 : float
        The rate of completion of the second machine.

    Returns
    -------
    Q : numpy.ndarray
        The transition rate matrix Q.
    """
    Q = np.zeros((C+3, C+3))

    Q[0, 0] = -mu_1
    Q[0, 1] = mu_1

    Q[C+2, C+2] = -mu_2
    Q[C+2, C+1] = mu_2

    for i in range(1, C+2):
        Q[i, i-1] = mu_1
        Q[i, i] = -mu_1 - mu_2
        Q[i, i+1] = mu_2

    print(Q)

    return Q


def test_Q_sing(C: int, mu_1: float, mu_2: float, eps: float = 1e-10):
    """
    Check if the transition rate matrix Q is singular.

    Parameters
    ----------
    C : int
        The capacity of the buffer.
    mu_1 : float
        The rate of completion of the first machine.
    mu_2 : float
        The rate of completion of the second machine.
    eps : float, optional
        The tolerance for singularity. Default is 1e-10.

    Returns
    -------
    bool
        If the matrix Q is singular, return True, else False.
    """
    Q = calc_Q(C, mu_1, mu_2)

    if np.abs(np.linalg.det(Q)) < eps:
        print("Q is singular. System is underdetermined. ")
        return True

    else:
        print("Q is not singular")
        return False


def calc_pi(C: int, mu_1: float, mu_2: float):
    """
    Calculate the stationary distribution of a two-machine system with a buffer of size C.

    Parameters
    ----------
    C : int
        The capacity of the buffer.
    mu_1 : float
        The rate of completion of the first machine.
    mu_2 : float
        The rate of completion of the second machine.

    Returns
    -------
    pi : numpy.ndarray
        The stationary distribution of the system.
    """
    # If mu_1 != mu_2 the geometric series converges
    if mu_1 != mu_2:
        pi = np.zeros(C+3)
        pi[0] = (1-(mu_1/mu_2))/(1-(mu_1/mu_2)**(C+3))

        for i in range(1, C+3):
            pi[i] = (mu_1/mu_2)**i * pi[0]

    # If mu_1 == mu_2 the geometric series diverges; pi_i = 1/(C+3); equally distributed
    else:
        pi = (1/(C+3)) * np.ones(C+3)

    return pi


def calc_n_bar(pi: list):
    """
    Calculate the average buffer size of a two-machine system given its stationary distribution.

    Parameters
    ----------
    pi : list
        The stationary distribution of the system.

    Returns
    -------
    n_bar : float
        The average buffer size of the system.
    """
    return sum([i * pi[i] for i in range(C+3)])


def calc_TH1(C: int, mu_1: float, mu_2: float, pi: list):
    """
    Calculate the throughput of the first machine of a two-machine system given its stationary distribution and buffer size.

    Parameters
    ----------
    C : int
        The capacity of the buffer.
    mu_1 : float
        The rate of completion of the first machine.
    mu_2 : float
        The rate of completion of the second machine.
    pi : list
        The stationary distribution of the system.

    Returns
    -------
    TH_1 : float
        The throughput of the first machine.
    """
    return float(mu_1 * sum(pi[:C+1]) + mu_2 * pi[C+2])


def calc_TH2(C: int, mu_2: float, pi: list):
    """
    Calculate the throughput of the second machine of a two-machine system given its stationary distribution and buffer size.

    Parameters
    ----------
    C : int
        The capacity of the buffer.
    mu_1 : float
        The rate of completion of the first machine.
    pi : list
        The stationary distribution of the system.

    Returns
    -------
    TH_2 : float
        The throughput of the second machine.
    """
    return float(mu_2 * sum(pi[1:C+3]))


if __name__ == "__main__":
    C = 5
    mu_1 = 0.45
    mu_2 = 0.45

    # test_Q_sing(C, mu_1, mu_2)

    pi = calc_pi(C, mu_1, mu_2)
    n_bar = calc_n_bar(pi)
    TH_i = [calc_TH1(C, mu_1, mu_2, pi), calc_TH2(C, mu_1, pi)]

    n_bar_exp, pi_exp, TH_i_exp = two_rel_machines(mu_1, mu_2, C, 1e5)
    
    print(pi, pi_exp)
    print(TH_i, TH_i_exp)
