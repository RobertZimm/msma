import numpy as np
import scipy.integrate as integrate


def exp_pdf(T: float, 
            lam: float) -> float:
    """
    The probability density function of an exponential distribution.

    Parameters
    ----------
    T : float
        Time
    lam : float
        Rate parameter

    Returns
    -------
    float
        The probability density at time T
    """
    
    if T < 0:
        return 0
    else:
        return lam * np.exp(-lam * T)
    

def exp_cdf(T: float, 
            lam: float) -> float:
    """
    The cumulative distribution function (CDF) of an exponential distribution.

    Parameters
    ----------
    T : float
        Time
    lam : float
        Rate parameter

    Returns
    -------
    float
        The cumulative probability up to time T
    """
    if T < 0:
        return 0
    else:
        return 1 - np.exp(-lam * T)
    

def numerical_integral(a: float, 
                       b: float, 
                       lam: float, 
                       delta: float) -> float:
    """
    A numerical method to evaluate the integral of the exponential probability
    density function between a and b, with rate parameter lam and step size delta.

    Parameters
    ----------
    a : float
        Lower limit of the integral
    b : float
        Upper limit of the integral
    lam : float
        Rate parameter
    delta : float
        Step size for the numerical integration

    Returns
    -------
    float
        The numerical integral of the exponential probability density function
        between a and b
    """
    integral = 0

    for t in np.linspace(a,b, int((b-a)/delta)+1):
        integral += exp_pdf(t, lam) * delta

    return integral


def P_a_b(lam: float,
           a: float, 
           b: float, 
           mode: str = "analytical", 
           delta: float = 0.01) -> float:
    """
    Calculate the probability of the exponential distribution between two points.

    Parameters
    ----------
    lam : float
        Rate parameter of the exponential distribution.
    a : float
        Lower bound of the interval.
    b : float
        Upper bound of the interval.
    mode : str, optional
        Method to calculate the probability. Options are 'analytical', 
        'numerical', and 'python_inbuilt'. Default is 'analytical'.
    delta : float, optional
        Step size for the numerical integration, used only in 'numerical' mode. 
        Default is 0.01.

    Returns
    -------
    float
        The probability of the exponential distribution between a and b.

    Raises
    ------
    ValueError
        If a is greater than b.
    ValueError
        If lambda is not positive.
    ValueError
        If mode is not one of 'analytical', 'python_inbuilt', or 'numerical'.
    """
    if a > b:
        raise ValueError("a must be less than b")
    
    if lam <= 0:
        raise ValueError("lambda must be positive")
    
    if mode == "analytical":
        return exp_cdf(b, lam) - exp_cdf(a, lam)
    
    elif mode == "numerical":
        return numerical_integral(a, b, lam, delta=delta)
    
    elif mode == "python_inbuilt":
        return integrate.quad(exp_pdf, a, b, args=(lam))[0]

    else:
        raise ValueError("mode must be 'analytical', 'python_inbuilt' or 'numerical'")
    

def compare_methods(lam: float, 
                    a: float, 
                    b: float) -> None:
    """
    Compare the analytical, numerical, and python inbuilt methods for calculating
    the probability of the exponential distribution between two points.

    Parameters
    ----------
    lam : float
        Rate parameter of the exponential distribution.
    a : float
        Lower bound of the interval.
    b : float
        Upper bound of the interval.

    Prints
    ------
    Comparison of the results from the analytical, python inbuilt, and numerical
    methods for different delta values, along with the associated errors.
    """
    analytical = P_a_b(lam, a, b, mode="analytical")
    inbuilt = P_a_b(lam, a, b, mode="python_inbuilt")
    error_inbuilt = np.abs(analytical - inbuilt)

    print("--- inbuilt and analytical ---")
    print(f"Analytical: {analytical}")
    print(f"Python inbuilt: {inbuilt}")
    print(f"Error: {error_inbuilt}\n")  

    print("calculate error between numerical and analytical for different delta values:\n")
    for i in range(1, 9):
        delta = 10**(-i)
        numerical = P_a_b(lam, a, b, mode="numerical", delta=delta)    
        error = np.abs(analytical - numerical)

        print(f"--- delta = {delta} ---")
        print(f"Numerical: {numerical}")
        print(f"Error: {error}\n")


if __name__ == "__main__":
    # Parameters
    a = 0
    b = 1
    lam = 1

    # Compare the analytical, python inbuilt, and numerical methods
    compare_methods(lam, a, b)
