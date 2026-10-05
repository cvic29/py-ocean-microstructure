'''
CV 2025/02/19 : compute viscosity of seawater 
'''
import numpy as np

def visc35(T):
    """
    Compute the kinematic viscosity of seawater for salinity S = 35 PSU.

    Parameters:
    - T (float or np.ndarray): Temperature in degrees Celsius.

    Returns:
    - v (float or np.ndarray): Kinematic viscosity in meters squared per second (m²/s).
    """
    # Polynomial coefficients from the MATLAB function
    pol = np.array([
        -1.131311019739306e-11,
         1.199552027472192e-09,
        -5.864346822839289e-08,
         1.828297985908266e-06
    ])
    
    # Compute viscosity using the polynomial
    v = np.polyval(pol, T)
    
    return v
 
