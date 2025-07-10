import numpy as np

def heston_characteristic_function(u, S0, r, T, kappa, theta, sigma, rho, v0):
    """Heston characteristic function implementation."""
    xi = kappa - rho * sigma * 1j * u
    d = np.sqrt((rho * sigma * 1j * u - xi)**2 - sigma**2 * (-u * 1j - u**2))
    g = (xi - rho * sigma * 1j * u - d) / (xi - rho * sigma * 1j * u + d)
    C = r * 1j * u * T + (kappa * theta)/sigma**2 * (
        (xi - rho*sigma*1j*u - d)*T - 2*np.log((1 - g*np.exp(-d*T))/(1 - g)))
    D = (xi - rho*sigma*1j*u - d)/sigma**2 * ((1 - np.exp(-d*T))/(1 - g*np.exp(-d*T)))
    return np.exp(C + D*v0 + 1j*u*np.log(S0))