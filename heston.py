import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

def heston_char(s, St, K, r, T, sigma, kappa, theta, v0, rho):
    prod = rho * sigma * 1j * s

    d1 = (prod - kappa)**2
    d2 = (sigma**2) * (1j * s + s**2)
    d = np.sqrt(d1 + d2)

    g1 = kappa - prod - d
    g2 = kappa - prod + d
    g = g1 / g2

    exp1 = np.exp(np.log(St) * 1j * s) * np.exp(1j * s * r * T)
    exp2 = 1 - g * np.exp(-d * T)
    exp3 = 1 - g
    mainExp1 = exp1 * np.power(exp2 / exp3, -2 * theta * kappa / (sigma**2))

    exp4 = theta * kappa * T / (sigma**2)
    exp5 = v0 / (sigma**2)
    exp6 = (1 - np.exp(-d * T)) / (1 - g * np.exp(-d * T))
    mainExp2 = np.exp((exp4 * g1) + (exp5 * g1 * exp6))

    return (mainExp1 * mainExp2)

def heston_call_price(St, K, r, T, sigma, kappa, theta, v0, rho):
    P, iterations, maxNumber = 0, 1000, 100
    ds = maxNumber / iterations

    element1 = 0.5 * (St - K * np.exp(-r * T))

    for j in range(1, iterations):
        s1 = ds * (2 * j + 1) / 2
        s2 = s1 - 1j

        numerator1 = heston_char(s2, St, K, r, T, sigma, kappa, theta, v0, rho)
        numerator2 = K * heston_char(s1, St, K, r, T, sigma, kappa, theta, v0,
                                 rho)
        denominator = np.exp(np.log(K) * 1j * s1) * 1j * s1

        P += ds * (numerator1 - numerator2) / denominator

    element2 = P / np.pi

    return np.real((element1 + element2))

def heston_put_price(St, K, r, T, sigma, kappa, theta, v0, rho):
    call = heston_call_price(St, K, r, T, sigma, kappa, theta, v0, rho)
    return call + K * np.exp(-r * T) - St
   

S0 = 100
r = 0.01
T = 1.0
kappa = 2.0
theta = 0.01
sigma = 0.2
rho = -0.7
v0 = 0.01

K_values = np.linspace(90, 110, 100)
call_prices = [heston_call_price(S0, K, r, T, sigma, kappa, theta, v0, rho) for K in K_values]
put_prices = [heston_put_price(S0, K, r, T, sigma, kappa, theta, v0, rho) for K in K_values]

plt.plot(K_values, call_prices, label= 'Call Price')
plt.plot(K_values, put_prices, label='Put Price') 
plt.xlabel('Strike Price K')
plt.ylabel('Price')
plt.title('Heston Call/put Option Price vs Strike')
plt.grid(True)
plt.legend()
plt.show()