import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from numpy.polynomial.legendre import leggauss

def heston_char(v0, vbar, rho, kappa, sigma, u , T, S0, r,q):
    xi = kappa - sigma * rho * 1j * u
    d = np.sqrt(xi**2 + sigma**2*(u**2 + 1j*u))
    A1 = (u**2 + 1j * u) * np.sinh(d*T/2)
    A2 = d/ v0 * np.cosh(d*T/2) + xi/v0 * np.sinh(d * T/2)
    A = A1/A2
    D = np.log(d/v0) + (kappa - d) * T/2 - np.log((d+xi)/(2*v0) + (d-xi)/(2*v0) * np.exp(-d * T))
    F = S0 * np.exp((r-q)*T)
    return np.exp(1j*u * (F/S0) - (kappa * vbar*rho* 1j* u)/sigma - A + (2*kappa*vbar)/(sigma**2) * D)

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

def heston_grad_comps(v0,vbar,rho,kappa,sigma, T,S0,r,q,u):
    xi = kappa - sigma * rho * 1j * u
    d = np.sqrt(xi**2 + sigma**2*(u**2 + 1j*u))
    A1 = (u**2 + 1j * u) * np.sinh(d*T/2)
    A2 = d/ v0 * np.cosh(d*T/2) + xi/v0 * np.sinh(d * T/2)
    A = A1/A2
    D = np.log(d/v0) + (kappa - d) * T/2 - np.log((d+xi)/(2*v0) + (d-xi)/(2*v0) * np.exp(-d * T))
    B = (d * np.exp(kappa*T/2))/(v0*A2)

    d_part_rho = -(xi *sigma* 1j * u)/d
    A2_part_rho = -((sigma* 1j*u*(2+xi*T))/(2*d*v0))*(xi * np.cosh(d*T/2) + d*np.sinh(d*T/2))
    B_part_rho = np.exp(kappa*T/2)/v0 * (1/A2 * d_part_rho - d/A2**2 * A2_part_rho)
    A1_part_rho = -(1j*u*(u**2+ 1j*u)*T*xi*sigma)/(2*d) * np.cosh(d*T/2)
    A_part_rho = (1/A2) * A1_part_rho - (A/A2)* A2_part_rho
    A_part_kappa = (1j/(sigma*u)) * A_part_rho
    b_part_kappa = (1j/(sigma*u)) * B_part_rho + (B*T)/2
    A1_part_sigma = (((u**2 + 1j*u)*T)/2) * d_part_rho * np.cosh(d*T/2)
    A2_part_sigma = (rho/sigma) * A2_part_rho - ((2+T * xi)/(v0*T*xi*1j*u)) * A1_part_rho + (sigma*T*A1)/(2*v0)
    A_part_sigma = (1/A2) * A1_part_sigma - (A/A2) * (A2_part_sigma)
    d_part_sigma = ((rho/sigma) - (1/xi) ) * d_part_rho + (sigma*u**2)/d 

    h1 = -A/v0
    h2 = (2*kappa)/(sigma**2) * D - (kappa*rho* T* 1j* u)/sigma
    h3= -A2_part_rho + ((2*kappa*vbar)/(sigma**2*d)) * (d_part_rho - (d/A2) * A2_part_rho) - (kappa*vbar*T* 1j*u)/sigma
    h4= A_part_kappa + ((2*vbar)/sigma**2) *D + (2*kappa*vbar)/(sigma**2 * B) * b_part_kappa - (vbar*rho*T*1j *u)/sigma
    h5= -(A_part_sigma) - (4*kappa*vbar)/sigma**3 * D + (2*kappa*vbar)/(sigma**2 * d) * (d_part_sigma - (d/A2) * A2_part_sigma) + (kappa * vbar * rho* T* 1j * u)/sigma**2

    h = np.array([h1,h2,h3,h4,h5])
    return h
def heston_gradient(v0,vbar,rho,kappa,sigma, T,S0,r,q,K,N=64):
    
    uk,wk = leggauss(N)
    umax = min(200, 50 * np.sqrt(T))
    uk = 0.5 * umax * (uk + 1)
    wk = 0.5 * umax * wk

    phi = np.array([heston_char(v0, vbar, rho, kappa, sigma, u , T, S0, r,q) for u in uk])
    h1 = np.array([heston_grad_comps(v0,vbar,rho,kappa,sigma,T,S0,r,q,u)[0] for u in uk])
    h2 = np.array([heston_grad_comps(v0,vbar,rho,kappa,sigma,T,S0,r,q,u)[1] for u in uk])
    h3 = np.array([heston_grad_comps(v0,vbar,rho,kappa,sigma,T,S0,r,q,u)[2] for u in uk])
    h4 = np.array([heston_grad_comps(v0,vbar,rho,kappa,sigma,T,S0,r,q,u)[3] for u in uk])
    h5 = np.array([heston_grad_comps(v0,vbar,rho,kappa,sigma,T,S0,r,q,u)[4] for u in uk])
    gradients = np.zeros(5, dtype=np.complex128)

    for k in range(len(uk)):
        gradients[0] = np.exp(-1j * uk[k] * np.log(K)) / (1j * uk[k]) * phi[k] *wk[k] * h1[k]
        gradients[1] = np.exp(-1j * uk[k] * np.log(K)) / (1j * uk[k]) * phi[k] *wk[k] * h2[k]
        gradients[2] = np.exp(-1j * uk[k] * np.log(K)) / (1j * uk[k]) * phi[k] *wk[k] * h3[k]
        gradients[3] = np.exp(-1j * uk[k] * np.log(K)) / (1j * uk[k]) * phi[k] *wk[k] * h4[k]
        gradients[4] = np.exp(-1j * uk[k] * np.log(K)) / (1j * uk[k]) * phi[k] *wk[k] * h5[k]
    
    return gradients

def heston_put_price(St, K, r, T, sigma, kappa, theta, v0, rho):
    call = heston_call_price(St, K, r, T, sigma, kappa, theta, v0, rho)
    return call + K * np.exp(-r * T) - St
   


S0 = 100
r = 0.04
q = 0.03
T = 1.0
kappa = 2.0
vbar = 0.01
sigma = 0.2
rho = -0.7
v0 = 0.01
u = 5
K = 105

print(heston_gradient(v0,vbar,rho,kappa,sigma,T,S0,r,q,K))