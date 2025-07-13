import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial.legendre import leggauss
from typing import NamedTuple

class theta(NamedTuple):
    v0 : float
    vbar : float
    rho : float
    kappa : float
    sigma : float

def heston_char(theta, u , T, S0, r,q):
    xi = theta.kappa - theta.sigma * theta.rho * 1j * u
    d = np.sqrt(xi**2 + theta.sigma**2*(u**2 + 1j*u))
    A1 = (u**2 + 1j * u) * np.sinh(d*T/2)
    A2 = d/ v0 * np.cosh(d*T/2) + xi/v0 * np.sinh(d * T/2)
    A = A1/A2
    D = np.log(d/v0) + (theta.kappa - d) * T/2 - np.log((d+xi)/(2*v0) + (d-xi)/(2*v0) * np.exp(-d * T))
    F = S0 * np.exp((r-q)*T)
    return np.exp(1j*u * (F/S0) - (theta.kappa * theta.vbar*theta.rho* 1j* u)/theta.sigma - A + (2*theta.kappa*theta.vbar)/(theta.sigma**2) * D)

def heston_grad_comps(theta, T,u):
    xi = theta.kappa - theta.sigma * theta.rho * 1j * u
    d = np.sqrt(xi**2 + theta.sigma**2*(u**2 + 1j*u))
    A1 = (u**2 + 1j * u) * np.sinh(d*T/2)
    A2 = d/ v0 * np.cosh(d*T/2) + xi/v0 * np.sinh(d * T/2)
    A = A1/A2
    D = np.log(d/v0) + (theta.kappa - d) * T/2 - np.log((d+xi)/(2*v0) + (d-xi)/(2*v0) * np.exp(-d * T))
    B = (d * np.exp(theta.kappa*T/2))/(v0*A2)

    d_part_rho = -(xi *theta.sigma* 1j * u)/d
    A2_part_rho = -((theta.sigma* 1j*u*(2+xi*T))/(2*d*v0))*(xi * np.cosh(d*T/2) + d*np.sinh(d*T/2))
    B_part_rho = np.exp(theta.kappa*T/2)/v0 * (1/A2 * d_part_rho - d/A2**2 * A2_part_rho)
    A1_part_rho = -(1j*u*(u**2+ 1j*u)*T*xi*theta.sigma)/(2*d) * np.cosh(d*T/2)
    A_part_rho = (1/A2) * A1_part_rho - (A/A2)* A2_part_rho
    A_part_kappa = (1j/(theta.sigma*u)) * A_part_rho
    b_part_kappa = (1j/(theta.sigma*u)) * B_part_rho + (B*T)/2
    A1_part_sigma = (((u**2 + 1j*u)*T)/2) * d_part_rho * np.cosh(d*T/2)
    A2_part_sigma = (theta.rho/theta.sigma) * A2_part_rho - ((2+T * xi)/(v0*T*xi*1j*u)) * A1_part_rho + (theta.sigma*T*A1)/(2*v0)
    A_part_sigma = (1/A2) * A1_part_sigma - (A/A2) * (A2_part_sigma)
    d_part_sigma = ((theta.rho/theta.sigma) - (1/xi) ) * d_part_rho + (theta.sigma*u**2)/d 

    h1 = -A/v0
    h2 = (2*theta.kappa)/(theta.sigma**2) * D - (theta.kappa*theta.rho* T* 1j* u)/theta.sigma
    h3= -A2_part_rho + ((2*theta.kappa*theta.vbar)/(theta.sigma**2*d)) * (d_part_rho - (d/A2) * A2_part_rho) - (theta.kappa*theta.vbar*T* 1j*u)/theta.sigma
    h4= A_part_kappa + ((2*theta.vbar)/theta.sigma**2) *D + (2*theta.kappa*theta.vbar)/(theta.sigma**2 * B) * b_part_kappa - (theta.vbar*theta.rho*T*1j *u)/theta.sigma
    h5= -(A_part_sigma) - (4*theta.kappa*theta.vbar)/theta.sigma**3 * D + (2*theta.kappa*theta.vbar)/(theta.sigma**2 * d) * (d_part_sigma - (d/A2) * A2_part_sigma) + (theta.kappa * theta.vbar * theta.rho* T* 1j * u)/theta.sigma**2

    h = np.array([h1,h2,h3,h4,h5])
    return h
def heston_gradient(theta, T,S0,r,q,K,N=64):
    
    uk,wk = leggauss(N)
    umax = min(200, 50 * np.sqrt(T))
    uk = 0.5 * umax * (uk + 1)
    wk = 0.5 * umax * wk

    phi = np.array([heston_char(theta, u , T, S0, r,q) for u in uk])
    h1 = np.array([heston_grad_comps(theta,T,u)[0] for u in uk])
    h2 = np.array([heston_grad_comps(theta,T,u)[1] for u in uk])
    h3 = np.array([heston_grad_comps(theta,T,u)[2] for u in uk])
    h4 = np.array([heston_grad_comps(theta,T,u)[3] for u in uk])
    h5 = np.array([heston_grad_comps(theta,T,u)[4] for u in uk])
    gradients = np.zeros(5, dtype=np.complex128)

    for k in range(len(uk)):
        gradients[0] = np.exp(-1j * uk[k] * np.log(K)) / (1j * uk[k]) * phi[k] *wk[k] * h1[k]
        gradients[1] = np.exp(-1j * uk[k] * np.log(K)) / (1j * uk[k]) * phi[k] *wk[k] * h2[k]
        gradients[2] = np.exp(-1j * uk[k] * np.log(K)) / (1j * uk[k]) * phi[k] *wk[k] * h3[k]
        gradients[3] = np.exp(-1j * uk[k] * np.log(K)) / (1j * uk[k]) * phi[k] *wk[k] * h4[k]
        gradients[4] = np.exp(-1j * uk[k] * np.log(K)) / (1j * uk[k]) * phi[k] *wk[k] * h5[k]
    
    return gradients


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

theta = theta(v0,vbar,rho,kappa,sigma)

print(heston_gradient(theta,T,S0,r,q,K))