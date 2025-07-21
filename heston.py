import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial.legendre import leggauss
from scipy.integrate import quad
from scipy.optimize import minimize, least_squares
from typing import NamedTuple, List, Tuple
import time
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

class HestonParameters(NamedTuple):
    v0: float     # Initial variance
    vbar: float   # Long-run variance
    rho: float    # Correlation between price and variance
    kappa: float  # Mean reversion speed
    sigma: float  # Volatility of volatility

class HestonModel:
    def __init__(self, integration_nodes: int = 32, max_integration_bound: float = 50):
        self.N = integration_nodes
        self.max_u = max_integration_bound
        self.bounds_lower = np.array([0.001, 0.001, -0.99, 0.01, 0.01])
        self.bounds_upper = np.array([5.0, 5.0, 0.99, 20.0, 5.0])
        self.eps = 1e-12
        self.max_exp_arg = 50
        
        # Pre-compute Gauss-Legendre nodes and weights for reuse
        self._setup_integration_grid()
    
    def _safe_exp(self, x):
        x_clipped = np.clip(np.real(x), -self.max_exp_arg, self.max_exp_arg)
        return np.exp(x_clipped + 1j * np.imag(x))
    
    def _safe_log(self, x):
        return np.log(np.maximum(np.abs(x), self.eps)) + 1j * np.angle(x)
    
    def _setup_integration_grid(self):
        self.integration_grids = {}
        for T in [0.25, 0.5, 1.0, 2.0, 5.0]:
            uk, wk = leggauss(self.N)
            umax = min(self.max_u, 30 * np.sqrt(T)) 
            uk = 0.5 * umax * (uk + 1)
            wk = 0.5 * umax * wk
            mask = uk > 1e-6
            self.integration_grids[T] = (uk[mask], wk[mask])
    
    def _get_integration_grid(self, T):
        best_T = min(self.integration_grids.keys(), key=lambda x: abs(x - T))
        if abs(best_T - T) < 0.1: 
            return self.integration_grids[best_T]
        else:
            uk, wk = leggauss(self.N)
            umax = min(self.max_u, 30 * np.sqrt(T))
            uk = 0.5 * umax * (uk + 1)
            wk = 0.5 * umax * wk
            mask = uk > 1e-6
            return uk[mask], wk[mask]
    
    def characteristic_function(self, params: HestonParameters, u: complex, T: float, 
                              S0: float, r: float, q: float) -> complex:
        if np.abs(u) < self.eps:
            return 1.0 + 0j
        v0 = max(params.v0, self.eps)
        sigma = max(params.sigma, self.eps)
        xi = params.kappa - sigma * params.rho * 1j * u
        d_squared = xi**2 + sigma**2 * (u**2 + 1j*u)
        d = np.sqrt(d_squared)
        if np.real(d) < 0:
            d = -d
        F = S0 * np.exp((r - q) * T)
        g2 = (xi - d) / (xi + d)
        dT = np.clip(d * T, -self.max_exp_arg, self.max_exp_arg)
        term1 = 1j * u * np.log(F / S0)
        term2 = (params.kappa * params.vbar / sigma**2) * (
            (xi - d) * T - 2 * np.log((1 - g2 * np.exp(-dT)) / (1 - g2))
        )
        term3 = (v0 / sigma**2) * (xi - d) * (1 - np.exp(-dT)) / (1 - g2 * np.exp(-dT))
        exponent = term1 + term2 + term3
        if np.abs(np.real(exponent)) > self.max_exp_arg:
            return 1e-10 + 0j
        return np.exp(exponent)
    
    def option_price(self, params: HestonParameters, T: float, S0: float, r: float, 
                    q: float, K: float, option_type: str = 'call') -> float:
        uk, wk = self._get_integration_grid(T)
        log_K_S0 = np.log(K / S0)
        try:
            phi_u_minus_i = np.array([self.characteristic_function(params, u - 1j, T, S0, r, q) for u in uk])
            phi_u = np.array([self.characteristic_function(params, u, T, S0, r, q) for u in uk])
            valid_mask = np.isfinite(phi_u_minus_i) & np.isfinite(phi_u)

            if not np.any(valid_mask):
                return max(S0 - K, 0) if option_type.lower() == 'call' else max(K - S0, 0)
            exp_factors = np.exp(-1j * uk * log_K_S0) / (1j * uk)
            

            integrand1 = np.real(exp_factors * phi_u_minus_i)
            integrand2 = np.real(exp_factors * phi_u)
            integrand1 = np.where(valid_mask, integrand1, 0)
            integrand2 = np.where(valid_mask, integrand2, 0)
            integral1 = np.sum(integrand1 * wk)
            integral2 = np.sum(integrand2 * wk)
            
            P1 = 0.5 + integral1 / np.pi
            P2 = 0.5 + integral2 / np.pi
            P1 = np.clip(P1, 0, 1)
            P2 = np.clip(P2, 0, 1)
            
            call_price = S0 * np.exp(-q * T) * P1 - K * np.exp(-r * T) * P2
            call_price = max(call_price, 0)
            
            if option_type.lower() == 'put':
                put_price = call_price - S0 * np.exp(-q * T) + K * np.exp(-r * T)
                return max(put_price, 0)
            return call_price
            
        except Exception:
            return max(S0 - K, 0) if option_type.lower() == 'call' else max(K - S0, 0)
    
    def _finite_difference_gradient(self, params_array: np.ndarray, T: float, S0: float, 
                                  r: float, q: float, K: float, h: float = 1e-4) -> np.ndarray:
        gradient = np.zeros(5)
        base_params = HestonParameters(*params_array)
        base_price = self.option_price(base_params, T, S0, r, q, K)
        step_sizes = np.array([1e-5, 1e-5, 1e-4, 1e-4, 1e-4])
        
        for i in range(5):
            step = step_sizes[i]
            params_plus = params_array.copy()
            params_plus[i] += step
            params_plus = np.clip(params_plus, self.bounds_lower, self.bounds_upper)
            params_obj_plus = HestonParameters(*params_plus)
            price_plus = self.option_price(params_obj_plus, T, S0, r, q, K)
            if np.isfinite(price_plus) and np.isfinite(base_price):
                gradient[i] = (price_plus - base_price) / step
            else:
                gradient[i] = 0
        return gradient
    
    def _project_to_bounds(self, params: np.ndarray) -> np.ndarray:
        return np.clip(params, self.bounds_lower, self.bounds_upper)
    
    def calibrate_lm(self, market_data: List[Tuple[float, float]], T: float, S0: float, 
                    r: float, q: float, initial_guess: HestonParameters = None,
                    max_iter: int = 50, tolerance: float = 1e-6, 
                    verbose: bool = True) -> Tuple[HestonParameters, dict]:
        if initial_guess is None:
            initial_guess = HestonParameters(0.04, 0.04, -0.5, 1.0, 0.2)
        
        strikes = np.array([data[0] for data in market_data])
        market_prices = np.array([data[1] for data in market_data])
        
        x = np.array([initial_guess.v0, initial_guess.vbar, initial_guess.rho, 
                      initial_guess.kappa, initial_guess.sigma])
        x = self._project_to_bounds(x)
        
        def residual_function(params_array):
            params = HestonParameters(*params_array)
            model_prices = np.array([self.option_price(params, T, S0, r, q, k) 
                                   for k in strikes])
            return model_prices - market_prices
        
        def jacobian_function(params_array):
            J = np.zeros((len(strikes), 5))
            for i, k in enumerate(strikes):
                J[i, :] = self._finite_difference_gradient(params_array, T, S0, r, q, k)
            return J
        
        residuals = residual_function(x)
        norm_r = np.linalg.norm(residuals)
        
        if verbose:
            print(f"Initial RMSE: {norm_r/np.sqrt(len(strikes)):.6e}")

        mu = 1e-3
        nu = 2
        
        info = {
            'initial_rmse': norm_r/np.sqrt(len(strikes)),
            'iterations': 0,
            'success': False,
            'final_rmse': None,
            'convergence_history': []
        }
        
        for iteration in range(max_iter):
            try:
                J = jacobian_function(x)
                JTJ = J.T @ J
                JTr = J.T @ residuals
                
                A = JTJ + mu * np.eye(len(x))
                delta = np.linalg.solve(A, -JTr)
                
                max_step = 0.5
                if np.linalg.norm(delta) > max_step:
                    delta = delta * max_step / np.linalg.norm(delta)
                
                x_new = self._project_to_bounds(x + delta)
                residuals_new = residual_function(x_new)
                norm_r_new = np.linalg.norm(residuals_new)
                
                if norm_r_new < norm_r:
                    x = x_new
                    residuals = residuals_new
                    norm_r = norm_r_new
                    mu = mu / 3
                    
                    rmse = norm_r / np.sqrt(len(strikes))
                    info['convergence_history'].append(rmse)
                    
                    if verbose:
                        print(f"Iteration {iteration+1}: RMSE = {rmse:.6e}")
                    
                    if rmse < tolerance:
                        info['success'] = True
                        if verbose:
                            print(f"Converged in {iteration+1} iterations")
                        break
                else:
                    mu = mu * nu
                    nu = min(2 * nu, 16)
                
            except Exception as e:
                if verbose:
                    print(f"Error at iteration {iteration}: {e}")
                break
            
            info['iterations'] = iteration + 1
        
        info['final_rmse'] = norm_r / np.sqrt(len(strikes))
        final_params = HestonParameters(*x)
        
        return final_params, info
    
    def calibrate_scipy(self, market_data: List[Tuple[float, float]], T: float, S0: float, 
                       r: float, q: float, initial_guess: HestonParameters = None,
                       method: str = 'least_squares', verbose: bool = True) -> Tuple[HestonParameters, dict]:
        """
        Fast calibration using scipy optimization
        """
        if initial_guess is None:
            initial_guess = HestonParameters(0.04, 0.04, -0.5, 1.0, 0.2)
        
        strikes = np.array([data[0] for data in market_data])
        market_prices = np.array([data[1] for data in market_data])
        
        x0 = np.array([initial_guess.v0, initial_guess.vbar, initial_guess.rho, 
                       initial_guess.kappa, initial_guess.sigma])
        
        price_cache = {}
        
        def objective_function(params_array):
            params_key = tuple(np.round(params_array, 8)) 
            
            if params_key in price_cache:
                return price_cache[params_key]
            
            params_array = self._project_to_bounds(params_array)
            params = HestonParameters(*params_array)
            
            model_prices = []
            for k in strikes:
                try:
                    price = self.option_price(params, T, S0, r, q, k)
                    if not np.isfinite(price):
                        price = max(S0 - k, 0) if k < S0 else 0.01
                    model_prices.append(price)
                except:
                    model_prices.append(max(S0 - k, 0) if k < S0 else 0.01)
            
            residuals = np.array(model_prices) - market_prices
            price_cache[params_key] = residuals
            return residuals
        
        bounds = [(self.bounds_lower[i], self.bounds_upper[i]) for i in range(5)]
        
        if verbose:
            initial_rmse = np.linalg.norm(objective_function(x0)) / np.sqrt(len(strikes))
            print(f"Initial RMSE: {initial_rmse:.6e}")
        
        try:
            if method == 'least_squares':
                result = least_squares(
                    objective_function, x0,
                    bounds=(self.bounds_lower, self.bounds_upper),
                    method='trf',
                    ftol=1e-6, 
                    xtol=1e-6,
                    max_nfev=200,  
                    verbose=1 if verbose else 0
                )
                
                success = result.success
                final_params = HestonParameters(*result.x)
                final_rmse = np.linalg.norm(result.fun) / np.sqrt(len(strikes))
                iterations = result.nfev
                
            else: 
                def scalar_objective(params_array):
                    residuals = objective_function(params_array)
                    return np.sum(residuals**2)
                
                result = minimize(
                    scalar_objective, x0,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'ftol': 1e-6, 'maxiter': 100} 
                )
                
                success = result.success
                final_params = HestonParameters(*result.x)
                residuals = objective_function(result.x)
                final_rmse = np.linalg.norm(residuals) / np.sqrt(len(strikes))
                iterations = result.nit
                
        except Exception as e:
            if verbose:
                print(f"Optimization failed: {e}")
            success = False
            final_params = initial_guess
            final_rmse = np.inf
            iterations = 0
        
        if verbose:
            print(f"Final RMSE: {final_rmse:.6e}")
            print(f"Function evaluations: {iterations}")
        
        info = {
            'success': success,
            'final_rmse': final_rmse,
            'iterations': iterations
        }
        
        return final_params, info
    
    def calibrate(self, market_data: List[Tuple[float, float]], T: float, S0: float, 
                 r: float, q: float, initial_guess: HestonParameters = None,
                 method: str = 'scipy', **kwargs) -> Tuple[HestonParameters, dict]:
        if method == 'lm':
            return self.calibrate_lm(market_data, T, S0, r, q, initial_guess, **kwargs)
        else:
            return self.calibrate_scipy(market_data, T, S0, r, q, initial_guess, **kwargs)
    
    def check_feller_condition(self, params: HestonParameters) -> bool:
        return 2 * params.kappa * params.vbar / params.sigma**2 > 1

