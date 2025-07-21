import numpy as np
import matplotlib.pyplot as plt
from heston import HestonModel, HestonParameters
import time

def plot_calibration_example():
    model = HestonModel(integration_nodes=24, max_integration_bound=30)
    S0, r, q, T = 100.0, 0.05, 0.0, 1.0
    true_params = HestonParameters(0.08, 0.10, -0.80, 3.00, 0.25)

    strikes = np.array([85, 90, 95, 100, 105, 110, 115])
    market_prices = [model.option_price(true_params, T, S0, r, q, K) for K in strikes]
    market_data = list(zip(strikes, market_prices))
    
    initial_guess = HestonParameters(0.05, 0.06, -0.5, 2.0, 0.3)
    calibrated_params, info = model.calibrate(
        market_data, T, S0, r, q, initial_guess, verbose=False
    )
    
    model_prices = [model.option_price(calibrated_params, T, S0, r, q, K) for K in strikes]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    ax1.plot(strikes, market_prices, 'o-', label='Market Prices', linewidth=2, markersize=8)
    ax1.plot(strikes, model_prices, 's--', label='Calibrated Model', linewidth=2, markersize=8)
    ax1.set_xlabel('Strike Price', fontsize=12)
    ax1.set_ylabel('Option Price', fontsize=12)
    ax1.set_title('Heston Model Calibration Results', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    residuals = np.array(model_prices) - np.array(market_prices)
    ax2.bar(strikes, residuals * 1e6, alpha=0.7, color='red')
    ax2.set_xlabel('Strike Price', fontsize=12)
    ax2.set_ylabel('Pricing Error (×10⁻⁶)', fontsize=12)
    ax2.set_title('Calibration Residuals', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
    
    plt.tight_layout()
    plt.savefig('calibration_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return calibrated_params, info

def plot_implied_volatility_surface():
    model = HestonModel(integration_nodes=20)
    
    params = HestonParameters(v0=0.06, vbar=0.08, rho=-0.7, kappa=2.5, sigma=0.3)
    
    S0, r, q = 100.0, 0.05, 0.0
    
    T_range = np.array([0.25, 0.5, 1.0, 1.5, 2.0])
    moneyness_range = np.array([0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2])
    
    iv_surface = np.zeros((len(T_range), len(moneyness_range)))
    
    for i, T in enumerate(T_range):
        for j, m in enumerate(moneyness_range):
            K = S0 * m
            try:
                heston_price = model.option_price(params, T, S0, r, q, K)
                from scipy.optimize import brentq
                from scipy.stats import norm
                
                def bs_call(sigma):
                    d1 = (np.log(S0/K) + (r - q + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
                    d2 = d1 - sigma*np.sqrt(T)
                    return S0*np.exp(-q*T)*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
                
                def iv_objective(sigma):
                    return bs_call(sigma) - heston_price
                
                if heston_price > 1e-6:
                    iv = brentq(iv_objective, 0.01, 2.0, xtol=1e-6)
                    iv_surface[i, j] = iv
                else:
                    iv_surface[i, j] = np.nan
                    
            except:
                iv_surface[i, j] = np.nan
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    X, Y = np.meshgrid(moneyness_range, T_range)
    
    mask = ~np.isnan(iv_surface)
    surface = ax.plot_surface(X, Y, iv_surface, cmap='viridis', alpha=0.8, antialiased=True)
    
    ax.set_xlabel('Moneyness (K/S₀)', fontsize=12)
    ax.set_ylabel('Time to Maturity (years)', fontsize=12)
    ax.set_zlabel('Implied Volatility', fontsize=12)
    ax.set_title('Heston Model Implied Volatility Surface', fontsize=14, fontweight='bold')
    
    fig.colorbar(surface, shrink=0.6, aspect=20)
    
    plt.savefig('implied_volatility_surface.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return iv_surface

def plot_parameter_sensitivity():
    model = HestonModel(integration_nodes=20)

    base_params = HestonParameters(v0=0.06, vbar=0.08, rho=-0.7, kappa=2.5, sigma=0.3)
    S0, r, q, T, K = 100.0, 0.05, 0.0, 1.0, 100.0
    
    param_ranges = {
        'v0': np.linspace(0.02, 0.12, 20),
        'vbar': np.linspace(0.04, 0.15, 20),
        'rho': np.linspace(-0.9, -0.1, 20),
        'kappa': np.linspace(0.5, 5.0, 20),
        'sigma': np.linspace(0.1, 0.5, 20)
    }
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    param_names = ['v₀ (Initial Variance)', 'v̄ (Long-run Variance)', 'ρ (Correlation)', 
                   'κ (Mean Reversion)', 'σ (Vol of Vol)']
    
    for i, (param, values) in enumerate(param_ranges.items()):
        if i >= 5:  
            break
            
        prices = []
        
        for val in values:
            params_dict = base_params._asdict()
            params_dict[param] = val
            test_params = HestonParameters(**params_dict)
            
            try:
                price = model.option_price(test_params, T, S0, r, q, K)
                prices.append(price)
            except:
                prices.append(np.nan)
        
        axes[i].plot(values, prices, 'o-', linewidth=2, markersize=6)
        axes[i].set_xlabel(param_names[i], fontsize=11)
        axes[i].set_ylabel('Option Price', fontsize=11)
        axes[i].set_title(f'Sensitivity to {param_names[i]}', fontsize=12, fontweight='bold')
        axes[i].grid(True, alpha=0.3)
        
        base_val = getattr(base_params, param)
        base_price = model.option_price(base_params, T, S0, r, q, K)
        axes[i].axvline(x=base_val, color='red', linestyle='--', alpha=0.7, label='Base Value')
        axes[i].legend()
    
    axes[5].remove()
    
    plt.tight_layout()
    plt.savefig('parameter_sensitivity.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_performance_comparison():
    """Plot performance comparison"""
    model = HestonModel(integration_nodes=24)
    
    strike_counts = [3, 5, 7, 10, 15, 20]
    calibration_times = []
    
    S0, r, q, T = 100.0, 0.05, 0.0, 1.0
    true_params = HestonParameters(0.08, 0.10, -0.80, 3.00, 0.25)
    initial_guess = HestonParameters(0.05, 0.06, -0.5, 2.0, 0.3)
    
    for n_strikes in strike_counts:
        strikes = np.linspace(85, 115, n_strikes)
        market_data = [(K, model.option_price(true_params, T, S0, r, q, K)) for K in strikes]
        
        start_time = time.time()
        _, _ = model.calibrate(market_data, T, S0, r, q, initial_guess, verbose=False)
        end_time = time.time()
        
        calibration_times.append(end_time - start_time)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    ax1.plot(strike_counts, calibration_times, 'o-', linewidth=3, markersize=8, color='blue')
    ax1.set_xlabel('Number of Market Prices', fontsize=12)
    ax1.set_ylabel('Calibration Time (seconds)', fontsize=12)
    ax1.set_title('Calibration Performance', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    node_counts = [16, 20, 24, 32, 40, 48]
    pricing_times = []
    
    for nodes in node_counts:
        test_model = HestonModel(integration_nodes=nodes)
        
        start_time = time.time()
        for _ in range(100):
            test_model.option_price(true_params, T, S0, r, q, 100.0)
        end_time = time.time()
        
        pricing_times.append((end_time - start_time) * 10) 
    
    ax2.plot(node_counts, pricing_times, 's-', linewidth=3, markersize=8, color='green')
    ax2.set_xlabel('Integration Nodes', fontsize=12)
    ax2.set_ylabel('Pricing Time per Option (ms)', fontsize=12)
    ax2.set_title('Pricing Performance vs Accuracy', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('performance_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_convergence_analysis():
    """Plot calibration convergence"""
    model = HestonModel(integration_nodes=24)
    
    S0, r, q, T = 100.0, 0.05, 0.0, 1.0
    true_params = HestonParameters(0.08, 0.10, -0.80, 3.00, 0.25)
    strikes = np.array([85, 95, 100, 105, 115])
    market_data = [(K, model.option_price(true_params, T, S0, r, q, K)) for K in strikes]

    initial_guesses = [
        HestonParameters(0.04, 0.04, -0.5, 1.0, 0.2),
        HestonParameters(0.06, 0.08, -0.6, 2.0, 0.3),
        HestonParameters(0.10, 0.12, -0.8, 4.0, 0.4),
        HestonParameters(0.03, 0.05, -0.3, 1.5, 0.15),
    ]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    colors = ['blue', 'red', 'green', 'orange']
    final_params = []
    
    for i, guess in enumerate(initial_guesses):
        calibrated, info = model.calibrate(market_data, T, S0, r, q, guess, verbose=False)
        final_params.append(calibrated)
        
        ax1.scatter([0, 1, 2, 3, 4], 
                   [guess.v0, guess.vbar, guess.rho + 1, guess.kappa/5, guess.sigma],
                   marker='o', s=100, alpha=0.6, color=colors[i], label=f'Initial Guess {i+1}')
        ax1.scatter([0, 1, 2, 3, 4], 
                   [calibrated.v0, calibrated.vbar, calibrated.rho + 1, calibrated.kappa/5, calibrated.sigma],
                   marker='s', s=100, color=colors[i])

    ax1.scatter([0, 1, 2, 3, 4], 
               [true_params.v0, true_params.vbar, true_params.rho + 1, true_params.kappa/5, true_params.sigma],
               marker='*', s=200, color='black', label='True Parameters')
    
    ax1.set_xticks([0, 1, 2, 3, 4])
    ax1.set_xticklabels(['v₀', 'v̄', 'ρ+1', 'κ/5', 'σ'])
    ax1.set_ylabel('Normalized Parameter Value', fontsize=12)
    ax1.set_title('Convergence from Different Initial Guesses', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    param_names = ['v₀', 'v̄', 'ρ', 'κ', 'σ']
    true_values = [true_params.v0, true_params.vbar, true_params.rho, true_params.kappa, true_params.sigma]
    
    errors = []
    for params in final_params:
        calibrated_values = [params.v0, params.vbar, params.rho, params.kappa, params.sigma]
        error = [abs(c - t) for c, t in zip(calibrated_values, true_values)]
        errors.append(error)
    
    x = np.arange(len(param_names))
    width = 0.2
    
    for i, error in enumerate(errors):
        ax2.bar(x + i*width, error, width, label=f'Initial Guess {i+1}', color=colors[i], alpha=0.7)
    
    ax2.set_xlabel('Parameters', fontsize=12)
    ax2.set_ylabel('Absolute Error', fontsize=12)
    ax2.set_title('Parameter Recovery Accuracy', fontsize=14, fontweight='bold')
    ax2.set_xticks(x + width * 1.5)
    ax2.set_xticklabels(param_names)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('convergence_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def generate_all_plots():
    print("Generating Heston Model visualizations...")
    print("This may take a few minutes...")
    
    print("\n1. Calibration example...")
    calibrated_params, info = plot_calibration_example()
    print(f"   Calibrated in {info.get('iterations', 'N/A')} function evaluations")
    print(f"   Final RMSE: {info['final_rmse']:.2e}")
    
    print("\n2. Implied volatility surface...")
    iv_surface = plot_implied_volatility_surface()
    
    print("\n3. Parameter sensitivity analysis...")
    plot_parameter_sensitivity()
    
    print("\n4. Performance analysis...")
    plot_performance_comparison()
    
    print("\n5. Convergence analysis...")
    plot_convergence_analysis()
    

if __name__ == "__main__":
    generate_all_plots()