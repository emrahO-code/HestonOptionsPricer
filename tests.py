import pytest
import numpy as np
from heston import heston_characteristic_function, heston_call_price, heston_put_price

def test_characteristic_function_basic():
    cf_value = heston_characteristic_function(
        u=1.5, 
        S0=100, 
        r=0.05, 
        T=1.0,
        kappa=1.5, 
        theta=0.04, 
        sigma=0.3, 
        rho=-0.7, 
        v0=0.04
    )
    assert isinstance(cf_value, (complex, np.complex128))
    assert not np.isnan(cf_value)
    assert not np.isinf(cf_value)

def test_call_function_basic():
    S0 = 100
    price = heston_call_price(
        S0,
        K = 100,
        r =  0.05,
        T = 1.0,
        kappa = 2.0,
        theta = 0.05,
        sigma = 0.3,
        rho = -0.5,
        v0 = 0.05
    )
    assert isinstance(price, float)
    assert 0 < price < S0
    print(price) 

def test_put_function_basic():
    S0 = 100
    price = heston_put_price(
        S0,
        K = 100,
        r =  0.05,
        T = 1.0,
        kappa = 2.0,
        theta = 0.05,
        sigma = 0.3,
        rho = -0.5,
        v0 = 0.05
    )
    assert isinstance(price, float)
    assert 0 < price < S0
    print(price)
