import pytest
import numpy as np
from heston import heston_characteristic_function 

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