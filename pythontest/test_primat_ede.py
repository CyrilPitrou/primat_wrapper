#!/usr/bin/env python
"""
Test script to evaluate PRIMAT likelihood at a single point with EDE parameters
"""

from primat_likelihood import PrimatLikelihood
import numpy as np

# Configure the likelihood
config = {
    'He4_mean': 0.2458,
    'He4_sigma': 0.0013,
    'DH_mean': 2.501e-5,
    'DH_sigma': 0.028e-5,
    'ReducedNetwork': False,
    'DeltaNeff': 0.0,
    'Verbose': True,
    # EDE parameters - set to null to allow them to be passed as parameters
    'fEDE': None,  # Will be provided in test point
    'zcEDE': None, # Will be provided in test point
    'wnEDE': 1.0,  # Fixed equation of state
}

# Create likelihood instance
likelihood = PrimatLikelihood(config)
likelihood.initialize()

# Create a mock provider that returns our test values
class MockProvider:
    def __init__(self, params):
        self.params = params
    
    def get_param(self, name):
        if name in self.params:
            return self.params[name]
        else:
            raise ValueError(f"Parameter {name} not provided")

# Test point with EDE
test_params = {
    'omegabh2': 0.022425,  # Planck 2018 value
    'fEDE': 0.08,          # 8% EDE fraction
    'zcEDE': 3e7,          # Critical redshift ~ 30 million
}

likelihood.provider = MockProvider(test_params)

# Evaluate likelihood
print(f"\n{'='*60}")
print("Testing PRIMAT likelihood with EDE parameters")
print(f"{'='*60}")
print(f"Test point:")
for key, value in test_params.items():
    print(f"  {key:15s} = {value}")
print(f"{'='*60}\n")

logp = likelihood.logp()

print(f"\n{'='*60}")
print(f"Log-likelihood: {logp:.4f}")
print(f"{'='*60}\n")

# Test without EDE for comparison
print(f"\n{'='*60}")
print("Testing without EDE (fEDE=0)")
print(f"{'='*60}")

test_params_no_ede = {
    'omegabh2': 0.022425,
    'fEDE': 0.0,
    'zcEDE': 1e8,
}

likelihood.provider = MockProvider(test_params_no_ede)
logp_no_ede = likelihood.logp()

print(f"\nLog-likelihood (no EDE): {logp_no_ede:.4f}")
print(f"Difference: {logp - logp_no_ede:.4f}")
print(f"{'='*60}\n")
