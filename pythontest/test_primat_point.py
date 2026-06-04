#!/usr/bin/env python
"""
Quick test script to evaluate PRIMAT likelihood at a single point.
This is useful for debugging and checking that everything works.

Usage:
    python test_primat_point.py
"""

from cobaya.model import get_model
import yaml
import numpy as np

# Configuration for single point evaluation
config = {
    'debug': True,
    'likelihood': {
        'primat_likelihood.PrimatLikelihood': {
            'He4_mean': 0.2458,
            'He4_sigma': 0.0013,
            'DH_mean': 2.527e-5,
            'DH_sigma': 0.030e-5,
            'MathKernelCommand': '/Applications/Wolfram.app/Contents/MacOS/MathKernel',
            'ReducedNetwork': False,
            'DeltaNeff': 0.0,
            'Verbose': True  # Keep verbose for debugging
        }
    },
    'params': {
        'omegabh2': 0.022425,  # Test value - only parameter BBN depends on
    }
}

def test_single_point():
    """Test PRIMAT likelihood at a single point"""
    print("=" * 70)
    print("Testing PRIMAT likelihood at Planck 2018 bestfit parameters")
    print("=" * 70)
    print()
    
    try:
        # Create model
        print("Creating Cobaya model...")
        model = get_model(config)
        print("✓ Model created successfully")
        print()
        
        # Get point to evaluate
        point = {
            'omegabh2': 0.022425,  # Test value
        }
        
        print(f"Evaluating at point:")
        for param, value in point.items():
            print(f"  {param:12s} = {value}")
        print()
        
        # Evaluate log-likelihood
        print("Computing theory and likelihood...")
        logpost = model.logpost(point)
        
        print()
        print("=" * 70)
        print("Results:")
        print("=" * 70)
        
        # Check if logpost is a LogPosterior object or just a float
        if hasattr(logpost, 'loglike'):
            print(f"Log-likelihood: {logpost.loglike:.4f}")
            print(f"Log-prior:      {logpost.logprior:.4f}")
            print(f"Log-posterior:  {logpost.logpost:.4f}")
            print()
            
            # Get derived parameters if any
            if hasattr(logpost, 'derived') and logpost.derived:
                print("Derived parameters:")
                for key, value in logpost.derived.items():
                    print(f"  {key:12s} = {value}")
        else:
            # If it returned -inf or a simple float
            print(f"Log-posterior:  {logpost:.4f}")
            if logpost == -np.inf:
                print("\n⚠ Likelihood returned -inf (evaluation failed)")
        
        print()
        print("✓ Test completed successfully!")
        return True
        
    except Exception as e:
        print()
        print("=" * 70)
        print("✗ Error occurred:")
        print("=" * 70)
        print(f"{type(e).__name__}: {e}")
        print()
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = test_single_point()
    exit(0 if success else 1)
