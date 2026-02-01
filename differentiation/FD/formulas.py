import numpy as np

def forward_difference(f, x, h):
    """Forward difference approximation of f'(x)"""
    return (f(x + h) - f(x)) / h

def backward_difference(f, x, h):
    """Backward difference approximation of f'(x)"""
    return (f(x) - f(x - h)) / h

def centered_difference(f, x, h):
    """Centered difference approximation of f'(x)"""
    return (f(x + h) - f(x - h)) / (2 * h)

def compare_methods(f, f_prime_exact, x, h):
    """
    Compare all three finite difference methods
    
    Returns approximations and errors
    """
    exact = f_prime_exact(x)
    
    fwd = forward_difference(f, x, h)
    bwd = backward_difference(f, x, h)
    ctr = centered_difference(f, x, h)
    
    print(f"At x = {x} with h = {h}:")
    print(f"{'Method':<15} {'Approximation':<20} {'Error':<20}")
    print("-" * 55)
    print(f"{'Exact':<15} {exact:<20.10f} {0:<20}")
    print(f"{'Forward':<15} {fwd:<20.10f} {abs(fwd - exact):<20.10e}")
    print(f"{'Backward':<15} {bwd:<20.10f} {abs(bwd - exact):<20.10e}")
    print(f"{'Centered':<15} {ctr:<20.10f} {abs(ctr - exact):<20.10e}")
    
    return {'exact': exact, 'forward': fwd, 'backward': bwd, 'centered': ctr}

# Example usage
f = lambda x: np.sin(x)
f_prime = lambda x: np.cos(x)

compare_methods(f, f_prime, np.pi/4, 0.01)