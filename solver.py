import numpy as np
import matplotlib.pyplot as plt
def euler_method(f, y0, x_span, n):
    a, b = x_span
    h = (b - a) / n
    x = np.linspace(a, b, n+1)  # n+1 nodes
    y = np.zeros(n+1)
    y[0] = y0
    
    for i in range(n):
        y[i+1] = y[i] + h * f(x[i], y[i])
    
    return x, y

def rk4(f, y0, x_span, n):
    """
    4th-order Runge-Kutta method
    
    Parameters:
    f: function defining dy/dx = f(x, y)
    y0: initial condition
    x_span: (a, b) domain
    n: number of intervals
    """
    a, b = x_span
    h = (b - a) / n
    x = np.linspace(a, b, n+1)
    y = np.zeros(n+1)
    y[0] = y0
    
    for i in range(n):
        k1 = f(x[i], y[i])
        k2 = f(x[i] + h/2, y[i] + h*k1/2)
        k3 = f(x[i] + h/2, y[i] + h*k2/2)
        k4 = f(x[i] + h, y[i] + h*k3)
        y[i+1] = y[i] + (h/6) * (k1 + 2*k2 + 2*k3 + k4)
    
    return x, y
def convergence_study(f, y_exact, y0, x_span, n_values):
    """
    Compare Euler and RK4 convergence
    
    f: ODE function
    y_exact: function for exact solution
    y0: initial condition
    x_span: domain
    n_values: list of n values to test (e.g., [10, 20, 50, 100])
    """
    euler_errors = []
    rk4_errors = []
    h_values = []
    
    a, b = x_span
    
    for n in n_values:
        h = (b - a) / n
        h_values.append(h)
        
        # Euler
        x_euler, y_euler = euler_method(f, y0, x_span, n)
        error_euler = np.max(np.abs(y_euler - y_exact(x_euler)))
        euler_errors.append(error_euler)
        
        # RK4
        x_rk4, y_rk4 = rk4(f, y0, x_span, n)
        error_rk4 = np.max(np.abs(y_rk4 - y_exact(x_rk4)))
        rk4_errors.append(error_rk4)
    
    # Plot on log-log scale
    plt.figure(figsize=(10, 6))
    plt.loglog(h_values, euler_errors, 'o-', label='Euler', linewidth=2)
    plt.loglog(h_values, rk4_errors, 's-', label='RK4', linewidth=2)
    
    # Reference lines for theoretical convergence
    plt.loglog(h_values, h_values, '--', label='O(h)', alpha=0.5)
    plt.loglog(h_values, np.array(h_values)**4, '--', label='O(h⁴)', alpha=0.5)
    
    plt.xlabel('Step size h', fontsize=12)
    plt.ylabel('Max error', fontsize=12)
    plt.title('Convergence Comparison: Euler vs RK4', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.show()

# Example usage
f = lambda x, y: -2*y
y_exact = lambda x: np.exp(-2*x)
y0 = 1.0
x_span = (0, 2)
n_values = [10, 20, 50, 100, 200, 500]

convergence_study(f, y_exact, y0, x_span, n_values)
