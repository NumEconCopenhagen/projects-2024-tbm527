from types import SimpleNamespace
import numpy as np
import matplotlib.pyplot as plt

# Defining default parameters using SimpleNamespace:
def default_parameters():
    return SimpleNamespace(
        alpha=0.3,   # Output elasticity of capital
        phi=0.5,     # Exponent in technology production function
        s=0.2,       # Savings rate
        delta=0.05,  # Depreciation rate
        n=0.015,     # Population growth rate
        lambda_val=1.0,  # Base labor participation rate
        N0=1,        # Initial population
        K0=1         # Initial capital
    )


# Cobb-Douglas production function (1):
def production(K, A, L, alpha):
    return K**alpha * (A * L)**(1-alpha)

# Capital function (2):
def capital(Kt, Yt, s, delta):
    return s * Yt + (1 - delta) * Kt

# Labor force function (3):
def labor(N, lambda_rate):
    return lambda_rate * N

# Population function (4):
def population(Nt, n):
    return (1 + n) * Nt

# Technology level (5):
def technology(K, phi):
    return K**phi

# Real wages
def real_wages(k, A, alpha, lambda_rate):
    return (1 - alpha) * (k / lambda_rate)**alpha * A**(1 - alpha)

# Real interest rate
def real_interest_rate(k, A, alpha, lambda_rate):
    return alpha * (k / lambda_rate)**(alpha - 1) * A**(1 - alpha)

# Simulation of economy
def economy_simulation(par, T, lambda_changes=None):
    K = np.zeros(T+1)
    N = np.zeros(T+1)
    L = np.zeros(T+1)
    Y = np.zeros(T+1)
    W = np.zeros(T)
    R = np.zeros(T)

    K[0], N[0] = par.K0, par.N0
    if lambda_changes is None:
        lambda_changes = np.ones(T) * par.lambda_val

    for t in range(T):
        A_t = technology(K[t], par.phi)
        L[t] = labor(N[t], lambda_changes[t])
        Y[t] = production(K[t], L[t], A_t, par.alpha)
        k_t = K[t] / N[t]
        W[t] = real_wages(k_t, A_t, par.alpha, lambda_changes[t])
        R[t] = real_interest_rate(k_t, A_t, par.alpha, lambda_changes[t])
        K[t+1] = par.s * Y[t] + (1 - par.delta) * K[t]
        N[t+1] = (1 + par.n) * N[t]

    return Y / N, W, R

# Plotting the results
def plot(gdp_constant, gdp_varying, wages_constant, wages_varying, interest_constant, interest_varying, percentage_deviation):
    plt.figure(figsize=(20, 5))
    titles = ['GDP per Capita', 'Real Wages', 'Real Interest Rates', 'Percentage Deviation in GDP Per Capita']
    data_pairs = [(gdp_constant, gdp_varying), (wages_constant, wages_varying), (interest_constant, interest_varying), (percentage_deviation,)]
    labels = [('Constant λ', 'Varying λ'), ('Constant λ', 'Varying λ'), ('Constant λ', 'Varying λ'), ('Percentage Deviation',)]
    
    for i, (data, label) in enumerate(zip(data_pairs, labels)):
        plt.subplot(1, 4, i + 1)
        for d, l in zip(data, label):
            plt.plot(d, label=l)
        plt.title(titles[i])
        plt.xlabel('Time (t)')
        plt.ylabel('Values')
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.show()
