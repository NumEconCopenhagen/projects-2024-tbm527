import numpy as np
from types import SimpleNamespace
from scipy.optimize import fsolve, minimize
import pandas as pd

### Problem 1 ###
# I set the parameters
par = SimpleNamespace()
par.A = 1.0
par.gamma = 0.5
par.alpha = 0.3
par.nu = 1.0
par.epsilon = 2.0
par.tau = 0.0
par.T = 0.0
par.kappa = 0.1
w = 1  # The wage is set as numeraire

# I create a function for the firms' optimal labor and output
def optimal_labor_output(w, p, A, gamma):
    optimal_labor = (p * A * gamma / w)**(1 / (1 - gamma))
    optimal_output = A * optimal_labor**gamma
    return optimal_labor, optimal_output

# I create a function for the consumers' optimal labor output
def consumer_optimal_labor(par, w, p1, p2):
    l1, y1 = optimal_labor_output(w, p1, par.A, par.gamma)
    l2, y2 = optimal_labor_output(w, p2, par.A, par.gamma)

    # Utility function
    def utility_function(l):
        c1 = par.alpha * (w * l + par.T + p1 * y1 - w * l1 + p2 * y2 - w * l2) / p1
        c2 = (1 - par.alpha) * (w * l + par.T + p1 * y1 - w * l1 + p2 * y2 - w * l2) / (p2 + par.tau)
        return - (np.log(c1**par.alpha * c2**(1 - par.alpha)) - par.nu * l**(1 + par.epsilon) / (1 + par.epsilon))

    result = minimize(utility_function, x0=1, bounds=[(0, None)])
    return result.x[0]

# I create a market clearing function
def market_clearing(p1, p2, par, w):
    l1, y1 = optimal_labor_output(w, p1, par.A, par.gamma)
    l2, y2 = optimal_labor_output(w, p2, par.A, par.gamma)

    l_star = consumer_optimal_labor(par, w, p1, p2)
    c1_star = par.alpha * (w * l_star + par.T + p1 * y1 - w * l1 + p2 * y2 - w * l2) / p1
    c2_star = (1 - par.alpha) * (w * l_star + par.T + p1 * y1 - w * l1 + p2 * y2 - w * l2) / (p2 + par.tau)

    return l1 + l2 - l_star, c1_star - y1, c2_star - y2

# I create a function that checks for market clearing and return it as a DataFrame
def check_market_clearing(par, w):
    p1_values = np.linspace(0.1, 2.0, 10)
    p2_values = np.linspace(0.1, 2.0, 10)
    results = []

    for p1 in p1_values:
        for p2 in p2_values:
            labor_market, good1_market, good2_market = market_clearing(p1, p2, par, w)
            results.append((p1, p2, labor_market, good1_market, good2_market))

    df = pd.DataFrame(results, columns=['p1', 'p2', 'Labor Market', 'Good 1 Market', 'Good 2 Market'])
    return df

# I create a function to solve for equilibrium prices
def equilibrium_prices(par, w):
    def equations(p):
        p1, p2 = p
        labor_market, good1_market, good2_market = market_clearing(p1, p2, par, w)
        return [good1_market, good2_market]  # Check good market clearings due to Walras' law
    
    p_initial_guess = [1, 1]
    p_equilibrium = fsolve(equations, p_initial_guess)
    return p_equilibrium

# I create a social welfare function
def social_welfare(par, w, tau, T):
    par.tau = tau
    par.T = T
    p1, p2 = equilibrium_prices(par, w)
    l1, y1 = optimal_labor_output(w, p1, par.A, par.gamma)
    l2, y2 = optimal_labor_output(w, p2, par.A, par.gamma)
    
    l_star = consumer_optimal_labor(par, w, p1, p2)
    c1_star = par.alpha * (w * l_star + T + p1 * y1 - w * l1 + p2 * y2 - w * l2) / p1
    c2_star = (1 - par.alpha) * (w * l_star + T + p1 * y1 - w * l1 + p2 * y2 - w * l2) / (p2 + tau)
    
    utility = np.log(c1_star**par.alpha * c2_star**(1 - par.alpha)) - par.nu * l_star**(1 + par.epsilon) / (1 + par.epsilon)
    swf = utility - par.kappa * y2
    
    return -swf  # We minimize the negative social welfare to maximize it

# I optimize the social welfare
def optimize_social_welfare(par, w):
    result = minimize(lambda x: social_welfare(par, w, x[0], x[1]), [0, 0], bounds=[(0, None), (0, None)])
    optimal_tau, optimal_T = result.x
    return optimal_tau, optimal_T

### Problem 3 ###
# Question 3.1
# I create a function to generate random points and the corresponding function values
def generate_random_points(seed=2024, size=50):
    rng = np.random.default_rng(seed)
    X = rng.uniform(size=(size, 2))
    return X, np.array([x[0] * x[1] for x in X])

# I find points A, B, C, D
def find_points(X, y):
    try:
        A = min([p for p in X if p[0] > y[0] and p[1] > y[1]], key=lambda p: np.linalg.norm(p - y))
    except ValueError:
        A = None

    try:
        B = min([p for p in X if p[0] > y[0] and p[1] < y[1]], key=lambda p: np.linalg.norm(p - y))
    except ValueError:
        B = None

    try:
        C = min([p for p in X if p[0] < y[0] and p[1] < y[1]], key=lambda p: np.linalg.norm(p - y))
    except ValueError:
        C = None

    try:
        D = min([p for p in X if p[0] < y[0] and p[1] > y[1]], key=lambda p: np.linalg.norm(p - y))
    except ValueError:
        D = None

    return A, B, C, D

# Question 3.2
# I create a Barycentric coordinates calculation
def barycentric_coordinates(A, B, C, P):
    denom = (B[1] - C[1]) * (A[0] - C[0]) + (C[0] - B[0]) * (A[1] - C[1])
    r1 = ((B[1] - C[1]) * (P[0] - C[0]) + (C[0] - B[0]) * (P[1] - C[1])) / denom
    r2 = ((C[1] - A[1]) * (P[0] - C[0]) + (A[0] - C[0]) * (P[1] - C[1])) / denom
    r3 = 1 - r1 - r2
    return r1, r2, r3

# Question 3.3
# I create a function to perform the full algorithm for a single point
def interpolate_f_for_y(y, X, F):
    A, B, C, D = find_points(X, y)

    if A is None or B is None or C is None or D is None:
        return np.nan

    if not np.isnan(A).any() and not np.isnan(B).any() and not np.isnan(C).any():
        r_ABC = barycentric_coordinates(A, B, C, y)
    else:
        r_ABC = (np.nan, np.nan, np.nan)

    if not np.isnan(C).any() and not np.isnan(D).any() and not np.isnan(A).any():
        r_CDA = barycentric_coordinates(C, D, A, y)
    else:
        r_CDA = (np.nan, np.nan, np.nan)

    inside_ABC = all(0 <= r <= 1 for r in r_ABC)
    inside_CDA = all(0 <= r <= 1 for r in r_CDA)

    if inside_ABC:
        approx_f_y = r_ABC[0] * F[np.where((X == A).all(axis=1))[0][0]] + \
                     r_ABC[1] * F[np.where((X == B).all(axis=1))[0][0]] + \
                     r_ABC[2] * F[np.where((X == C).all(axis=1))[0][0]]
    elif inside_CDA:
        approx_f_y = r_CDA[0] * F[np.where((X == C).all(axis=1))[0][0]] + \
                     r_CDA[1] * F[np.where((X == D).all(axis=1))[0][0]] + \
                     r_CDA[2] * F[np.where((X == A).all(axis=1))[0][0]]
    else:
        approx_f_y = np.nan

    return approx_f_y

# Question 3.4
# I create the main function to process all points in Y
def process_points(Y, X, F):
    results = [(y, interpolate_f_for_y(y, X, F), y[0] * y[1]) for y in Y]
    return results