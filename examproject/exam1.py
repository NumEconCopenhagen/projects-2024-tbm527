import numpy as np
from types import SimpleNamespace
from scipy.optimize import fsolve, minimize
import pandas as pd

# Parameters
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

def optimal_labor_output(w, p, A, gamma):
    optimal_labor = (p * A * gamma / w)**(1 / (1 - gamma))
    optimal_output = A * optimal_labor**gamma
    return optimal_labor, optimal_output

def consumer_optimal_labor(par, w, p1, p2):
    l1, y1 = optimal_labor_output(w, p1, par.A, par.gamma)
    l2, y2 = optimal_labor_output(w, p2, par.A, par.gamma)

    def utility_function(l):
        c1 = par.alpha * (w * l + par.T + p1 * y1 - w * l1 + p2 * y2 - w * l2) / p1
        c2 = (1 - par.alpha) * (w * l + par.T + p1 * y1 - w * l1 + p2 * y2 - w * l2) / (p2 + par.tau)
        return - (np.log(c1**par.alpha * c2**(1 - par.alpha)) - par.nu * l**(1 + par.epsilon) / (1 + par.epsilon))

    result = minimize(utility_function, x0=1, bounds=[(0, None)])
    return result.x[0]

def market_clearing(p1, p2, par, w):
    l1, y1 = optimal_labor_output(w, p1, par.A, par.gamma)
    l2, y2 = optimal_labor_output(w, p2, par.A, par.gamma)

    l_star = consumer_optimal_labor(par, w, p1, p2)
    c1_star = par.alpha * (w * l_star + par.T + p1 * y1 - w * l1 + p2 * y2 - w * l2) / p1
    c2_star = (1 - par.alpha) * (w * l_star + par.T + p1 * y1 - w * l1 + p2 * y2 - w * l2) / (p2 + par.tau)

    return l1 + l2 - l_star, c1_star - y1, c2_star - y2

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


# Function to solve for equilibrium prices
def equilibrium_prices(par, w):
    def equations(p):
        p1, p2 = p
        labor_market, good1_market, good2_market = market_clearing(p1, p2, par, w)
        return [good1_market, good2_market]  # Check good market clearings due to Walras' law
    
    p_initial_guess = [1, 1]
    p_equilibrium = fsolve(equations, p_initial_guess)
    return p_equilibrium

# Social Welfare Function
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

# Optimize Social Welfare
def optimize_social_welfare(par, w):
    result = minimize(lambda x: social_welfare(par, w, x[0], x[1]), [0, 0], bounds=[(0, None), (0, None)])
    optimal_tau, optimal_T = result.x
    return optimal_tau, optimal_T

