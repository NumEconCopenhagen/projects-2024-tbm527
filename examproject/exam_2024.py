import numpy as np
from types import SimpleNamespace
import pandas as pd
from scipy.optimize import fsolve
from scipy.optimize import minimize

class ExamProject:
    def __init__(self):
        self.par = SimpleNamespace()

        # firms
        self.par.A = 1.0
        self.par.gamma = 0.5

        # households
        self.par.alpha = 0.3
        self.par.nu = 1.0
        self.par.epsilon = 2.0

        # government
        self.par.tau = 0.0
        self.par.T = 0.0

        # Question 3
        self.par.kappa = 0.1

        self.w = 1  # Numeraire

    # Function to calculate optimal labor and output
    def optimal_labor_output(self, w, p, A, gamma):
        optimal_labor = (p * A**gamma / w)**(1 / (1 - gamma))
        optimal_output = A * optimal_labor**gamma
        return optimal_labor, optimal_output

    # Function to calculate market clearing conditions
    def market_clearing(self, p1, p2, par, w):
        l1, y1 = self.optimal_labor_output(w, p1, par.A, par.gamma)
        l2, y2 = self.optimal_labor_output(w, p2, par.A, par.gamma)
    
        c1 = par.alpha * (w * (l1 + l2) + par.T + (p1 * y1 - w * l1) + (p2 * y2 - w * l2)) / p1
        c2 = (1 - par.alpha) * (w * (l1 + l2) + par.T + (p1 * y1 - w * l1) + (p2 * y2 - w * l2)) / (p2 + par.tau)
    
        return l1 + l2, c1 - y1, c2 - y2  # Labor market, good market 1, good market 2
    
    # Function to solve for equilibrium prices
    def equilibrium_prices(self, par, w):
        def equations(p):
            p1, p2 = p
            labor_market, good1_market, _ = self.market_clearing(p1, p2, par, w)
            return [labor_market, good1_market]  # Check labor market and good 1 market clearing
    
        p_initial_guess = [1, 1]
        p_equilibrium = fsolve(equations, p_initial_guess)
        return p_equilibrium
    
    # Social welfare function
    def social_welfare(self, params, par, w):
        tau, T = params
        par.tau = tau
        par.T = T
        p1_eq, p2_eq = self.equilibrium_prices(par, w)
        l1, y1 = self.optimal_labor_output(w, p1_eq, par.A, par.gamma)
        l2, y2 = self.optimal_labor_output(w, p2_eq, par.A, par.gamma)
    
        # Calculate utility U
        labor_market, _, _ = self.market_clearing(p1_eq, p2_eq, par, w)
        l_opt = labor_market
        c1_opt = par.alpha * (w * l_opt + T + p1_eq * y1 - w * l1 + p2_eq * y2 - w * l2) / p1_eq
        c2_opt = (1 - par.alpha) * (w * l_opt + T + p1_eq * y1 - w * l1 + p2_eq * y2 - w * l2) / (p2_eq + tau)
    
        U = np.log(c1_opt**par.alpha * c2_opt**(1 - par.alpha)) - par.nu * l_opt**(1 + par.epsilon) / (1 + par.epsilon)
        SWF = U - par.kappa * y2  # Social welfare function
        return -SWF  # Minimize negative SWF to maximize SWF


