from types import SimpleNamespace
from scipy import optimize
from scipy.optimize import minimize

class InauguralProjectClass:

    def __init__(self):
        self.par = SimpleNamespace()

        # a. preferences
        self.par.alpha = 1/3
        self.par.beta = 2/3

        # b. endowments
        self.par.w1A = 0.8
        self.par.w2A = 0.3
        self.par.w1B = 1 - self.par.w1A
        self.par.w2B = 1 - self.par.w2A

    # Utility functions
    def utility_A(self, x1A, x2A):
        return x1A**self.par.alpha * x2A**(1-self.par.alpha)

    def utility_B(self, x1B, x2B):
        return x1B**self.par.beta * x2B**(1-self.par.beta)

    # Demand functions
    def demand_A(self, p1):
        # The numeraire is set to p2 = 1
        p2 = 1

        # For simplicity, we define the income/budget for consumer A
        IncomeA = p1 * self.par.w1A + p2 * self.par.w2A

        # The demand functions of consumer A is given by
        x1A = (self.par.alpha * IncomeA) / p1
        x2A = ((1-self.par.alpha) * IncomeA) / p2
        return x1A, x2A

    def demand_B(self, p1):
        # The numeraire is set to p2 = 1
        p2 = 1

        # For simplicity, we define the income/budget for consumer B
        IncomeB = p1 * self.par.w1B + p2 * self.par.w2B

        # The demand functions of consumer B is given by
        x1B = (self.par.beta * IncomeB) / p1
        x2B = ((1-self.par.beta) * IncomeB) / p2
        return x1B, x2B

    # Market clearing function
    def check_market_clearing(self, p1):
        x1A, x2A = self.demand_A(p1)
        x1B, x2B = self.demand_B(p1)

        eps1 = x1A - self.par.w1A + x1B - self.par.w1B
        eps2 = x2A - self.par.w2A + x2B - self.par.w2B

        return eps1, eps2

# For Question 4:
# We define the function that calculates consumer A's utility influenced by consumer B's demands
def objective_function(p1, model):
    # We calculate demands for consumer B at this p1
    x1B, x2B = model.demand_B(p1)
    
    # We calculate new endowments for consumer A based on B's demands
    x1A_new = 1 - x1B
    x2A_new = 1 - x2B
    
    # We check for invalid values
    if x1A_new <= 0 or x2A_new <= 0:
        return np.inf
    
    # We calculate utility for A given these new endowments
    utility_A = model.utility_A(x1A_new, x2A_new)
    
    # Since we are using a minimizer, we return the negative utility to maximize it
    return -utility_A