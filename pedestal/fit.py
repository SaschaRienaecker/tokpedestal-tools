#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, Bounds
from scipy.interpolate import interp1d
from pedestal.fped_functions import fped as fped_mtanh, fped_exp

def huber_loss(residuals, delta=1.0): # presently not used, but could be if we want to fine-tune the loss function
    """Huber loss function. The delta parameter controls the threshold between linear and quadratic loss."""
    absolute_residuals = np.abs(residuals)
    quadratic_loss = (absolute_residuals**2)
    linear_loss = (absolute_residuals)
    loss = np.where(absolute_residuals <= delta, quadratic_loss, linear_loss)
    return np.sum(loss)

def lin_transition(x, x0, w):
    """Linear function acting as a transition between the pedestal and the proffit profile, with width w."""
    return np.clip((x - x0) / w, 0, 1)

def smooth_transition(x, x0, w):
    """Serves as a smooth transition between the pedestal function and the proffit profile, with width w. Better than lin,_transition."""
    return 0.5 * (1 + np.tanh((x - x0) / w))

def objective_function(params, x, y, f_core, w=0.1, fped=fped_mtanh):
    """Objective function to minimize. It returns the sum of the squared residuals between the data and the model."""   
    model_values = full_profile_with_pedestal(x, params, f_core, w, positive_only=False, fped=fped)
    # residuals = np.sum(np.abs(model_values - y)) # linear residuals, should be less prone to outliers but maybe not always good for fitting.
    residuals = np.sum((model_values - y)**2)
    return residuals
    
def full_profile_with_pedestal(x, params, f_core, w=0.1, positive_only=True, fped=fped_mtanh):
        """Smooth transition between the pedestal function and the core profile function, with width w.

        Args:
            x (np.ndarray): independent variable
            params (tuple): parameters of the pedestal function `fped` + `x_edge_core_bnd` which is the free parameter that defined the transition between the pedestal and the core profile
            f_core (callable): core profile function
            w (float): transition width
        """
        fped_args = params[:-1]
        x_edge_core_bnd = params[-1]
        f = smooth_transition(x, x_edge_core_bnd, w)  # smoothing factor
        # f = lin_transition(x, x_edge_core_bnd, w)  # smoothing factor
        y = f * fped(x, *fped_args) + (1 - f) * f_core(x)
        
        if positive_only:
            y[y<0] = 0.0
            
        return y
# %%
if __name__ == '__main__':
    
    pass