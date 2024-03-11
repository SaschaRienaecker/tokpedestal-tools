"""Define the modified tanh function (mtanh) as per the equations of 
Ref. E. Stefanikova et al. 2016 REVIEW OF SCIENTIFIC INSTRUMENTS 87, see http://dx.doi.org/10.1063/1.4961554
With the addition of an additional parameter `c_slope_SOL` to account for the slope in the outer (SOL) region, similarly for the b_slope parameter for the inner part of the pedestal."""

#%% native python implementations
import numpy as np
def mtanh_py(x, b_slope, c_slope_SOL):
    return ((1 + b_slope * x) * np.exp(x) - (1 + c_slope_SOL) * np.exp(-x)) / (np.exp(x) + np.exp(-x))

def fped_py(r, b_height, b_SOL, b_pos, b_width, b_slope, c_slope_SOL):
    return (b_height - b_SOL) / 2 * (mtanh_py((b_pos - r) / (2 * b_width), b_slope, c_slope_SOL) + 1) + b_SOL

def _fped_exp(r, b_height, b_SOL, b_pos, b_width, c_slope_SOL):
    k_exp=1.0
    return b_height * np.exp(-( (r - b_pos) / b_width)**k_exp) + b_SOL + c_slope_SOL * r

fped_exp = np.vectorize(_fped_exp)
# #%%
# import matplotlib.pyplot as plt
# fig, ax = plt.subplots()
# x = np.linspace(-.1, 1, 100)
# ax.plot(x, fexp_py(x, 1, 0.05, 1, 0.0))

#%% implementations using sympy (symbolic math), which is more elegant in particular if we want to apply derivatives, print the functions as LaTeX, etc.
import matplotlib.pyplot as plt
from sympy import symbols, diff, exp, simplify, lambdify
# Define the symbols
x, b_slope = symbols('x b_slope')
b_height, b_SOL, b_pos, b_width, r = symbols('b_height b_SOL b_pos b_width r')
a_height, a_width, a_exp = symbols('a_height a_width a_exp')
c_slope_SOL = symbols('c_slope_SOL')

# Define the mtanh function (s for symbolic)
smtanh = ((1 + b_slope * x) * exp(x) - (1 + c_slope_SOL * x) * exp(-x)) / (exp(x) + exp(-x))

# Convert the sympy expression to a Python function that can use numpy (f for python function)
fmtanh = lambdify([x, b_slope, c_slope_SOL], smtanh, 'numpy')


# Define the pedestal function (s for symbolic)
sped = (b_height - b_SOL) / 2 * \
    (smtanh.subs(x, (b_pos - r) / (2 * b_width)) + 1) + b_SOL
    

# Convert the sympy expression to a Python function that can use numpy (f for python function)
fped = lambdify([r, b_height, b_SOL, b_pos,
                  b_width, b_slope, c_slope_SOL], sped, 'numpy')


# Define the full function F_full combining F_ped and the exponential decay (sfull for symbolic)
sfull = sped + (a_height - sped) * exp(-(r / a_width)**a_exp)


# Convert the sympy expression to a Python function that can use numpy (ffull for python function)
ffull = lambdify([r, b_height, b_SOL, b_pos, b_width, b_slope, c_slope_SOL,
                 a_height, a_width, a_exp], sfull, 'numpy')


# Show the symbolic expression of the pedestal function
sped

#%%
# Calculate the derivative of mtanh with respect to x
d_mtanh_dx = diff(smtanh, x)

# Convert the sympy expression for the derivative to a numpy-compatible function
f_d_mtanh_dx = lambdify([x, b_slope, c_slope_SOL], d_mtanh_dx, 'numpy')


# Calculate the derivative of ped with respect to r (might be useful eventually)
d_ped_dr = diff(sped, r)
f_d_ped_dr = lambdify([r, b_height, b_SOL, b_pos,
                          b_width, b_slope, c_slope_SOL], d_ped_dr, 'numpy')

#%%
if __name__ == '__main__':
    
    # Some plots to visualize how we can make use of the functions
    
    params = {
        'b_height': 5,
        'b_SOL': 1,
        'b_pos': 5,
        'b_width': 1,
        'b_slope': 0.02,
        'c_slope_SOL': 0.0,
    }
    def _check(f1,f2, x, *args):
        fig, ax = plt.subplots()
        ax.plot(x, f1(x, *args))
        ax.plot(x, f2(x, *args),ls='--')
        
    # check for consistency with the native python function `mtanh_py`
    _x = np.linspace(-5, 5, 100)
    _check(fmtanh, mtanh_py, _x, params['b_slope'], params['c_slope_SOL'])
    plt.gca().legend(['sympy', 'pedestal.py'])
    plt.gca().set_title('mtanh')
    
    b,c = params['b_slope'], params['c_slope_SOL'] # shorthand

    
    # check for consistency with pedestal.py
    _check(fped, fped_py, _x, *params.values())
    plt.gca().legend(['sympy', 'pedestal.py'])
    plt.gca().set_title('Fped')
    
    from scipy.misc import derivative
    d_mtanh_dx_num = derivative(fmtanh, _x, dx=np.diff(_x).mean(), args=[b,c])
    fig, ax = plt.subplots()
    ax.plot(_x, f_d_mtanh_dx(_x, b,c), label='analytical (sympy)')
    ax.plot(_x, d_mtanh_dx_num, ls='--', label='numerical (scipy)')
    ax.set_title('mtanh derivative')
    ax.legend()
    
    # _r = params['b_pos'] - 2 * params['b_width'] * _x
    _r = np.copy(_x)
    d_ped_dx_num = derivative(
        fped, _r, dx=np.diff(_r).mean(), args=params.values())
    fig, ax = plt.subplots()
    ax.plot(_r, f_d_ped_dr(_r, *params.values()), label='analytical (sympy)')
    ax.plot(_r, d_ped_dx_num, ls='--', label='numerical (scipy)')
    ax.set_title('Fped derivative')
    ax.legend()
    
    
    #%% Latex printing of both expressions
    from sympy import init_printing
    init_printing(use_latex='mathjax')
    # Simplify the results
    d_mtanh_dx_simplified = simplify(d_mtanh_dx)
    d_ped_dr_simplified = simplify(d_ped_dr)
    d_mtanh_dx_simplified, d_ped_dr_simplified

# %%
