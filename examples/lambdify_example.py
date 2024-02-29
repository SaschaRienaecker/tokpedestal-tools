#%%
from sympy import symbols, lambdify

# Define symbols
x = symbols('x')
a = symbols('a')

# Define a sympy expression
expr = x**2 + 3*x + 2 + a

# Convert the sympy expression to a Python function that can use numpy
f = lambdify([x,a], expr, 'numpy')

# Now you can use this function as you would use any Python function
result = f(2,-1)  # Should return 2**2 + 3*2 + 2 - 1= 11
result
# %%
