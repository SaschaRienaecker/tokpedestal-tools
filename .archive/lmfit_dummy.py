#%%
import matplotlib.pyplot as plt
import numpy as np
import lmfit
from pedestal.fped_functions import f_d_ped_dr as dF_ped_dr
from pedestal.fped_functions import fped as F_ped

# Define the target F_ped value and derivative at a specific r
target_F_ped_value = 6.5  # the target F_ped value
target_F_ped_derivative = -0.5  # the target derivative value
target_r = 2  # the r value at which F_ped and its derivative should match the target values

# Create a model for F_ped
model = lmfit.Model(F_ped)

# Define the parameters for the model, with initial values
params = model.make_params(
    b_height=5, b_SOL=1, b_pos=5, b_width=1, b_slope=0.5)

# Optionally set bounds or constraints on parameters
# params['b_height'].min = 0
# params['b_height'].max = 10
# ...

# Define a custom objective function that includes the constraints

def custom_objective(params, r, data, continuity_weight=1000, der_continuity_weight=1):
    # Evaluate the current model given the parameters and r
    model_values = model.eval(params, r=r)

    # Calculate the residual normally (data - model)
    residual = data - model_values

    # Calculate the constraint for F_ped at target_r
    constraint_value = F_ped(target_r, **params) - target_F_ped_value

    # Calculate the constraint for the derivative of F_ped at target_r
    constraint_derivative = dF_ped_dr(
        target_r, **params) - target_F_ped_derivative

    # Append the constraints to the residuals array
    return np.append(residual, [constraint_value * continuity_weight, constraint_derivative * der_continuity_weight])


# Dummy data for illustration - replace with your actual data
r_data = np.linspace(0, 10, 100)
region_to_fit = (r_data > target_r)
F_ped_data = F_ped(r_data, 5, 1, 5, 1, 0.5)  # Replace with actual data

# Perform the fit
result = lmfit.minimize(custom_objective, params, args=(r_data[region_to_fit], F_ped_data[region_to_fit]))

# Print out the fit results
# print(result.fit_report())


# %%
import matplotlib.pyplot as plt
# Generate the fitted curve using the best-fit parameters
fitted_values = model.eval(result.params, r=r_data[region_to_fit])

# Plot the original data
plt.figure(figsize=(10, 6))
plt.plot(r_data, F_ped_data, 'o', label='Data')

# Plot the fitted curve
plt.plot(r_data[region_to_fit], fitted_values, label='Best Fit', linewidth=2)

# Optionally, you can also plot the residuals
# residuals = F_ped_data - fitted_values
# plt.figure(figsize=(10, 6))
# plt.plot(r_data, residuals, 'o', label='Residuals')
# plt.axhline(0, color='red', linestyle='--')
# plt.figure(2)
# plt.xlabel('r')
# plt.ylabel('Residuals')
# plt.title('Residuals of the Fit')
# plt.legend()

# Add labels and legend
plt.figure(1)
plt.xlabel('r')
plt.ylabel('F_ped')
plt.title('Data vs. Fitted Model with boundary cond.')
plt.legend()

# indicate the point at which the boundary is enforced:
xbnd, ybnd = target_r, target_F_ped_value
slope = target_F_ped_derivative
plt.plot(xbnd, ybnd, 'o', color='red', label='Boundary condition')
_xbnd = np.linspace(0, target_r, 100)
plt.plot(_xbnd, ybnd + slope * (_xbnd - xbnd), color='red', ls='--')
plt.legend()

# %%
