
#%%
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from pedestal import fped

# Let's create a dummy data set to illustrate how the fitting would work
# Since we don't have real data, we'll generate some synthetic data using the fped function
# with some added noise.

# Generate some synthetic 'r' values (radius values)
r_values = np.linspace(0, 10, 50)  # 50 points between 0 and 10

# Parameters for the synthetic data
b_height_true = 5
b_SOL_true = 1
b_pos_true = 5
b_width_true = 1
b_slope_true = 0.5
c_slope_SOL_true = -0.1

# Generate synthetic 'fped' values using the true parameters and add some noise
F_ped_values = fped(r_values, b_height_true, b_SOL_true, b_pos_true, b_width_true, b_slope_true, c_slope_SOL_true)
F_ped_values_noise = F_ped_values + 0.2 * np.random.normal(size=F_ped_values.size)

if __name__ == '__main__':

    # Now let's fit the model to this synthetic data using scipy's curve_fit
    # Initial guess for the parameters
    initial_guess = [4, 1, 4, 1, 0.5, 0.]

    # Perform the curve fitting
    popt, pcov = curve_fit(fped, r_values, F_ped_values_noise, p0=initial_guess)

    # popt contains the best fit parameters
    b_height_fit, b_SOL_fit, b_pos_fit, b_width_fit, b_slope_fit, c_slope_SOL_true = popt

    # Generate the fitted curve
    F_ped_fitted = fped(r_values, *popt)

    # Plot the synthetic data and the fitted curve
    plt.figure(figsize=(10, 6))
    plt.scatter(r_values, F_ped_values_noise, label='Data (with noise)')
    plt.plot(r_values, F_ped_fitted, label='Fitted curve', color='red')
    plt.title('Fit of fped to synthetic data')
    plt.xlabel('r')
    plt.ylabel('fped(r,b)')
    plt.legend()
    plt.show()

    # Return the fitted parameters
    popt, np.sqrt(np.diag(pcov))  # Return the parameters and their standard deviations

    # %%
