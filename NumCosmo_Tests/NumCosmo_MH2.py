import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from numcosmo_py import Nc, Ncm
import time  # For timing

# Initialize the NumCosmo library
Ncm.cfg_init()

# Create a cosmological model (ΛCDM)
cosmo = Nc.HICosmoDEXcdm.new()

# Access default parameters of the cosmological model
H0_init = cosmo.props.H0
Omegab_init = cosmo.props.Omegab

# Create a distance object
dist = Nc.Distance.new(2.0)  # Distance object for calculations
dist.prepare(cosmo)  # Prepare the distance object for the cosmology

# Generate mock data
z_min = 0.01
z_max = 2
num_data_points = 150
z = np.linspace(z_min, z_max, num_data_points)
d_lum = np.array([dist.luminosity(cosmo, z_i) for z_i in z])

# Add Gaussian noise to simulate observational uncertainties
np.random.seed(42)  # For reproducibility
noise_level = 0.15 * d_lum  # 15% noise
mock_d_lum = d_lum + np.random.normal(0, noise_level)

# Define the log-likelihood function
def Log_Likelihood(H0, Omegab, cosmo, dist, z, mock_d_lum, noise_level):
    if H0 <= 0 or Omegab < 0 or Omegab > 1:
        return -np.inf  # Reject unphysical values

    # Update the cosmological model
    cosmo.props.H0 = H0
    cosmo.props.Omegab = Omegab

    # Compute the model-predicted luminosity distances
    d_lum_model = np.array([dist.luminosity(cosmo, z_i) for z_i in z])

    # Compute the chi-squared
    chi2 = np.sum(((mock_d_lum - d_lum_model) ** 2) / (noise_level ** 2))
    return -0.5 * chi2

# Metropolis-Hastings algorithm
def metropolis_hastings(iterations, H0_init=70, Omegab_init=0.3, step_size=1.0):
    H0_chain = [H0_init]
    Omegab_chain = [Omegab_init]
    Log_Likelihood_chain = [Log_Likelihood(H0_init, Omegab_init, cosmo, dist, z, mock_d_lum, noise_level)]

    for _ in range(iterations):
        # Propose new parameters
        H0_new = np.random.normal(H0_chain[-1], step_size)
        Omegab_new = np.random.normal(Omegab_chain[-1], step_size * 0.1)

        # Compute the new log-likelihood
        Log_Likelihood_new = Log_Likelihood(H0_new, Omegab_new, cosmo, dist, z, mock_d_lum, noise_level)
        Log_Likelihood_old = Log_Likelihood_chain[-1]

        # Compute the acceptance ratio
        accept_ratio = np.exp(Log_Likelihood_new - Log_Likelihood_old)

        # Accept or reject the proposal
        if np.random.rand() < accept_ratio:
            H0_chain.append(H0_new)
            Omegab_chain.append(Omegab_new)
            Log_Likelihood_chain.append(Log_Likelihood_new)
        else:
            H0_chain.append(H0_chain[-1])
            Omegab_chain.append(Omegab_chain[-1])
            Log_Likelihood_chain.append(Log_Likelihood_old)

    return np.array(H0_chain), np.array(Omegab_chain)

# Run the Metropolis-Hastings algorithm
iterations = 10000
H0_sample, Omegab_sample = metropolis_hastings(iterations)

# Plot results
plt.figure(figsize=(12, 5))

# Histogram of H0
plt.subplot(1, 2, 1)
plt.hist(H0_sample[1000:], bins=30, density=True, alpha=0.7)
plt.axvline(H0_init, color='r', linestyle='--', label=f"True H0={H0_init}")
plt.xlabel("$H_0$")
plt.ylabel("Density")
plt.legend()

# Scatter plot of Omega_b vs H0
plt.subplot(1, 2, 2)
plt.scatter(H0_sample[1000:], Omegab_sample[1000:], s=1, alpha=0.5)
plt.axhline(Omegab_init, color='r', linestyle='--', label=f"True $\\Omega_b$={Omegab_init}")
plt.xlabel("$H_0$")
plt.ylabel("$\\Omega_b$")
plt.legend()

plt.tight_layout()
plt.show()

