#!/usr/bin/env python
#
# example_mcmc_likelihood.py
#
# Updated version using Metropolis-Hastings with NumCosmo
#
import os
import numpy as np
import matplotlib.pyplot as plt
from numcosmo_py import Nc, Ncm

# Initialize the NumCosmo library
Ncm.cfg_init()

# Define the path to the mock data file
mock_data_path = os.path.join(os.path.dirname(__file__), "mock_supernova_data.txt")

# Check if the file exists
if not os.path.exists(mock_data_path):
    raise FileNotFoundError(f"Mock data file not found at: {mock_data_path}")

# Load the mock data
mock_data = np.loadtxt(mock_data_path)
z = mock_data[:, 0]
d_lum = mock_data[:, 1]
error = mock_data[:, 2]

# Store data in NumCosmo vectors
z_vec = Ncm.Vector.new_array(z)
d_lum_vec = Ncm.Vector.new_array(d_lum)
error_vec = Ncm.Vector.new_array(error)

# Create a cosmological model (ΛCDM)
cosmo = Nc.HICosmoDEXcdm.new()

# Set initial values for parameters
cosmo.props.Omegab = 0.05  # Baryon density
cosmo.props.H0 = 70.0      # Hubble constant

# Create the NumCosmo likelihood
like_d_lum = Nc.DataDL.new()
like_d_lum.set_z_data(z_vec)
like_d_lum.set_d_lum_data(d_lum_vec)
like_d_lum.set_d_lum_error(error_vec)

# Create dataset and likelihood
dataset = Ncm.Dataset()
dataset.append_data(like_d_lum)
likelihood = Ncm.Likelihood(dataset=dataset)

# Metropolis-Hastings Algorithm
iterations = 10000

def metropolis_hastings(iterations, init_Omegab=0.05, init_H0=70.0, step_size=0.01):
    H0_chain = [init_H0]
    Omegab_chain = [init_Omegab]
    likelihood_chain = [likelihood.eval(None)]

    for _ in range(iterations):
        new_H0 = np.random.normal(H0_chain[-1], step_size)
        new_Omegab = np.random.normal(Omegab_chain[-1], step_size)
        
        cosmo.props.H0 = new_H0
        cosmo.props.Omegab = new_Omegab
        likelihood_new = likelihood.eval(None)
        likelihood_old = likelihood_chain[-1]
        accept = np.exp(likelihood_new - likelihood_old)

        if np.random.rand() <= accept:
            H0_chain.append(new_H0)
            Omegab_chain.append(new_Omegab)
            likelihood_chain.append(likelihood_new)
        else:
            H0_chain.append(H0_chain[-1])
            Omegab_chain.append(Omegab_chain[-1])
            likelihood_chain.append(likelihood_old)

    return np.array(H0_chain), np.array(Omegab_chain)

H0_sample, Omegab_sample = metropolis_hastings(iterations)

# Plot results
plt.figure(figsize=(12, 5))

# Histogram of H0
plt.subplot(1, 2, 1)
plt.hist(H0_sample, bins=20, density=True, alpha=0.7)
plt.axvline(70.0, color='r', linestyle='--', label="True H0=70.0")
plt.xlabel("$H_0$")
plt.ylabel("Density")
plt.legend()

# Scatter plot of Omega_b vs H0
plt.subplot(1, 2, 2)
plt.scatter(H0_sample, Omegab_sample, s=1, alpha=0.5)
plt.axhline(0.05, color='r', linestyle='--', label="True $\\Omega_b$=0.05")
plt.xlabel("$H_0$")
plt.ylabel("$\\Omega_b$")
plt.legend()

plt.tight_layout()
plt.show()

# Compute mean and standard deviation of the parameters
omegab_mean = np.mean(Omegab_sample)
omegab_std = np.std(Omegab_sample)
h0_mean = np.mean(H0_sample)
h0_std = np.std(H0_sample)

print(f"Estimated Omegab: {omegab_mean:.4f} ± {omegab_std:.4f}")
print(f"Estimated H0: {h0_mean:.2f} ± {h0_std:.2f} km/s/Mpc")