import numpy as np
import matplotlib.pyplot as plt
from numcosmo_py import Nc, Ncm

# Initialize the NumCosmo library
Ncm.cfg_init()

# Define the likelihood function using NumCosmo and luminosity distance
def likelihood(Omegab, H0):
    # Create a cosmological model
    cosmo = Nc.HICosmoDEXcdm.new()
    cosmo.props.Omegab = Omegab
    cosmo.props.H0 = H0
    # Compute the luminosity distance
    dist = Nc.Distance.new(1.5)
    dist.prepare(cosmo)
    z = np.linspace(0.01, 1.5, len(data))  # Redshift values for the data points
    d_lum = np.array([dist.luminosity(cosmo, z_i) for z_i in z])  # Luminosity distance in Mpc

    # Create a data point for comparison
    return np.exp(-0.5 * np.sum((data - d_lum) ** 2))

# Generate some fake data for demonstration purposes
np.random.seed(42)
true_Omegab = 0.3
true_H0 = 70
z = np.linspace(0.01, 1.5, 30)
true_cosmo = Nc.HICosmoDEXcdm.new()
true_cosmo.props.Omegab = true_Omegab
true_cosmo.props.H0 = true_H0
dist = Nc.Distance.new(1.5)
dist.prepare(true_cosmo)
data = np.array([dist.luminosity(true_cosmo, z_i) for z_i in z]) + np.random.normal(0, 0.1, size=30)

iterations = 2500
# Metropolis-Hastings algorithm 
def Metropolis_Hastings(iterations, init_Omegab=true_Omegab, init_H0=true_H0, step_size=0.2):
    H0_chain = [init_H0]
    Omegab_chain = [init_Omegab]
    likelihood_chain = [likelihood(init_Omegab, init_H0)]

    for _ in range(iterations):
        new_H0 =  np.random.normal(H0_chain[-1], step_size)
        new_Omegab = np.random.normal(Omegab_chain[-1], step_size)
        
        likelihood_new = likelihood(new_Omegab, new_H0)
        likelihood_old = likelihood_chain[-1]
        accept = likelihood_new/likelihood_old

        if np.random.rand() <= accept:
            H0_chain.append(new_H0)
            Omegab_chain.append(new_Omegab)
            likelihood_chain.append(likelihood_new)
        else:
            H0_chain.append(H0_chain[-1])
            Omegab_chain.append(Omegab_chain[-1])
            likelihood_chain.append(likelihood_old)

    return np.array(H0_chain), np.array(Omegab_chain)

H0_sample, Omegab_sample = Metropolis_Hastings(iterations)

# Plot results
plt.figure(figsize=(12, 5))

# Histogram of H0
plt.subplot(1, 2, 1)
plt.hist(H0_sample, bins=20, density=True, alpha=0.7)
plt.axvline(true_H0, color='r', linestyle='--', label=f"True H0={true_H0}")
plt.xlabel("$H_0$")
plt.ylabel("Density")
plt.legend()

# Scatter plot of Omega_m vs H0
plt.subplot(1, 2, 2)
plt.scatter(H0_sample, Omegab_sample, s=1, alpha=0.5)
plt.axhline(true_Omegab, color='r', linestyle='--', label=f"True $\\Omega_m$={true_Omegab}")
plt.xlabel("$H_0$")
plt.ylabel("$\\Omega_m$")
plt.legend()

plt.tight_layout()
plt.show()
