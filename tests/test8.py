import numpy as np
import matplotlib.pyplot as plt
from numcosmo_py import Nc, Ncm

# Initialize the NumCosmo library
Ncm.cfg_init()

# Define the likelihood function using NumCosmo and luminosity distance
def likelihood(data, Omegab, H0):
    # Create a cosmological model
    cosmo = Nc.HICosmoDEXcdm.new()
    cosmo.props.Omegab = Omegab
    cosmo.props.H0 = H0

    # Compute the luminosity distance
    dist = Nc.Distance.new(2.0)
    dist.prepare(cosmo)
    z = np.linspace(0.01, 1.0, len(data))  # Redshift values for the data points
    d_lum = np.array([dist.luminosity(cosmo, z_i) for z_i in z])  # Luminosity distance in Mpc

    # Create a data point for comparison
    cosmo_data = d_lum
    return np.exp(-0.5 * np.sum((data - cosmo_data) ** 2))

# Define the prior distribution (uniform prior in this example)
def prior(Omegab, H0):
    if 0 < Omegab < 1 and 50 < H0 < 100:
        return 1.0
    else:
        return 0.0

# Define the posterior distribution
def posterior(data, Omegab, H0):
    return likelihood(data, Omegab, H0) * prior(Omegab, H0)

# Generate some fake data for demonstration purposes
np.random.seed(1)
true_Omegab = 0.3
true_H0 = 70
z = np.linspace(0.01, 1.0, 100)
true_cosmo = Nc.HICosmoDEXcdm.new()
true_cosmo.props.Omegab = true_Omegab
true_cosmo.props.H0 = true_H0
dist = Nc.Distance.new(2.0)
dist.prepare(true_cosmo)
data = np.array([dist.luminosity(true_cosmo, z_i) for z_i in z]) + np.random.normal(0, 0.1, size=100)

# Metropolis-Hastings algorithm with debug prints
def metropolis_hastings(data, initial_params, iterations, step_size):
    Omegab_samples = []
    H0_samples = []
    Omegab, H0 = initial_params
    current_posterior = posterior(data, Omegab, H0)

    for i in range(iterations):
        # Propose new parameters within bounds
        Omegab_new = np.clip(np.random.normal(Omegab, step_size), 0, 1)
        H0_new = np.clip(np.random.normal(H0, step_size), 0, 100)
        new_posterior = posterior(data, Omegab_new, H0_new)

        # Acceptance ratio
        acceptance_ratio = new_posterior / current_posterior

        # Print the debug information
        #print(f"Iteration {i+1}")
        #print(f"Proposed Omegab: {Omegab_new}, Proposed H0: {H0_new}")
        #print(f"New Posterior: {new_posterior}, Current Posterior: {current_posterior}")
        #print(f"Acceptance Ratio: {acceptance_ratio}\n")

        # Accept or reject the new parameters
        if np.random.rand() < acceptance_ratio:
            Omegab = Omegab_new
            H0 = H0_new
            current_posterior = new_posterior

        Omegab_samples.append(Omegab)
        H0_samples.append(H0)

    return np.array(Omegab_samples), np.array(H0_samples)

# Parameters
initial_params = [0.5, 80]  # Initial guess
iterations = 10000  # Increased number of iterations
step_size = 0.5

# Run the Metropolis-Hastings algorithm with debug prints
Omegab_samples, H0_samples = metropolis_hastings(data, initial_params, iterations, step_size)

# Plot the results
plt.figure(figsize=(24, 5))

# Plot the luminosity distance
plt.subplot(1, 4, 1)
plt.plot(z, np.array([dist.luminosity(true_cosmo, z_i) for z_i in z]), label='Luminosity Distance')
plt.xlabel('Redshift (z)')
plt.ylabel('Luminosity Distance (Mpc)')
plt.title('Luminosity Distance vs Redshift')
plt.legend()

# Plot the posterior distributions
plt.subplot(1, 4, 2)
plt.hist(Omegab_samples, bins=100, density=True, alpha=0.6, color='b')
plt.xlabel('Omegab')
plt.ylabel('Density')
plt.title('Posterior Distribution of Omegab')

plt.subplot(1, 4, 3)
plt.hist(H0_samples, bins=50, density=True, alpha=0.6, color='r')
plt.xlabel('H0')
plt.ylabel('Density')
plt.title('Posterior Distribution of H0')

# Plot the correlation between Omegab and H0
plt.subplot(1, 4, 4)
plt.scatter(Omegab_samples, H0_samples, alpha=0.6)
plt.xlabel('Omegab')
plt.ylabel('H0')
plt.title('Correlation between Omegab and H0')

plt.tight_layout()
plt.show()
