import numpy as np
import matplotlib.pyplot as plt
from numcosmo_py import Nc, Ncm

# Initialize the NumCosmo library
Ncm.cfg_init()

# Define the likelihood function using NumCosmo
def likelihood(data, Omegab, H0):
    # Create a cosmological model
    model = Nc.HICosmoDEXcdm.new()
    model.props.Omegab = Omegab
    model.props.H0 = H0
    
    # Create a data point for comparison
    model_data = model.props.Omegab * model.props.H0  # Simplified model calculation
    return np.exp(-0.5 * np.sum((data - model_data) ** 2))

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
data = true_Omegab * true_H0 + np.random.normal(0, 5, size=100)

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
        print(f"Iteration {i+1}")
        print(f"Proposed Omegab: {Omegab_new}, Proposed H0: {H0_new}")
        print(f"New Posterior: {new_posterior}, Current Posterior: {current_posterior}")
        print(f"Acceptance Ratio: {acceptance_ratio}\n")

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
iterations = 100  # Reduced for debugging purposes
step_size = 0.5

# Run the Metropolis-Hastings algorithm with debug prints
Omegab_samples, H0_samples = metropolis_hastings(data, initial_params, iterations, step_size)

# Plot the results
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.hist(Omegab_samples, bins=100, density=True, alpha=0.6, color='b')
plt.xlabel('Omegab')
plt.ylabel('Density')

plt.subplot(1, 2, 2)
plt.hist(H0_samples, bins=50, density=True, alpha=0.6, color='r')
plt.xlabel('H0')
plt.ylabel('Density')

plt.tight_layout()
plt.show()
