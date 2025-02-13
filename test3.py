import numpy as np
import matplotlib.pyplot as plt

# Define the likelihood function (simplified example)
def likelihood(data, omega_m, h0):
    model = omega_m * h0 + (1 - omega_m) * h0 * np.random.normal(0, 0.1, size=len(data))
    return np.exp(-0.5 * np.sum((data - model))) ** 2

# Define the prior distribution (uniform prior in this example)
def prior(omega_m, h0):
    if 0 < omega_m < 1 and 50 < h0 < 100:
        return 1.0
    else:
        return 0.0

# Define the posterior distribution
def posterior(data, omega_m, h0):
    return likelihood(data, omega_m, h0) * prior(omega_m, h0)

# Generate some fake data for demonstration purposes
np.random.seed(1)
true_omega_m = 0.3
true_h0 = 70
data = true_omega_m * true_h0 + np.random.normal(0, 5, size=100)

# Metropolis-Hastings algorithm with debug prints
def metropolis_hastings(data, initial_params, iterations, step_size):
    omega_m_samples = []
    h0_samples = []
    omega_m, h0 = initial_params
    current_posterior = posterior(data, omega_m, h0)

    for i in range(iterations):
        # Propose new parameters within bounds
        omega_m_new = np.clip(np.random.normal(omega_m, step_size), 0, 1)
        h0_new = np.clip(np.random.normal(h0, step_size), 50, 100)
        new_posterior = posterior(data, omega_m_new, h0_new)

        # Acceptance ratio
        acceptance_ratio = new_posterior / current_posterior

        # Print the debug information
        print(f"Iteration {i+1}")
        print(f"Proposed omega_m: {omega_m_new}, Proposed h0: {h0_new}")
        print(f"New Posterior: {new_posterior}, Current Posterior: {current_posterior}")
        print(f"Acceptance Ratio: {acceptance_ratio}\n")

        # Accept or reject the new parameters
        if np.random.rand() < acceptance_ratio:
            omega_m = omega_m_new
            h0 = h0_new
            current_posterior = new_posterior

        omega_m_samples.append(omega_m)
        h0_samples.append(h0)

    return np.array(omega_m_samples), np.array(h0_samples)

# Parameters
initial_params = [0.5, 80]  # Initial guess
iterations = 100  # Reduced for debugging purposes
step_size = 0.1

# Run the Metropolis-Hastings algorithm with debug prints
omega_m_samples, h0_samples = metropolis_hastings(data, initial_params, iterations, step_size)


# Plot the results
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.hist(omega_m_samples, bins=50, density=True, alpha=0.6, color='b')
plt.xlabel('Omega_m')
plt.ylabel('Density')

plt.subplot(1, 2, 2)
plt.hist(h0_samples, bins=50, density=True, alpha=0.6, color='r')
plt.xlabel('H0')
plt.ylabel('Density')

plt.tight_layout()
plt.show()

