import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

# Constants
c = 299792.458  # Speed of light in km/s

# Luminosity distance function
def luminosity_distance(z, H0, Omega_m):
    def E_inv(z):
        return 1.0 / np.sqrt(Omega_m * (1 + z)**3 + (1 - Omega_m))
   
    integral, _ = quad(E_inv, 0, z)
    return (c / H0) * (1 + z) * integral

# Distance modulus function
def distance_modulus(z, H0, Omega_m):
    d_L = luminosity_distance(z, H0, Omega_m)
    return 5 * np.log10(d_L) + 25

# Generate synthetic data (mock supernova observations)
np.random.seed(42)
z_data = np.linspace(0.01, 1.5, 30)  # Redshifts of supernovae
true_H0 = 70
true_Omega_m = 0.3
mu_data = np.array([distance_modulus(z, true_H0, true_Omega_m) for z in z_data])
mu_data += np.random.normal(0, 0.2, size=len(mu_data))  # Adding noise

# Log-likelihood function
def log_likelihood(H0, Omega_m):
    if H0 <= 0 or Omega_m < 0 or Omega_m > 1:
        return -np.inf  # Reject unphysical values
   
    mu_model = np.array([distance_modulus(z, H0, Omega_m) for z in z_data])
    chi2 = np.sum((mu_data - mu_model) ** 2 / 0.2**2)
    return -0.5 * chi2

# Metropolis-Hastings MCMC
def metropolis_hastings(iterations, init_H0=70, init_Omega_m=0.3, step_size=1.0):
    H0_chain = [init_H0]
    Omega_m_chain = [init_Omega_m]
    logL_chain = [log_likelihood(init_H0, init_Omega_m)]
   
    for _ in range(iterations):
        # Propose new parameters
        new_H0 = np.random.normal(H0_chain[-1], step_size)
        new_Omega_m = np.random.normal(Omega_m_chain[-1], step_size * 0.1)
       
        # Compute likelihood ratio
        logL_new = log_likelihood(new_H0, new_Omega_m)
        logL_old = logL_chain[-1]
        accept_ratio = np.exp(logL_new - logL_old)
       
        # Accept or reject step
        if np.random.rand() < accept_ratio:
            H0_chain.append(new_H0)
            Omega_m_chain.append(new_Omega_m)
            logL_chain.append(logL_new)
        else:
            H0_chain.append(H0_chain[-1])
            Omega_m_chain.append(Omega_m_chain[-1])
            logL_chain.append(logL_old)

    return np.array(H0_chain), np.array(Omega_m_chain)

# Run MCMC
iterations = 10000
H0_samples, Omega_m_samples = metropolis_hastings(iterations)

# Plot results
plt.figure(figsize=(12, 5))

# Histogram of H0
plt.subplot(1, 2, 1)
plt.hist(H0_samples, bins=30, density=True, alpha=0.7)
plt.axvline(true_H0, color='r', linestyle='--', label=f"True H0={true_H0}")
plt.xlabel("$H_0$")
plt.ylabel("Density")
plt.legend()

# Scatter plot of Omega_m vs H0
plt.subplot(1, 2, 2)
plt.scatter(H0_samples, Omega_m_samples, s=1, alpha=0.5)
plt.axhline(true_Omega_m, color='r', linestyle='--', label=f"True $\\Omega_m$={true_Omega_m}")
plt.xlabel("$H_0$")
plt.ylabel("$\\Omega_m$")
plt.legend()

plt.tight_layout()
plt.show()