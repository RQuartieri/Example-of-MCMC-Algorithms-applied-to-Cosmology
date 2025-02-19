import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
import emcee
from multiprocessing import Pool
import time

start_time = time.time()

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
z_data = np.linspace(0.01, 2.0, 100)  # Redshifts of supernovae
true_H0 = 70
true_Omega_m = 0.3
mu_data = np.array([distance_modulus(z, true_H0, true_Omega_m) for z in z_data])
mu_data += np.random.normal(0, 0.2, size=len(mu_data))  # Adding noise

# Log-likelihood function
def log_likelihood(theta):
    H0, Omega_m = theta
    if H0 <= 0 or Omega_m < 0 or Omega_m > 1:
        return -np.inf  # Reject unphysical values

    mu_model = np.array([distance_modulus(z, H0, Omega_m) for z in z_data])
    chi2 = np.sum((mu_data - mu_model) ** 2 / 0.2**2)
    return -0.5 * chi2

# Run emcee with parallelization
nwalkers = 30
ndim = 2

# Initialize the sampler
initial_H0 = np.random.normal(70, 10, nwalkers)  # Centered around 70 with some spread
initial_Omega_m = np.random.normal(0.3, 0.1, nwalkers)  # Centered around 0.3 with some spread
initial_positions = np.vstack((initial_H0, initial_Omega_m)).T

with Pool() as pool:
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_likelihood, pool=pool)
    nsteps = 10000  # Number of steps to run the sampler
    sampler.run_mcmc(initial_positions, nsteps, progress=True)

# Extract the samples
samples = sampler.get_chain(flat=True)
burnin = int(0.2 * len(samples))  # Discard first 20% of samples
samples = samples[burnin:]
H0_samples = samples[:, 0]
Omega_m_samples = samples[:, 1]

# Compute statistics
H0_mean = np.mean(H0_samples)
H0_std = np.std(H0_samples)
Omega_m_mean = np.mean(Omega_m_samples)
Omega_m_std = np.std(Omega_m_samples)

# Compute autocorrelation time
tau = sampler.get_autocorr_time(tol=0)
tau_H0, tau_Omega_m = tau[0], tau[1]

# Compute the effective sample size (nESS)
nESS_H0 = len(H0_samples) / (1 + 2 * tau_H0)
nESS_Omega_m = len(Omega_m_samples) / (1 + 2 * tau_Omega_m)


# Plot results
plt.figure(figsize=(18, 5))

# Histogram of H0
plt.subplot(1, 3, 1)
plt.hist(H0_samples, bins=30, density=True, alpha=0.7, color='blue', label="H0 Samples")
plt.axvline(true_H0, color='r', linestyle='--', label=f"True H0={true_H0}")
plt.axvline(H0_mean, color='k', linestyle='-', label=f"Estimated H0={H0_mean:.2f} ± {H0_std:.2f}")
plt.xlabel("$H_0$ (km/s/Mpc)")
plt.ylabel("Density")
plt.legend()
plt.title(f"$H_0$ Distribution via emcee (Autocorrelation Time: {tau_H0:.2f})")

# Add additional information as text
plt.text(0.05, 0.95, f"Sample Size: {len(H0_samples)}\nESS: {nESS_H0:.2f}",
         transform=plt.gca().transAxes, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8))

# Scatter plot of Omega_m vs H0
plt.subplot(1, 3, 2)
plt.scatter(H0_samples, Omega_m_samples, s=1, alpha=0.5, color='green', label="Samples")
plt.axhline(true_Omega_m, color='r', linestyle='--', label=f"True $\\Omega_m$={true_Omega_m}")
plt.axhline(Omega_m_mean, color='k', linestyle='-', label=f"Estimated $\\Omega_m$={Omega_m_mean:.2f} ± {Omega_m_std:.2f}")
plt.axvline(true_H0, color='r', linestyle='--', label=f"True H0={true_H0}")
plt.axvline(H0_mean, color='k', linestyle='-', label=f"Estimated H0={H0_mean:.2f} ± {H0_std:.2f}")
plt.xlabel("$H_0$ (km/s/Mpc)")
plt.ylabel("$\\Omega_m$")
plt.legend()
plt.title(f"$H_0$ vs $\\Omega_m$ (Autocorrelation Time: {tau_H0:.2f})")

# Add additional information as text
plt.text(0.05, 0.95, f"Sample Size: {len(H0_samples)}\nESS (H0): {nESS_H0:.2f}\nESS ($\\Omega_m$): {nESS_Omega_m:.2f}",
         transform=plt.gca().transAxes, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8))

# Histogram of Omega_m
plt.subplot(1, 3, 3)
plt.hist(Omega_m_samples, bins=30, density=True, alpha=0.7, color='orange', label="$\\Omega_m$ Samples")
plt.axvline(true_Omega_m, color='r', linestyle='--', label=f"True $\\Omega_m$={true_Omega_m}")
plt.axvline(Omega_m_mean, color='k', linestyle='-', label=f"Estimated $\\Omega_m$={Omega_m_mean:.2f} ± {Omega_m_std:.2f}")
plt.xlabel("$\\Omega_m$")
plt.ylabel("Density")
plt.legend()
plt.title(f"$\\Omega_m$ Distribution via emcee (Autocorrelation Time: {tau_Omega_m:.2f})")

# Add additional information as text
plt.text(0.05, 0.95, f"Sample Size: {len(Omega_m_samples)}\nESS: {nESS_Omega_m:.2f}",
         transform=plt.gca().transAxes, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig("FinalProject/figs/parameter_distribution_emcee.png", dpi=300)
#plt.show()

# Plot the chains
fig, axes = plt.subplots(2, figsize=(10, 7), sharex=True)
samples_chain = sampler.get_chain()
labels = ["H0", "Omega_m"]
for i in range(ndim):
    ax = axes[i]
    ax.plot(samples_chain[:, :, i], "k", alpha=0.3)
    ax.set_xlim(0, len(samples_chain))
    ax.set_ylabel(labels[i])
    ax.yaxis.set_label_coords(-0.1, 0.5)

axes[-1].set_xlabel("Step number")
plt.savefig("FinalProject/figs/chains_emcee.png", dpi=300)
#plt.show()


end_time = time.time()
execution_time = end_time - start_time
print(f"Tempo de execução: {execution_time:.2f} segundos")