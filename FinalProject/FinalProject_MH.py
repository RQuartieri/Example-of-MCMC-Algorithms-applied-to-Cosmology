import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
import time

start_time = time.time()
# Metropolis-Hastings

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

def autocorrelation_time(chain, max_lag=None):
    n = len(chain)
    if max_lag is None:
        max_lag = n // 2

    # Normalize the chain
    chain = chain - np.mean(chain)

    # Compute the autocorrelation function using FFT
    autocorr = np.correlate(chain, chain, mode='full')[n - 1:] / np.sum(chain**2)

    # Truncate the sum when autocorrelation becomes negligible
    truncate_lag = np.where(np.abs(autocorr) < 0.05)[0]
    if len(truncate_lag) > 0:
        max_lag = truncate_lag[0]

    # Calculate the autocorrelation time
    tau = 1 + 2 * np.sum(autocorr[:max_lag])
    return tau

def effective_sample_size(chain, max_lag=None):
    n = len(chain)
    tau = autocorrelation_time(chain, max_lag)  # Pass only chain and max_lag
    ess = n / (1 + 2 * tau)
    return ess

# Run MCMC
iterations = 299999
H0_sample, Omega_m_sample = metropolis_hastings(iterations)

print(f"Final sample size H0: {len(H0_sample)}")
print(f"Final sample size Omega_m: {len(Omega_m_sample)}")

# Discard the first 20% of the chains as burn-in
burn_in = int(0.2 * len(H0_sample))
H0_burnt = H0_sample[burn_in:]
Omega_m_burnt = Omega_m_sample[burn_in:]

# Calculate mean and standard deviation for H0 and Omega_m after burn-in
H0_mean = np.mean(H0_burnt)
H0_std = np.std(H0_burnt)
Omega_m_mean = np.mean(Omega_m_burnt)
Omega_m_std = np.std(Omega_m_burnt)

# Calculate autocorrelation time and ESS after burn-in
tau_H0 = autocorrelation_time(H0_burnt)
tau_Omega_m = autocorrelation_time(Omega_m_burnt)

ess_H0 = effective_sample_size(H0_burnt)
ess_Omega_m = effective_sample_size(Omega_m_burnt)

# Print results
print(f"Final sample size after burn-in (H0): {len(H0_burnt)}")
print(f"Final sample size after burn-in (Omega_m): {len(Omega_m_burnt)}")
print(f"Autocorrelation time (H0): {tau_H0:.2f}")
print(f"Autocorrelation time (Omega_m): {tau_Omega_m:.2f}")
print(f"Effective Sample Size (H0): {ess_H0:.2f}")
print(f"Effective Sample Size (Omega_m): {ess_Omega_m:.2f}")

# Plot results with burn-in
plt.figure(figsize=(18, 5))

# Histogram of H0
plt.subplot(1, 3, 1)
plt.hist(H0_burnt, bins=30, density=True, alpha=0.7, color='blue', label="H0 Samples")
plt.axvline(true_H0, color='r', linestyle='--', label=f"True H0={true_H0}")
plt.axvline(H0_mean, color='k', linestyle='-', label=f"Estimated H0={H0_mean:.2f} ± {H0_std:.2f}")
plt.xlabel("$H_0$ (km/s/Mpc)")
plt.ylabel("Density")
plt.legend()
plt.title(f"$H_0$ Distribution via MH (Autocorrelation Time: {tau_H0:.2f})")

# Add additional information as text
plt.text(0.05, 0.95, f"Sample Size: {len(H0_burnt)}\nESS: {ess_H0:.2f}",
         transform=plt.gca().transAxes, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8))

# Histogram of Omega_m
plt.subplot(1, 3, 3)
plt.hist(Omega_m_burnt, bins=30, density=True, alpha=0.7, color='orange', label="$\\Omega_m$ Samples")
plt.axvline(true_Omega_m, color='r', linestyle='--', label=f"True $\\Omega_m$={true_Omega_m}")
plt.axvline(Omega_m_mean, color='k', linestyle='-', label=f"Estimated $\\Omega_m$={Omega_m_mean:.2f} ± {Omega_m_std:.2f}")
plt.xlabel("$\\Omega_m$")
plt.ylabel("Density")
plt.legend()
plt.title(f"$\\Omega_m$ Distribution via MH (Autocorrelation Time: {tau_Omega_m:.2f})")

# Add additional information as text
plt.text(0.05, 0.95, f"Sample Size: {len(Omega_m_burnt)}\nESS: {ess_Omega_m:.2f}",
         transform=plt.gca().transAxes, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8))

# Scatter plot of Omega_m vs H0
plt.subplot(1, 3, 2)
plt.scatter(H0_burnt, Omega_m_burnt, s=1, alpha=0.5, color='green', label="Samples")
plt.axhline(true_Omega_m, color='r', linestyle='--', label=f"True $\\Omega_m$={true_Omega_m}")
plt.axhline(Omega_m_mean, color='k', linestyle='-', label=f"Estimated $\\Omega_m$={Omega_m_mean:.2f} ± {Omega_m_std:.2f}")
plt.axvline(true_H0, color='r', linestyle='--', label=f"True H0={true_H0}")
plt.axvline(H0_mean, color='k', linestyle='-', label=f"Estimated H0={H0_mean:.2f} ± {H0_std:.2f}")
plt.xlabel("$H_0$")
plt.ylabel("$\\Omega_m$")
plt.legend()
plt.title(f"$H_0$ vs $\\Omega_m$ (Autocorrelation Time: {tau_H0:.2f})")

# Add additional information as text
plt.text(0.05, 0.95, f"Sample Size: {len(H0_burnt)}\nESS (H0): {ess_H0:.2f}\nESS ($\\Omega_m$): {ess_Omega_m:.2f}",
         transform=plt.gca().transAxes, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig("FinalProject/figs/parameter_distribution_MH_burnin.png", dpi=300)
#plt.show()


end_time = time.time()
execution_time = end_time - start_time
print(f"Tempo de execução: {execution_time:.2f} segundos")