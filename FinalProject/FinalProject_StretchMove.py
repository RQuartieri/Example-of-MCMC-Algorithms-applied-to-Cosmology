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


def stretch_move_sampling(iterations, n_walkers, init_H0=70, init_Omega_m=0.3):
    # Initialize walkers
    ndim = 2  # Number of parameters (H0, Omega_m)
    walkers = np.zeros((n_walkers, ndim))
    walkers[:, 0] = init_H0 + 1e-4 * np.random.randn(n_walkers)  # H0
    walkers[:, 1] = init_Omega_m + 1e-4 * np.random.randn(n_walkers)  # Omega_m

    # Initialize chains for each walker
    H0_chains = np.zeros((iterations, n_walkers))  # Shape: (iterations, n_walkers)
    Omega_m_chains = np.zeros((iterations, n_walkers))  # Shape: (iterations, n_walkers)

    # Stretch Move parameters
    a = 2.0  # Scale factor for the stretch move

    for i in range(iterations):
        for k in range(n_walkers):
            # Select a random walker (excluding the current one)
            j = np.random.randint(n_walkers)
            while j == k:
                j = np.random.randint(n_walkers)

            # Generate a random stretch factor
            z = (a - 1) * np.random.rand() + 1
            z = z**2 / a  # Scale the stretch factor

            # Propose a new position
            y = walkers[j] + z * (walkers[k] - walkers[j])

            # Compute the log-likelihood ratio
            logL_new = log_likelihood(y)
            logL_old = log_likelihood(walkers[k])
            accept_ratio = z**(ndim - 1) * np.exp(logL_new - logL_old)

            # Accept or reject the proposal
            if np.random.rand() < accept_ratio:
                walkers[k] = y
            else:
                walkers[k] = walkers[k]

        # Save the current state of the walkers
        H0_chains[i, :] = walkers[:, 0]  # Save all walkers' H0 values
        Omega_m_chains[i, :] = walkers[:, 1]  # Save all walkers' Omega_m values

    return H0_chains, Omega_m_chains

def autocorrelation_time(chain, max_lag=None, truncate_threshold=0.01):
    n = len(chain)
    if max_lag is None:
        max_lag = n // 2

    # Normalize the chain
    chain = chain - np.mean(chain)

    # Compute the autocorrelation function
    autocorr = np.zeros(max_lag)
    for t in range(max_lag):
        autocorr[t] = np.sum(chain[:n - t] * chain[t:]) / np.sum(chain**2)

    # Truncate the sum when autocorrelation becomes negligible
    truncate_lag = np.where(np.abs(autocorr) < truncate_threshold)[0]
    if len(truncate_lag) > 0:
        max_lag = truncate_lag[0]

#     # Calculate the autocorrelation time
    tau = 1 + 2 * np.sum(autocorr[:max_lag])

    return tau

def effective_sample_size(chain, max_lag=None, truncate_threshold=0.01):
    n = len(chain)
    tau = autocorrelation_time(chain, max_lag, truncate_threshold)
    ess = n / (1 + 2 * tau)
    return ess


# Run MCMC
iterations = 10000
n_walkers = 30
H0_chains, Omega_m_chains = stretch_move_sampling(iterations, n_walkers)

# Discard burn-in (first 20% of the chain)
burn_in = iterations // 5
H0_chains = H0_chains[burn_in:, :]
Omega_m_chains = Omega_m_chains[burn_in:, :]

# Compute autocorrelation time for each walker
tau_H0 = np.mean([autocorrelation_time(H0_chains[:, w]) for w in range(n_walkers)])
tau_Omega_m = np.mean([autocorrelation_time(Omega_m_chains[:, w]) for w in range(n_walkers)])

print(f"Autocorrelation time for H0: {tau_H0}")
print(f"Autocorrelation time for Omega_m: {tau_Omega_m}")

# Compute ESS for each walker's chain
ess_H0 = [effective_sample_size(H0_chains[:, w]) for w in range(n_walkers)]
ess_Omega_m = [effective_sample_size(Omega_m_chains[:, w]) for w in range(n_walkers)]
avg_ess_H0 = np.mean(ess_H0)
avg_ess_Omega_m = np.mean(ess_Omega_m)
print(f"Average ESS for H0: {avg_ess_H0}")
print(f"Average ESS for Omega_m: {avg_ess_Omega_m}")

# Flatten the chains
H0_samples = H0_chains.flatten()  # Combine all walkers' chains
Omega_m_samples = Omega_m_chains.flatten()  # Combine all walkers' chains
print(f"Final sample size H0: {len(H0_samples)}")
print(f"Final sample size Omega_m: {len(Omega_m_samples)}")

# Calculate mean and standard deviation for H0 and Omega_m
H0_mean = np.mean(H0_samples)
H0_std = np.std(H0_samples)
Omega_m_mean = np.mean(Omega_m_samples)
Omega_m_std = np.std(Omega_m_samples)


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
plt.title(f"$H_0$ Distribution via Stretch Move (Autocorrelation Time: {tau_H0:.2f})")

# Add additional information as text
plt.text(0.05, 0.95, f"Sample Size: {len(H0_samples)}\nESS: {avg_ess_H0:.2f}",
         transform=plt.gca().transAxes, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8))

# Scatter plot of Omega_m vs H0
plt.subplot(1, 3, 2)
plt.scatter(H0_samples, Omega_m_samples, s=1, alpha=0.5, color='green', label="Samples")
plt.axhline(true_Omega_m, color='r', linestyle='--', label=f"True $\\Omega_m$={true_Omega_m}")
plt.axhline(Omega_m_mean, color='k', linestyle='-', label=f"Estimated $\\Omega_m$={Omega_m_mean:.2f} ± {Omega_m_std:.2f}")
plt.axvline(true_H0, color='r', linestyle='--', label=f"True H0={true_H0}")
plt.axvline(H0_mean, color='k', linestyle='-', label=f"Estimated H0={H0_mean:.2f} ± {H0_std:.2f}")
plt.xlabel("$H_0$")
plt.ylabel("$\\Omega_m$")
plt.legend()
plt.title(f"$H_0$ vs $\\Omega_m$ (Autocorrelation Time: {tau_H0:.2f})")

# Add additional information as text
plt.text(0.05, 0.95, f"Sample Size: {len(H0_samples)}\nESS (H0): {avg_ess_H0:.2f}\nESS ($\\Omega_m$): {avg_ess_Omega_m:.2f}",
         transform=plt.gca().transAxes, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8))

# Histogram of Omega_m
plt.subplot(1, 3, 3)
plt.hist(Omega_m_samples, bins=30, density=True, alpha=0.7, color='orange', label="$\\Omega_m$ Samples")
plt.axvline(true_Omega_m, color='r', linestyle='--', label=f"True $\\Omega_m$={true_Omega_m}")
plt.axvline(Omega_m_mean, color='k', linestyle='-', label=f"Estimated $\\Omega_m$={Omega_m_mean:.2f} ± {Omega_m_std:.2f}")
plt.xlabel("$\\Omega_m$")
plt.ylabel("Density")
plt.legend()
plt.title(f"$\\Omega_m$ Distribution via Stretch Move (Autocorrelation Time: {tau_Omega_m:.2f})")

# Add additional information as text
plt.text(0.05, 0.95, f"Sample Size: {len(Omega_m_samples)}\nESS: {avg_ess_Omega_m:.2f}",
         transform=plt.gca().transAxes, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig("FinalProject/figs/parameter_distribution_StretchMove.png", dpi=300)
#plt.show()