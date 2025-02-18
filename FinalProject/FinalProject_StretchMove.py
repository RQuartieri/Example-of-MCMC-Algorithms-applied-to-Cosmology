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
def log_likelihood(theta):
    H0, Omega_m = theta
    if H0 <= 0 or Omega_m < 0 or Omega_m > 1:
        return -np.inf  # Reject unphysical values

    mu_model = np.array([distance_modulus(z, H0, Omega_m) for z in z_data])
    chi2 = np.sum((mu_data - mu_model) ** 2 / 0.2**2)
    return -0.5 * chi2

# Stretch Move sampling
def stretch_move_sampling(iterations, n_walkers, init_H0=70, init_Omega_m=0.3):
    # Initialize walkers
    ndim = 2  # Number of parameters (H0, Omega_m)
    walkers = np.zeros((n_walkers, ndim))
    walkers[:, 0] = init_H0 + 1e-4 * np.random.randn(n_walkers)  # H0
    walkers[:, 1] = init_Omega_m + 1e-4 * np.random.randn(n_walkers)  # Omega_m

    # Initialize chains
    H0_chain = []
    Omega_m_chain = []

    # Stretch Move parameters
    a = 2.0  # Scale factor for the stretch move

    for i in range(iterations):
        for k in range(n_walkers):
            # Select a random walker (excluding the current one)
            j = np.random.randint(n_walkers)
            while j == k:
                j = np.random.randint(n_walkers)

            # Generate a random stretch factor
            z = np.random.uniform(1/a, a)
            g = 1/np.sqrt(z)

            # Propose a new position
            new_theta = walkers[j] + z * (walkers[k] - walkers[j])

            # Compute the log-likelihood ratio
            logL_new = log_likelihood(new_theta)
            logL_old = log_likelihood(walkers[k])
            accept_ratio = g**(ndim - 1) * np.exp(logL_new - logL_old)

            # Accept or reject the proposal
            if np.random.rand() < accept_ratio:
                walkers[k] = new_theta
            else:
                walkers[k] = walkers[k]

        # Save the current state of the walkers
        H0_chain.extend(walkers[:, 0])
        Omega_m_chain.extend(walkers[:, 1])

    return np.array(H0_chain), np.array(Omega_m_chain)

def autocorrelation_time(chain, max_lag=None):
    """
    Calculate the autocorrelation time for an MCMC chain.

    Parameters:
        chain (ndarray): The MCMC chain for a single parameter (1D array).
        max_lag (int): Maximum lag to compute the autocorrelation. If None, use len(chain)//2.

    Returns:
        float: The autocorrelation time.
    """
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
    truncate_lag = np.where(np.abs(autocorr) < 0.05)[0]
    if len(truncate_lag) > 0:
        max_lag = truncate_lag[0]

    # Calculate the autocorrelation time
    tau = 1 + 2 * np.sum(autocorr[:max_lag])
    return tau

# Run MCMC
iterations = 1000
n_walkers = 32
H0_samples, Omega_m_samples = stretch_move_sampling(iterations, n_walkers)

# Discard burn-in (first 1000 samples)
H0_samples = H0_samples[1000:]
Omega_m_samples = Omega_m_samples[1000:]

# Calculate mean and standard deviation for H0 and Omega_m
H0_mean = np.mean(H0_samples)
H0_std = np.std(H0_samples)
Omega_m_mean = np.mean(Omega_m_samples)
Omega_m_std = np.std(Omega_m_samples)

# Calculate the autocorrelation time
tau_H0 = autocorrelation_time(H0_samples)
tau_Omega_m = autocorrelation_time(Omega_m_samples)

# Plot results
plt.figure(figsize=(18, 5))

# Histogram of H0
plt.subplot(1, 3, 1)
plt.hist(H0_samples, bins=30, density=True, alpha=0.7)
plt.axvline(true_H0, color='r', linestyle='--', label=f"True H0={true_H0}")
plt.axvline(H0_mean, color='k', linestyle='-', label=f"Estimated H0={H0_mean:.2f} ± {H0_std:.2f}")
plt.xlabel("$H_0$")
plt.ylabel("Density")
plt.legend()
plt.title(f"Autocorrelation time (H0): {tau_H0:.2f}")

# Scatter plot of Omega_m vs H0
plt.subplot(1, 3, 2)
plt.scatter(H0_samples, Omega_m_samples, s=1, alpha=0.5)
plt.axhline(true_Omega_m, color='r', linestyle='--', label=f"True $\\Omega_m$={true_Omega_m}")
plt.axhline(Omega_m_mean, color='k', linestyle='-', label=f"Estimated $\\Omega_m$={Omega_m_mean:.2f} ± {Omega_m_std:.2f}")
plt.xlabel("$H_0$")
plt.ylabel("$\\Omega_m$")
plt.legend()
plt.title(f"Autocorrelation time (H0): {tau_H0:.2f}")

# Histogram of Omega_m
plt.subplot(1, 3, 3)
plt.hist(Omega_m_samples, bins=30, density=True, alpha=0.7, color='orange')
plt.axvline(true_Omega_m, color='r', linestyle='--', label=f"True $\\Omega_m$={true_Omega_m}")
plt.axvline(Omega_m_mean, color='k', linestyle='-', label=f"Estimated $\\Omega_m$={Omega_m_mean:.2f} ± {Omega_m_std:.2f}")
plt.xlabel("$\\Omega_m$")
plt.ylabel("Density")
plt.legend()
plt.title(f"Autocorrelation time ($\\Omega_m$): {tau_Omega_m:.2f}")

plt.tight_layout()
plt.savefig("FinalProject/figs/parameter_distribution_StretchMove.png", dpi=300)
#plt.show()