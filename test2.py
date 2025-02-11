import numpy as np
import emcee
import matplotlib.pyplot as plt

# Generate synthetic data from a Gaussian distribution
np.random.seed(42)
true_mu = 3.0
true_sigma = 1.5
data = np.random.normal(true_mu, true_sigma, 1000)

# Define the log-likelihood function
def log_likelihood(theta, data):
    mu, sigma = theta
    if sigma <= 0:
        return -np.inf  # Log of a non-positive sigma is undefined
    return -0.5 * np.sum(((data - mu) / sigma) ** 2 + np.log(2 * np.pi * sigma ** 2))

# Define the log-prior function
def log_prior(theta):
    mu, sigma = theta
    if 0 < sigma < 10:  # Uniform prior for sigma
        return 0.0
    return -np.inf

# Define the log-posterior function
def log_posterior(theta, data):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, data)

# Set up the MCMC sampler
ndim = 2  # Number of parameters (mu and sigma)
nwalkers = 10  # Number of walkers
initial_positions = [true_mu + 0.1 * np.random.randn(ndim) for _ in range(nwalkers)]

# Initialize the sampler
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, args=[data])

# Run the MCMC sampler
nsteps = 5000
sampler.run_mcmc(initial_positions, nsteps, progress=True)

# Extract the samples
discard = 1000  # Burn-in period
samples = sampler.get_chain(discard=discard, flat=True)

# Plot the results
fig, axes = plt.subplots(2, figsize=(8, 6), sharex=True)
axes[0].hist(samples[:, 0], bins=30, alpha=0.6, color='blue', label='Posterior of mu')
axes[0].axvline(true_mu, color='red', linestyle='dashed', label='True mu')
axes[0].legend()
axes[1].hist(samples[:, 1], bins=30, alpha=0.6, color='green', label='Posterior of sigma')
axes[1].axvline(true_sigma, color='red', linestyle='dashed', label='True sigma')
axes[1].legend()
plt.xlabel("Parameter value")
plt.show()
