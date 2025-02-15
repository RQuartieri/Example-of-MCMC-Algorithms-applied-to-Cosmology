import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd  # Import pandas for DataFrame functionality
from numcosmo_py import Nc, Ncm
import emcee 


# Initialize the NumCosmo library
Ncm.cfg_init()

# Define the path to the mock data file
mock_data_path = os.path.join(os.path.dirname(__file__), "mock_supernova_data.txt")

# Check if the file exists
if not os.path.exists(mock_data_path):
    raise FileNotFoundError(f"Mock data file not found at: {mock_data_path}")

# Load the mock data
mock_data = np.loadtxt(mock_data_path)
z = mock_data[:, 0]  # Redshift
d_lum = mock_data[:, 1]  # Luminosity distance
error = mock_data[:, 2]  # Error in luminosity distance

# Create a cosmological model (ΛCDM)
cosmo = Nc.HICosmoDEXcdm.new()

# Set initial values for parameters
param_init = {
    "H0": 70.0,
    "Omegab": 0.05,
    "Omegac": 0.25,
    "Omegax": 0.7,
}

# Set the initial values in the cosmology object
for param, value in param_init.items():
    setattr(cosmo.props, param, value)

# Print the initial parameters
print("# Initial model parameters: ")
cosmo.params_log_all()

# Define a custom likelihood function
class CustomLikelihood:
    def __init__(self, z, d_lum, error, cosmo):
        self.z = z
        self.d_lum = d_lum
        self.error = error
        self.cosmo = cosmo
        self.dist = Nc.Distance.new(2.0)  # Distance object for calculations
        self.dist.prepare(cosmo)  # Prepare the distance object for the cosmology

    def log_likelihood(self, params):
        """Compute the log-likelihood for a given set of parameters."""
        H0, Omegab, Omegac = params
        Omegax = 1.0 - Omegab - Omegac  # Enforce the constraint Omegab + Omegac + Omegax = 1

        # Check parameter constraints
        if Omegab < 0 or Omegab > 1 or Omegac < 0 or Omegac > 1 or Omegax < 0 or Omegax > 1:
            return -np.inf  # Reject the proposal if constraints are violated

        # Set parameters in the cosmology object
        self.cosmo.props.H0 = H0
        self.cosmo.props.Omegab = Omegab
        self.cosmo.props.Omegac = Omegac
        self.cosmo.props.Omegax = Omegax

        # Compute the chi-squared statistic
        chi2 = 0.0
        for z_i, d_lum_i, error_i in zip(self.z, self.d_lum, self.error):
            d_lum_model = self.dist.luminosity(self.cosmo, z_i)
            chi2 += ((d_lum_i - d_lum_model) / error_i) ** 2

        # Return the log-likelihood
        return -0.5 * chi2

# Create the custom likelihood
likelihood = CustomLikelihood(z, d_lum, error, cosmo)

# Define the prior function
def log_prior(params):
    H0, Omegab, Omegac = params
    # Uniform priors for H0, Omegab, and Omegac
    if 50 <= H0 <= 100 and 0 <= Omegab <= 1 and 0 <= Omegac <= 1:
        return 0.0  # Uniform prior
    return -np.inf  # Reject the proposal if outside the prior range

# Define the log-posterior function
def log_posterior(params):
    lp = log_prior(params)
    if not np.isfinite(lp):
        return -np.inf
    return lp + likelihood.log_likelihood(params)

# Set up the emcee sampler
nwalkers = 32  # Number of walkers
ndim = 3  # Number of parameters (H0, Omegab, Omegac)
nsteps = 10000  # Number of MCMC steps

# Initial positions for the walkers
initial_params = np.array([param_init["H0"], param_init["Omegab"], param_init["Omegac"]])
initial_positions = initial_params + 1e-4 * np.random.randn(nwalkers, ndim)

# Create the sampler
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior)

# Run the MCMC sampler
print("Running MCMC...")
sampler.run_mcmc(initial_positions, nsteps, progress=True)
print("MCMC complete.")

# Extract the chains
samples = sampler.get_chain()
samples = samples.reshape(-1, ndim)  # Flatten the chains

# Convert chains to a DataFrame for easier plotting
param_names = ["H0", "Omegab", "Omegac"]
samples_df = pd.DataFrame(samples, columns=param_names)

# Add Omegax to the DataFrame (computed from Omegab and Omegac)
samples_df["Omegax"] = 1.0 - samples_df["Omegab"] - samples_df["Omegac"]

# Compute mean and standard deviation for each parameter
param_estimates = {param: (np.mean(samples_df[param]), np.std(samples_df[param])) for param in param_names + ["Omegax"]}

# Plot 1: Marginal Distributions with Legends
plt.figure(figsize=(12, 8))
for i, param in enumerate(param_names + ["Omegax"]):
    plt.subplot(2, 2, i + 1)
    sns.histplot(samples_df[param], kde=True, color='blue', stat='density', label='Posterior')
    
    # Add vertical lines for initial and estimated values
    initial_value = param_init.get(param, 1.0 - param_init["Omegab"] - param_init["Omegac"])  # Handle Omegax
    estimated_value, _ = param_estimates[param]
    
    plt.axvline(initial_value, color='red', linestyle='--', label=f"Initial: {initial_value:.4f}")
    plt.axvline(estimated_value, color='green', linestyle='--', label=f"Estimated: {estimated_value:.4f}")
    
    plt.title(f"Distribution of {param}")
    plt.xlabel(param)
    plt.ylabel("Density")
    plt.legend()

plt.tight_layout()
plt.savefig("FinalProject/Figs/parameter_distributions_emcee.png", dpi=300)
#plt.show()

# Plot 2: Pairwise Correlations
plt.figure(figsize=(12, 8))
sns.pairplot(
    data=samples_df,
    kind='scatter',
    plot_kws={'alpha': 0.5, 'edgecolor': 'none'}
)
plt.suptitle("Pairwise Correlations", y=1.02)
plt.tight_layout()
plt.savefig("FinalProject/Figs/parameter_correlations_emcee.png", dpi=300) 
#plt.show()

# Print results
print("\nEstimated Cosmological Parameters:")
for param, (mean, std) in param_estimates.items():
    print(f"{param}: {mean:.4f} ± {std:.4f}")