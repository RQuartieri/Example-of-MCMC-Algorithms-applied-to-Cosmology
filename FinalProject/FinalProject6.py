import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd  # Import pandas for DataFrame functionality
from numcosmo_py import Nc, Ncm

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

# Print the default parameters of the cosmological model
print("# Default model parameters: ")
cosmo.params_log_all()

# Define a custom likelihood function
class CustomLikelihood(Ncm.Likelihood):
    def __init__(self, z, d_lum, error, cosmo):
        super().__init__()
        self.z = z
        self.d_lum = d_lum
        self.error = error
        self.cosmo = cosmo
        self.dist = Nc.Distance.new(2.0)  # Distance object for calculations
        self.dist.prepare(cosmo)  # Prepare the distance object for the cosmology

    def do_eval(self, _):
        chi2 = 0.0
        for z_i, d_lum_i, error_i in zip(self.z, self.d_lum, self.error):
            # Compute the model-predicted luminosity distance
            d_lum_model = self.dist.luminosity(self.cosmo, z_i)
            # Compute the chi-squared contribution for this data point
            chi2 += ((d_lum_i - d_lum_model) / error_i) ** 2
        # Return the log-likelihood (-0.5 * chi^2)
        return -0.5 * chi2

# Create the custom likelihood
likelihood = CustomLikelihood(z, d_lum, error, cosmo)

# Metropolis-Hastings Algorithm
iterations = 100000
step_size = 0.01

# Store parameter chains
params_to_estimate = ["H0", "Omegab", "Omegac", "Omegax"]
param_chains = {param: [getattr(cosmo.props, param)] for param in params_to_estimate}
likelihood_chain = [likelihood.do_eval(None)]

def enforce_omega_constraints(params):
    """Enforce constraints on Omega parameters."""
    # Ensure 0 <= Omegab <= 1, 0 <= Omegac <= 1, and Omegab + Omegac <= 1
    if (
        params["Omegab"] < 0 or params["Omegab"] > 1 or
        params["Omegac"] < 0 or params["Omegac"] > 1 or
        (params["Omegab"] + params["Omegac"]) > 1
    ):
        return False  # Reject the proposal
    # Compute Omegax to satisfy the constraint Omegab + Omegac + Omegax = 1
    params["Omegax"] = 1.0 - params["Omegab"] - params["Omegac"]
    return True  # Accept the proposal

def metropolis_hastings(iterations):
    for _ in range(iterations):
        # Propose new parameters
        new_params = {param: np.random.normal(param_chains[param][-1], step_size) for param in params_to_estimate}
        
        # Enforce constraints on Omega parameters
        if not enforce_omega_constraints(new_params):
            # Reject the proposal if constraints are violated
            for param in param_chains:
                param_chains[param].append(param_chains[param][-1])
            likelihood_chain.append(likelihood_chain[-1])
            continue

        # Set new parameters in the cosmology object
        for param, value in new_params.items():
            setattr(cosmo.props, param, value)

        # Compute the new likelihood
        likelihood_new = likelihood.do_eval(None)
        likelihood_old = likelihood_chain[-1]
        accept = np.exp(likelihood_new - likelihood_old)

        # Accept or reject the proposal
        if np.random.rand() <= accept:
            for param in param_chains:
                param_chains[param].append(new_params[param])
            likelihood_chain.append(likelihood_new)
        else:
            for param in param_chains:
                param_chains[param].append(param_chains[param][-1])
            likelihood_chain.append(likelihood_old)

metropolis_hastings(iterations)

# Convert chains to numpy arrays
param_samples = {param: np.array(values) for param, values in param_chains.items()}

# Combine all parameter samples into a single array for plotting
samples_array = np.column_stack([param_samples[param] for param in params_to_estimate])

# Create a DataFrame for easier plotting
samples_df = pd.DataFrame(samples_array, columns=params_to_estimate)

# Plot marginal distributions and correlations
plt.figure(figsize=(12, 12))
sns.pairplot(
    data=samples_df,
    kind='hist',
    diag_kind='kde',
    plot_kws={'alpha': 0.5, 'edgecolor': 'none'},
    diag_kws={'fill': True}
)
plt.suptitle("Parameter Distributions and Correlations", y=1.02)
plt.tight_layout()
plt.savefig("FinalProject/Figs/parameter_correlations.png", dpi=300)
plt.show()

# Print results
print("\nEstimated Cosmological Parameters:")
for param, values in param_samples.items():
    mean, std = np.mean(values), np.std(values)
    print(f"{param}: {mean:.4f} ± {std:.4f}")