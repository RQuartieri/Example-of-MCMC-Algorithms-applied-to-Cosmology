import os
import numpy as np
import matplotlib.pyplot as plt
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
z = mock_data[:, 0]
d_lum = mock_data[:, 1]
error = mock_data[:, 2]

# Store data in NumCosmo vectors
z_vec = Ncm.Vector.new_array(z)
d_lum_vec = Ncm.Vector.new_array(d_lum)
error_vec = Ncm.Vector.new_array(error)

# Create a cosmological model (ΛCDM)
cosmo = Nc.HICosmoDEXcdm.new()

# Set initial values for parameters
param_init = {
    "H0": 70.0,
    "Omegab": 0.05,
    "Omegac": 0.25,
    "Omegax": 0.7,
    "Tgamma0": 2.7255,
    "Yp": 0.24,
    "ENnu": 3.046
}

print("# Model parameters: ")
cosmo.params_log_all()

for param, value in param_init.items():
    setattr(cosmo.props, param, value)

# Define a custom likelihood function
class CustomLikelihood(Ncm.Likelihood):
    def __init__(self, z, d_lum, error, cosmo):
        super().__init__()
        self.z = z
        self.d_lum = d_lum
        self.error = error
        self.cosmo = cosmo
        self.dist = Nc.Distance.new(2.0)
        self.dist.prepare(cosmo)

    def do_eval(self, _):
        chi2 = 0.0
        for z_i, d_lum_i, error_i in zip(self.z, self.d_lum, self.error):
            d_lum_model = self.dist.luminosity(self.cosmo, z_i)
            chi2 += ((d_lum_i - d_lum_model) / error_i) ** 2
        return -0.5 * chi2

# Create the custom likelihood
likelihood = CustomLikelihood(z, d_lum, error, cosmo)

# Metropolis-Hastings Algorithm
iterations = 100000
step_size = 0.01

# Store parameter chains
param_chains = {param: [value] for param, value in param_init.items()}
likelihood_chain = [likelihood.do_eval(None)]

def metropolis_hastings(iterations):
    for _ in range(iterations):
        new_params = {param: np.random.normal(param_chains[param][-1], step_size) for param in param_init}
        
        for param, value in new_params.items():
            setattr(cosmo.props, param, value)

        likelihood_new = likelihood.do_eval(None)
        likelihood_old = likelihood_chain[-1]
        accept = np.exp(likelihood_new - likelihood_old)

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

# Compute mean and standard deviation for each parameter
param_estimates = {param: (np.mean(values), np.std(values)) for param, values in param_samples.items()}

# Define number of rows and columns dynamically
num_params = len(param_samples)
ncols = 3  # Number of columns
nrows = (num_params + ncols - 1) // ncols  # Calculate rows dynamically

# Save plots
fig, axs = plt.subplots(nrows, ncols, figsize=(15, 5 * nrows))
axs = axs.flatten()  # Flatten to handle indexing safely

for i, (param, values) in enumerate(param_samples.items()):
    mean, std = param_estimates[param]
    true_value = param_init[param]
    
    axs[i].hist(values, bins='auto', density=True, alpha=0.7)
    axs[i].axvline(true_value, color='r', linestyle='--', label=f"True {param} = {true_value:.4f}")
    
    # Add text annotation with estimated values below the true values
    text_x = mean + 0.2 * std  # Position the text slightly right of the mean
    text_y = axs[i].get_ylim()[1] * 0.75  # Position the true value slightly higher
    axs[i].text(text_x, text_y, f"True: {true_value:.4f}", fontsize=10, color='red')

    text_y_est = axs[i].get_ylim()[1] * 0.65  # Estimated value slightly below the true value
    axs[i].text(text_x, text_y_est, f"Est: {mean:.4f} ± {std:.4f}", fontsize=10, color='black')

    axs[i].set_xlabel(param)
    axs[i].set_ylabel("Density")
    axs[i].legend()

# Remove empty subplots if the number of parameters is not a multiple of `ncols`
for j in range(i + 1, len(axs)):
    fig.delaxes(axs[j])

plt.tight_layout()
plt.savefig("FinalProject/Figs/parameter_distributions.png", dpi=300)
#plt.show()

# Print results
print("\nEstimated Cosmological Parameters:")
for param, (mean, std) in param_estimates.items():
    print(f"{param}: {mean:.4f} ± {std:.4f} (True: {param_init[param]:.4f})")
