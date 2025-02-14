import os
from numcosmo_py import Nc, Ncm
import numpy as np
import matplotlib.pyplot as plt

# Initialize the NumCosmo library
Ncm.cfg_init()

# Define the path to the mock data file
mock_data_path = os.path.join(os.path.dirname(__file__), "mock_supernova_data.txt")

# Check if the file exists
if not os.path.exists(mock_data_path):
    raise FileNotFoundError(f"Mock data file not found at: {mock_data_path}")

# 1. Load the mock data
mock_data = np.loadtxt(mock_data_path)
z = mock_data[:, 0]
d_lum = mock_data[:, 1]
error = mock_data[:, 2]

# 2. Store data in NumCosmo vectors
z_vec = Ncm.Vector.new_array(z)
d_lum_vec = Ncm.Vector.new_array(d_lum)
error_vec = Ncm.Vector.new_array(error)

# 3. Create a cosmological model (ΛCDM)
cosmo = Nc.HICosmoDEXcdm.new()

# Set initial values for parameters
cosmo.props.Omegab = 0.05  # Baryon density
cosmo.props.H0 = 70.0      # Hubble constant

# 4. Define a custom likelihood function
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

# 5. Set up the Metropolis-Hastings sampler
mcmc = Nc.MCMC.new(Nc.MCMCAlgoMHS, likelihood)

# 6. Set priors for the parameters to be estimated
# Omegab: Baryon density (flat prior between 0.01 and 0.1)
mcmc.param_set_ftype(cosmo, "Omegab", Nc.FTYPE_FREE, 0.01, 0.1)

# H0: Hubble constant (flat prior between 60 and 80 km/s/Mpc)
mcmc.param_set_ftype(cosmo, "H0", Nc.FTYPE_FREE, 60.0, 80.0)

# 7. Run the MCMC sampler
n_steps = 10000  # Number of MCMC steps
mcmc.run(n_steps)

# 8. Extract the chain
chain = mcmc.get_chain()

# 9. Analyze the results
# Extract Omegab and H0 values from the chain
omegab_chain = chain[:, 0]
h0_chain = chain[:, 1]

# Plot the chains
plt.figure(figsize=(12, 6))

# Omegab chain
plt.subplot(2, 1, 1)
plt.plot(omegab_chain, label="Omegab")
plt.xlabel("Step")
plt.ylabel("Omegab")
plt.title("MCMC Chain for Omegab")
plt.legend()

# H0 chain
plt.subplot(2, 1, 2)
plt.plot(h0_chain, label="H0")
plt.xlabel("Step")
plt.ylabel("H0 (km/s/Mpc)")
plt.title("MCMC Chain for H0")
plt.legend()

plt.tight_layout()
plt.show()

# 10. Compute mean and standard deviation of the parameters
omegab_mean = np.mean(omegab_chain)
omegab_std = np.std(omegab_chain)
h0_mean = np.mean(h0_chain)
h0_std = np.std(h0_chain)

print(f"Estimated Omegab: {omegab_mean:.4f} ± {omegab_std:.4f}")
print(f"Estimated H0: {h0_mean:.2f} ± {h0_std:.2f} km/s/Mpc")