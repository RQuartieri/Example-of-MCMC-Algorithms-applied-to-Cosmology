from numcosmo_py import Nc, Ncm
import numpy as np
import matplotlib.pyplot as plt

# Initialize the NumCosmo library
Ncm.cfg_init()

# 1. Define a fiducial cosmological model (ΛCDM)
fiducial_cosmo = Nc.HICosmoDEXcdm.new()
print("# Model parameters: ")
fiducial_cosmo.params_log_all()

#fiducial_cosmo.props.Omegab = 0.05  # Baryon density
#fiducial_cosmo.props.Omegac = 0.25  # Cold dark matter density
#fiducial_cosmo.props.H0 = 70.0      # Hubble constant
#fiducial_cosmo.props.Omegak = 0.0   # Curvature density
#fiducial_cosmo.props.Tgamma0 = 2.7255  # CMB temperature

# 2. Create a distance object and prepare it for the fiducial cosmology
dist = Nc.Distance.new(2) # Maximum redshift for distance calculations
dist.prepare(fiducial_cosmo)

# 3. Generate a range of redshifts
z_min = 0.01
z_max = 2
num_data_points = 150
z = np.linspace(z_min, z_max, num_data_points)

# 4. Calculate the luminosity distance for each redshift
d_lum = np.array([dist.luminosity(fiducial_cosmo, z_i) for z_i in z])

# 5. Add Gaussian noise to simulate observational uncertainties
np.random.seed(42)  # For reproducibility
noise_level = 0.15 * d_lum  # 15% noise
mock_d_lum = d_lum + np.random.normal(0, noise_level)

# 6. Plot the mock data
plt.figure(figsize=(8, 6))
plt.errorbar(z, mock_d_lum, yerr=noise_level, fmt="o", label="Mock Data", capsize=3)
plt.plot(z, d_lum, label="Fiducial Model", color="red")
plt.xlabel("Redshift (z)")
plt.ylabel("Luminosity Distance (Mpc)")
plt.title("Mock Supernova Data")
plt.legend()
plt.grid()
plt.savefig("FinalProject/Figs/MockData", dpi=300)
#plt.show()

# 7. Save the mock data to a file (optional)
mock_data = np.column_stack((z, mock_d_lum, noise_level))
np.savetxt("FinalProject/mock_supernova_data.txt", mock_data, header="z d_lum error", fmt="%.6f")