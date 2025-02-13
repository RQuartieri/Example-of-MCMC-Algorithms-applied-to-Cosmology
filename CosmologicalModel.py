from numcosmo_py import Nc, Ncm
import matplotlib.pyplot as plt
import numpy as np

# Initialize the NumCosmo library
Ncm.cfg_init()

def test_distances():
    """Example computing cosmological distances."""
    # New homogeneous and isotropic cosmological model NcHICosmoDEXcdm
    # with one massive neutrino.
    cosmo = Nc.HICosmoDEXcdm(massnu_length=1)
    cosmo.set_reparam(Nc.HICosmoDEReparamCMB.new(cosmo.len()))

    # New cosmological distance objects optimizied to perform calculations
    # up to redshift 2.0.
    dist = Nc.Distance.new(2.0)

    # Setting values for the cosmological model, those not set stay in the
    # default values. Remember to use the _orig_ version to set the original
    # parameters when a reparametrization is used.
    cosmo.orig_param_set(Nc.HICosmoDESParams.H0, 70.00)
    cosmo.orig_param_set(Nc.HICosmoDESParams.OMEGA_C, 0.25)
    cosmo.orig_param_set(Nc.HICosmoDESParams.OMEGA_X, 0.70)
    cosmo.orig_param_set(Nc.HICosmoDESParams.T_GAMMA0, 2.72)
    cosmo.orig_param_set(Nc.HICosmoDESParams.OMEGA_B, 0.05)
    cosmo.orig_param_set(Nc.HICosmoDEXCDMSParams.W, -1.10)

    cosmo.orig_vparam_set(Nc.HICosmoDEVParams.M, 0, 0.06)

    # Printing the parameters used.
    print("# Model parameters: ")
    cosmo.params_log_all()

    dist.prepare(cosmo)

    # Generate some fake data from the cosmological model
    # Example: Redshift values and corresponding distances
    z = np.linspace(0, 1, 100)  # Redshift values
    distances = np.array([dist.luminosity_distance(cosmo, z_val) for z_val in z]) + np.random.normal(0, 0.1, size=len(z))  # Distances with noise

    # Define the likelihood function (simplified example)
    def likelihood(cosmo, omega_m, h0, z, distances):
        cosmo.param_set_by_name("Omegak", omega_m)
        cosmo.param_set_by_name("H0", h0)
        dist.prepare(cosmo)
        model_distances = np.array([dist.luminosity_distance(cosmo, z_val) for z_val in z])
        return np.exp(-0.5 * np.sum((distances - model_distances) ** 2))

    # Define the posterior function
    def posterior(cosmo, omega_m, h0, z, distances):
        prior_omega_m = np.exp(-0.5 * ((omega_m - 0.3) / 0.1) ** 2)  # Gaussian prior
        prior_h0 = np.exp(-0.5 * ((h0 - 70.0) / 5.0) ** 2)            # Gaussian prior
        return likelihood(cosmo, omega_m, h0, z, distances) * prior_omega_m * prior_h0

    # Metropolis-Hastings algorithm
    def metropolis_hastings(cosmo, z, distances, initial_params, iterations, step_size):
        omega_m_samples = []
        h0_samples = []
        omega_m = initial_params[0]
        h0 = initial_params[1]
        current_posterior = posterior(cosmo, omega_m, h0, z, distances)

        for i in range(iterations):
            # Propose new parameters
            omega_m_new = np.random.normal(omega_m, step_size)
            h0_new = np.random.normal(h0, step_size)
            new_posterior = posterior(cosmo, omega_m_new, h0_new, z, distances)

            # Acceptance ratio
            acceptance_ratio = new_posterior / current_posterior

            # Accept or reject the new parameters
            if np.random.rand() < acceptance_ratio:
                omega_m = omega_m_new
                h0 = h0_new
                current_posterior = new_posterior

            omega_m_samples.append(omega_m)
            h0_samples.append(h0)

        return np.array(omega_m_samples), np.array(h0_samples)

    # Parameters
    initial_params = [0.5, 80]  # Initial guess
    iterations = 10000
    step_size = 0.1

    # Run the Metropolis-Hastings algorithm
    omega_m_samples, h0_samples = metropolis_hastings(cosmo, z, distances, initial_params, iterations, step_size)

    # Calculate the posterior probabilities
    omega_m_posteriors = [posterior(cosmo, omega_m, true_h0, z, distances) for omega_m in omega_m_samples]
    h0_posteriors = [posterior(cosmo, true_omega_m, h0, z, distances) for h0 in h0_samples]

    # Plot the results
    plt.figure(figsize=(12, 5))

    # Plot the posterior of Omega_m
    plt.subplot(1, 2, 1)
    plt.hist(omega_m_samples, bins=50, density=True, alpha=0.6, color='b', label='Omega_m Samples')
    plt.scatter(omega_m_samples, omega_m_posteriors, color='r', s=1, label='Omega_m Posterior')
    plt.xlabel('Omega_m')
    plt.ylabel('Density / Posterior')
    plt.legend()

    # Plot the posterior of H0
    plt.subplot(1, 2, 2)
    plt.hist(h0_samples, bins=50, density=True, alpha=0.6, color='r', label='H0 Samples')
    plt.scatter(h0_samples, h0_posteriors, color='b', s=1, label='H0 Posterior')
    plt.xlabel('H0')
    plt.ylabel('Density / Posterior')
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    test_distances()
