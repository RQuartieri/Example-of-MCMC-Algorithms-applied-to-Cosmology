from numcosmo_py import Nc, Ncm

Ncm.cfg_init()

# Create a new instance of the LambdaCDM model
cosmo = Nc.HICosmoDEXcdm.new() 

# Set the parameters correctly using the properties provided
#cosmo.param_set_by_name("H0", 70.00)
#cosmo.param_set_by_name("Omegac", 0.25)
#cosmo.param_set_by_name("Omegax", 0.70)
#cosmo.param_set_by_name("Tgamma0", 2.72)
#cosmo.param_set_by_name("Omegab", 0.05)
#cosmo.param_set_by_name("w", -1.10)

# Setting additional parameters
#cosmo.param_set_by_name("massnu", 0.06)

print("# Model parameters: ")
cosmo.params_log_all()


z = 2.0  # redshift
dist = Nc.Distance.new(z)

print("Distance Modulus at redshift z =", z)
