import numpy as np
import matplotlib.pyplot as plt
import math

def target_distribution(x):
    """Example target distribution: standard normal distribution"""
    return np.exp(-x**2 / 2) / np.sqrt(2 * np.pi)

def proposal_distribution(x, step_size=0.5):
    """Proposal distribution: normal distribution centered at x"""
    #return np.random.normal(x, step_size)
    return np.random.uniform(-2,2)

def metropolis_hastings(iterations, initial_value, target_dist, proposal_dist):
    x = initial_value
    samples = [x]

    for _ in range(iterations):
        x_new = proposal_dist(x)
        acceptance_prob = min(1, target_dist(x_new) / target_dist(x))
        
        if np.random.rand() < acceptance_prob:
            x = x_new
        
        samples.append(x)
    
    return samples

# Parameters
iterations = 10000
initial_value = 0.0

# Running the Metropolis-Hastings algorithm
samples = metropolis_hastings(iterations, initial_value, target_distribution, proposal_distribution)

# Plotting the results
plt.hist(samples, bins=50, density=True, label='Metropolis-Hastings Samples')
x = np.linspace(-5, 5, 100)
plt.plot(x, target_distribution(x), label='Target Distribution')
plt.legend()
plt.show()
