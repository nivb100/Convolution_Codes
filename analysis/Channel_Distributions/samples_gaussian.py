import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from numpy import array
import math 

n_samples = 100000
# Laplace Parameters Laplace(mu, b_p)
mu_list        = [0,   5, -10]
sigma_list     = [81, 16, 9]

x_bins = 100000
x      = np.linspace(-200, 200, x_bins)

gaussian_samples = np.zeros((len(mu_list), n_samples))
gaussian_pdf     = np.zeros((len(mu_list), n_samples))

for idx in range(len(mu_list)):
    mu    = mu_list[idx]
    sigma = sigma_list[idx]
    gaussian_samples[idx,:] = np.random.normal(loc=mu, scale=sigma, size=n_samples)
    sigma_2 = np.power(sigma,2)
    gaussian_pdf[idx,:]     = np.sqrt(1/(2*np.pi*sigma_2)) * np.exp(-np.power((x-mu),2)/(2*sigma_2))

fig = plt.figure(figsize=(8, 8))
fig.text(
    0.5, 0.05, f'Simulation: N = {n_samples} Samples',
    ha='center', va='center',
    transform=plt.gcf().transFigure,
    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Uniform Samples created by histogram Vs Uniform pdf
# plt.subplot(1,1,1)
# plt.plot(x, uniform_pdf, linewidth=2, label="Uniform PDF", color='blue')
# plt.hist(uniform_samples, bins=200, density=True, alpha=0.5, label="X~U[0,1] Samples")
# plt.title("X Samples Vs Uniform Distribution".format(n_samples))
# plt.xlim(0,1) # limiting the x values
# plt.legend()

# Laplace Samples created by histogram Vs Laplace pdf
for idx in range(3):
    plt.subplot(1,3,1+idx)
    plt.hist(gaussian_samples[idx], bins=100, density=True, alpha=0.5, label = 'Gaussian Samples', color='blue', linewidth=2)
    plt.title(f'Gaussian Samples Vs Gaussian Distribution\n' rf'$\mu={mu_list[idx]}$, $\sigma={sigma_list[idx]}$')
    # Real pdf Laplace Distribution 
    plt.plot(x, gaussian_pdf[idx], linewidth=2,label="Gaussian PDF", color='black')
    plt.axvline(x=mu_list[idx], color='red')
    plt.grid()
    plt.legend()
plt.show()