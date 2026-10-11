import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from numpy import array
import math 

n_samples = 50000
# Uniform parameters
uni_start = 0
uni_end   = 1
# Laplace Parameters Laplace(mu, b_p)
mu_list        = [0, 5, -10]
b_p_list       = [15, 2, 0.5]
# creating from Uniform distribution X~U[0,1] laplace distribution with lambda param
uniform_samples = np.random.rand(n_samples)
x_bins = 50000
x      = np.linspace(-100, 100, x_bins)
# using np.nan in order for not getting "breaking curve for 0/1 to 1/0 on edges x=0,1"
uniform_pdf = [
    (1/(uni_end-uni_start)) if 0<idx<1
    else np.nan for idx in x] 

laplace_samples = np.zeros((len(mu_list), len(uniform_samples)))
laplace_pdf     = np.zeros((len(mu_list), x_bins))
# Using inverse Transform of cumulative function will create laplace distribution
# Y = h(X) = {mu+b_p*ln(2X) , 0<X<1/2 : mu-b_p*ln(2(1-X)) , 1/2<X<1}
for idx in range(len(mu_list)):
    mu  = mu_list[idx]
    b_p = b_p_list[idx]
    laplace_samples[idx,:] = [ mu+b_p*math.log(2*x) if 0<x<0.5 else mu-b_p*math.log(2*(1-x)) for x in uniform_samples]
    laplace_pdf[idx,:]     = (1/(2*b_p)) * np.exp(-np.abs(x-mu)/b_p)

fig = plt.figure(figsize=(8, 8))
fig.text(
    0.5, 0.05, f'Simulation: N = {n_samples} Samples',
    ha='center', va='center',
    transform=plt.gcf().transFigure,
    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Uniform Samples created by histogram Vs Uniform pdf
plt.subplot(2,1,1)
plt.plot(x, uniform_pdf, linewidth=2, label="Uniform PDF", color='blue')
plt.hist(uniform_samples, bins=200, density=True, alpha=0.5, label="X~U[0,1] Samples")
plt.title("X Samples Vs Uniform Distribution".format(n_samples))
plt.xlim(0,1) # limiting the x values
plt.legend()

# Laplace Samples created by histogram Vs Laplace pdf
for idx in range(3):
    plt.subplot(2,3,4+idx)
    plt.hist(laplace_samples[idx], bins=100, density=True, alpha=0.5, label = 'h(X) Samples', color='blue', linewidth=2)
    plt.title(f'h(X) Samples Vs Laplace Distribution\n' rf'$\mu={mu_list[idx]}$, $b={b_p_list[idx]}$')
    # Real pdf Laplace Distribution 
    plt.plot(x, laplace_pdf[idx], linewidth=2,label="Laplace PDF", color='black')
    plt.axvline(x=mu_list[idx], color='red')
    plt.grid()
    plt.plot()
    plt.legend()
plt.show()