import matplotlib.pyplot as plt
import numpy as np
from numpy import array
import math 
import commpy as cc
from commpy.channelcoding.convcode import Trellis
from commpy.channelcoding.convcode import conv_encode

import pandas as pd             # For saving the data after getting it
from datetime import datetime   # For date and time

def print_dict_contents(label, d):
    print(f"--- {label} ---")
    for key, value in d.items():
        arr = np.asarray(value)  # Ensures value is treated as a NumPy array
        print(f"{key}: shape={arr.shape}, dtype={arr.dtype}\n{value}\n")


file_path = "Simulation_results_laplace_2026-10-08_12-20-32.npz"



with np.load(file_path) as data:
    data_dict = dict(data)

print_dict_contents("BEFORE CHANGES", data_dict)




channel = data_dict["channel"]
G       = data_dict["selected_G"]


num_bits_for_polynom = data_dict["n_bits_poly_G1"]


k = 1      if (num_bits_for_polynom>0 and G.ndim==1) else G.shape[0]
n = G.size if (num_bits_for_polynom>0 and G.ndim==1) else G.shape[1]
R = k/n
print(f"k={k}, n={n}, R={R}")

if(channel == 'laplace'):
    Eb_N0_approx_ls = np.linspace(10, 12, 4)
elif (channel == 'gaussian'):
    Eb_N0_approx_ls = np.linspace(8.4, 10, 4)
else:
    exit('channel error')
Eb_N0_approx_ls = [float(x) for x in Eb_N0_approx_ls]



N0_approx_ls    = []
b_p_approx_ls   = []

for eb_n0_approx in Eb_N0_approx_ls:
    Eb_N0 = eb_n0_approx                     # in dB
    gamma = np.power(10, Eb_N0/10) # in linear
    # Calculate Symbol Energy - BPSK Symbols are -1,1 -> E_s(Not coded) = 1
    E_s = 1
    E_b = (1/R)*E_s
    N0  = E_b/gamma # define N0 of AWGN for defining later b of laplace
    N0_approx_ls.append(float(N0))
    # laplace distribution b param - Laplace Noise Variance = 2*b^2 
    # (which will give same AWGN Noise Variance -> variance_laplace = N0/2)
    b_p   = math.sqrt(N0)/2
    b_p_approx_ls.append(b_p)

### For  Eb_N0>>1 - P_bapproximated as ~Z^5
# Gaussian Channel  
Z_g_approx     = np.array([np.exp(-1/(N0)) for N0 in N0_approx_ls])
Z_g_5_approx   = np.array([np.power(Z, 5)  for Z in Z_g_approx])
P_b_approx_Z_g = Z_g_5_approx
# Laplace Channel
Z_lp_approx     = np.array([(1+1/b)*np.exp(-1/b) for b in b_p_approx_ls])       
Z_lp_5_approx   = np.array([np.power(Z, 5) for Z in Z_lp_approx])       
P_b_approx_Z_lp = Z_lp_5_approx

data_dict['Eb_N0_approx_ls']  = Eb_N0_approx_ls
data_dict['N0_approx_ls']     = N0_approx_ls
data_dict['b_p_approx_ls']    = b_p_approx_ls
data_dict['P_b_approx_Z_g']   = P_b_approx_Z_g
data_dict['P_b_approx_Z_lp']  = P_b_approx_Z_lp


print_dict_contents("AFTER CHANGES", data_dict)



np.savez(file_path, **data_dict)