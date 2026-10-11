import matplotlib.pyplot as plt
import numpy as np
# Load the file data
data = np.load("Simulation_results_laplace_2026-10-08_12-20-32.npz")
# All the Variables names
print(data.files)

print("G=",data['selected_G'])
Eb_N0_db        = data['Eb_N0_db']
Theor_P_b_G1    = data['Theor_P_b_G1']
P_b             = data['P_b']
channel         = data['channel']
Eb_N0_approx_db = data['Eb_N0_approx_ls']
if(channel=='laplace'):
    P_b_approx = data['P_b_approx_Z_lp']
elif(channel=='gaussian'):
    P_b_approx = data['P_b_approx_Z_g']
else:
    exit("error channel")



fig = plt.figure(figsize=(8, 8))
plt.plot(Eb_N0_db, Theor_P_b_G1, marker='*', mec='black', linewidth=2, linestyle='--', label="Theory_Upper_Bound",  color='purple')
plt.plot(Eb_N0_db, P_b, marker='*', mec='black',  linewidth=2, label="$P_b$", color='blue')
plt.plot(Eb_N0_approx_db, P_b_approx, marker='*', linestyle='--', mec='black',  linewidth=2, label="$P_b$ approx.", color='green')
plt.yscale('log')
plt.grid()
plt.legend(fontsize=12)
if (channel =="laplace"):
    plt.title(f'Bit Error Probability Vs $E_b/N_0$\nLaplace Channel $G_1$ Code')
elif(channel =="gaussian"):
    plt.title(f'Bit Error Probability Vs $E_b/N_0$\nGaussian Channel $G_1$ Code')
else:
    exit("error channel")
plt.xlabel(f'$E_b/N_0(dB)$', fontsize=12)
plt.ylabel(f'$P_b$', fontsize=12)
plt.show()