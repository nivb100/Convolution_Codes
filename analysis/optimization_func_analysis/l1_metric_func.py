# --- Module: L1 Metric Performance Analysis ---
# Compares iterative vs. optimized NumPy methods for L1 distance calculation.

from pytictoc import TicToc
import numpy as np

# Calculates L1 distance (Iterative approach)
# Input: r (ndarray), v (ndarray) | Returns: float
def l1_metric(r, v):
    sum = 0
    for i in range(len(r)):
        sum = sum + np.abs(r[i]-v[i])
    return sum

# Calculates L1 distance (Optimized NumPy approach)
# NumPy computes differences and absolute values across the arrays in compiled
# code, avoiding Python-level work for each element. This can be faster for
# large arrays, though the built-in sum still reduces the results iteratively.
# Input: r (ndarray), v (ndarray) | Returns: float
def l1_metric_opt(r, v):
    return sum(np.abs(r-v))

# Main
vec_size = 100000 
r = np.random.rand(vec_size)
v = np.random.randint(0,2, vec_size)

print("r=",r)
print("v=",v)

t = TicToc()
t.tic()

# Method 1
l1_metric_val      = l1_metric(r,v)
l1_metric_time     = t.tocvalue(restart=True)
# Method 2
l1_metric_opt_val  = l1_metric_opt(r,v)
l1_metric_opt_time = t.tocvalue(restart=True)


print("l1_metric_val     =", l1_metric_val)
print("l1_metric_opt_val =", l1_metric_opt_val)
print("l1_metric_time    =", l1_metric_time)
print("l1_metric_opt_time=", l1_metric_opt_time)

version      = ["l1_metric","l1_metric_opt"]
version_time = [l1_metric_time,l1_metric_opt_time]
best_version_idx = np.argmin(version_time)
print(f"\n")
print(f"best_version_is - {version[best_version_idx]} = {version_time[best_version_idx]}")