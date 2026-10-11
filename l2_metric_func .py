from pytictoc import TicToc
import numpy as np

def l2_metric(r,v):
    sum = 0
    for i in range(len(r)):
        sum = sum + np.square(r[i]-v[i])
    return sum

def l2_metric_opt1(r, v):
    return ((r @ r) + (v @ v) - 2*(r @ v))

def l2_metric_opt2(r, v):
    return sum(np.power(r-v,2))

# Main
vec_size = 1000
np.random.seed(1)
r = np.random.rand(vec_size)
v = np.random.randint(0,2, vec_size)

# print("r=",r)
# print("v=",v)

t = TicToc()
t.tic()

# Method 1
l2_metric_val       = l2_metric(r,v)
l2_metric_time      = t.tocvalue(restart=True)
# Method 2
l2_metric_opt1_val  = l2_metric_opt1(r,v)
l2_metric_opt1_time = t.tocvalue(restart=True)
# Method 3
l2_metric_opt2_val  = l2_metric_opt2(r,v)
l2_metric_opt2_time = t.tocvalue(restart=True)


print("l2_metric_val      =", l2_metric_val)
print("l2_metric_opt1_val =", l2_metric_opt1_val)
print("l2_metric_opt2_val =", l2_metric_opt2_val)
print("l2_metric_time     =", l2_metric_time)
print("l2_metric_opt1_time=", l2_metric_opt1_time)
print("l2_metric_opt2_time=", l2_metric_opt2_time)

version      = ["l2_metric","l2_metric_opt1", "l2_metric_opt2"]
version_time = [l2_metric_time,l2_metric_opt1_time,l2_metric_opt2_time]

best_version_idx = np.argmin(version_time)
print(f"\n")
print(f"best_version_is - {version[best_version_idx]} = {version_time[best_version_idx]}")
