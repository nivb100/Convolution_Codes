# --- Module: Binary to Decimal Performance Analysis ---
# Compares iterative vs. NumPy vectorized methods for binary-to-decimal conversion.
from pytictoc import TicToc
import numpy as np

# Converts binary vector to decimal (Iterative approach)
# Input: vec (list/ndarray) | Returns: int
def bin_vec_to_dec(vec):
    num    = 0
    weight = 0
    for i in range(len(vec)-1, -1, -1):
        num    = num + vec[i]*(np.power(2, weight))
        weight = weight + 1
    return num

# Converts binary vector to decimal (Vectorized NumPy approach)
# Input: vec (list/ndarray) | Returns: int
def bin_vec_to_dec_vector(vec):
    weight_vec = list(range(len(vec)-1, -1, -1))  
    weight_vec = np.power(2,weight_vec)
    num        = weight_vec @ vec
    return num

# --- Performance Testing Section ---
num_bits = 10000
# Initial example vector
vec = [1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0]

# Generate a large random binary sequence for benchmarking
info_seq = np.random.binomial(n=1, p=0.5, size=num_bits)
info_seq = np.array(info_seq)
vec = info_seq
print("Information sequence (large binary vector) generated.")

t = TicToc()
t.tic()
dec1        = bin_vec_to_dec(vec)
end_time_t1 = t.tocvalue(restart=True)
dec2        = bin_vec_to_dec_vector(vec)
end_time_t2 = t.tocvalue(restart=True)

print("first_func-dec ={} time_to_calc={}".format(dec1, end_time_t1))
print("second_func-dec={} time_to_calc={}".format(dec2, end_time_t2))