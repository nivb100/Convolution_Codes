import matplotlib.pyplot as plt
import numpy as np
from numpy import array
import math 
import commpy as cc
from commpy.channelcoding.convcode import Trellis
from commpy.channelcoding.convcode import conv_encode

import pandas as pd             # For saving the data after getting it
from datetime import datetime   # For date and time

class Trellis_Cell():
    def __init__(self, curr_state, ts):
        self.curr_state         = curr_state
        self.timestamp          = ts
        self.acumulative_metric = 0
        self.prev_optimal_state = -1

# BPSK Modulation 
def mod_bpsk(x):
    return 1 - 2*x

# BPSK DeModulation 
def demod_bpsk(x):
    return (1-x)/2

# Calculate memory 
def calc_memory(g_matrix, num_bits_for_polynom):
    if num_bits_for_polynom > 0:
        g_matrix = np.array(g_matrix)
        if g_matrix.ndim == 1:
            g_matrix = np.atleast_2d(g_matrix)
    num_in   = g_matrix.shape[0]
    mem_list = []
    
    for i in range(num_in):
        max_deg = 0 
        for poly in g_matrix[i]:
            if num_bits_for_polynom > 0: 
                # convert octal to binary string with length num_bits_for_polynom
                bin_str = format(poly, f'0{num_bits_for_polynom}b') 
                degree  = bin_str.rfind('1')
            else:
                poly   = "".join(poly.astype(str)) # convert poly to string
                degree = poly.rfind('1')
            if degree > max_deg:
                max_deg = degree 
        mem_list.append(max_deg)
    return np.array(mem_list)


def octal_to_binary_vec(num, num_bits_for_polynom):
    n       = num
    vec     = []
    str_vec = ''
    if n==0:
        str_vec = format(0, f'0{num_bits_for_polynom}b')
    else:
        while n != 0:
            oct_val = n % 8
            oct_str = format(oct_val, f'0{num_bits_for_polynom}b')
            str_vec = oct_str + str_vec
            n = n // 8
    vec = [int(b) for b in str_vec]    
    return vec

def decimal_to_bits(x, n_bits):
    return [int(b) for b in format(x, f'0{n_bits}b')]



def bin_vec_to_dec(vec):
    num    = 0
    weight = 0
    for i in range(len(vec)-1, -1, -1):
        num = num + vec[i]*(np.power(2, weight))
        weight = weight + 1
    return num

# Optimized Version of bin_vec_to_dec
def bin_vec_to_dec_vector(vec):
    weight_vec = list(range(len(vec)-1, -1, -1))  
    weight_vec = np.power(2,weight_vec)
    num        = weight_vec @ vec
    return num


def deg_poly(poly):
    bin_str      = format(int(str(poly), 8), 'b')
    bin_str_rev  = bin_str[::-1]
    degree       = bin_str_rev.rfind('1')
    return degree

def build_trellis_fsm(memory, g_poly, constraint_length, num_bits_for_polynom):
    sum_memory = sum(memory)
    num_states = np.power(2, sum_memory)
    k          = 1      if (num_bits_for_polynom>0 and g_poly.ndim==1) else g_poly.shape[0]
    num_inputs = np.power(2,k)
    num_bits   = sum_memory

    trellis_next_state = np.zeros((num_states,num_inputs), dtype=int)   
    outputs            = np.zeros((num_states,num_inputs,n), dtype=int) 
    # print(g_poly) # before transpose if doing so
    # Using G transpose for more comfortable way of multiplication
    if num_bits_for_polynom>0:
        g_poly = g_poly.transpose()        
    # print(g_poly) # after transpose if doing so
    g_mat = []
    if num_bits_for_polynom > 0: # Octal representation
        for i in range(len(g_poly)):
            concat_vec = []
            if k>1: # for k>1 we will concatenate the same columns of G^T to one vetor
                for j in range(len(g_poly[i])):
                    vec        = octal_to_binary_vec(g_poly[i][j], memory[j]+1)
                    concat_vec = concat_vec + vec[:memory[j]+1] # crop indices that doesn't suppose to be joined
            else:
                concat_vec = octal_to_binary_vec(g_poly[i], memory[0]+1)
            g_mat.append(concat_vec)
    else:
        if k>1: # for k>1 we will concatenate the same columns of G^T to one vetor
            for j in range(g_poly.shape[1]):
                concat_vec = []            
                for i in range(g_poly.shape[0]): # notice for running here on the row loop 
                    vec        = list(g_poly[i][j])
                    concat_vec = concat_vec + vec
                g_mat.append(concat_vec)
        else:
            for j in range(g_poly.shape[1]):
                vec        = g_poly[0][j]
                concat_vec = vec
                g_mat.append(concat_vec)
    # print(g_mat)
    g_mat = np.array(g_mat)
    # print("start loop")
    for curr_state in range(num_states):
        state_vec = decimal_to_bits(curr_state, num_bits)
        # Need to arrange state vector according to length of each m_i
        # print("state_vec=", state_vec)
        # print("curr_state=", curr_state)
        index = 0
        state_list   = [] 
        for m_i in memory:
            # taking the state_vec lsbs tp be on state for m_1 - then why slicing from the end of list in reverse order
            state_list = state_list + [state_vec[sum_memory-1-m_i-index+1:sum_memory-1-index + 1]]
            index      = index+m_i
        # print("state_list=", state_list)
        for input_val in range(num_inputs):
            next_state      = []
            u_list          = []
            next_state_list = []
            input_val_vec  = decimal_to_bits(input_val, k)
            # print("input_val=", input_val)
            # print("input_val_vec=", input_val_vec)
            # print("curr_state_vec=", state_list)
            # Next state - shift all the latest value of each register and add the input
            # Concat the LSB input_val to m_1 register (m_k will be concatenated with MSB)
            next_state_list = state_list.copy()
            for idx in range(k):
                next_state_list[idx] = [input_val_vec[k-1-idx]] + next_state_list[idx][:-1] 
            # print("next_state_list=", next_state_list)
            # Concat all lists together for one list of defining next state
            for idx in range(len(next_state_list)):
                next_state = next_state_list[idx] + next_state
            # print("next_state=", next_state)
            # u_vec = input_val + state_vec       # concat lists
            # print("u_vec=", u_vec)
            for idx in range(k):
                u_list =  u_list + ([input_val_vec[k-1-idx]] + state_list[idx])
            # print("u_list=", u_list)
            u_list = np.array(u_list)
            outputs[curr_state, input_val] = (g_mat @ u_list) % 2
            next_state_val = bin_vec_to_dec_vector(next_state)
            # print("next_state_val=", next_state_val)
            # print(f"outputs[{curr_state}, {input_val}]=", outputs[curr_state, input_val])
            # print("#######")
            trellis_next_state[curr_state][input_val] = next_state_val

    return trellis_next_state, outputs


def conv_encoding(info_seq, k, trellis_next_state, outputs, total_memory):
    state            = 0 # initial state
    encoded_seq_taps = []
    # adding to info_seq termimation bits
    # print(info_seq)
    # print(f"k={k}")
    term_bits = np.zeros((1,k), dtype=int)
    for m in range(total_memory):
        info_seq = np.append(info_seq, term_bits)
    # print("info_seq_with_m_term_bits", info_seq)

    info_len_by_k = int(len(info_seq)/k)
    for idx in range(0,info_len_by_k):
        input_vec     = info_seq[idx*k:k+idx*k]
        input_vec     = input_vec[::-1]   # reverse input so later input will go to upper u_k
        input_vec_val = bin_vec_to_dec_vector(input_vec)
        coded_bits    = outputs[state][input_vec_val]
        state         = trellis_next_state[state][input_vec_val]
        # print(f"state={state}, coded_bits={coded_bits}" )
        encoded_seq_taps.append(coded_bits)
    
    # concatenate all to one sequence 
    encoded_seq = np.concatenate(encoded_seq_taps)
    # print(f"encoded_seq={encoded_seq} state={state}")
    if state != 0:
        print("error non-zero-tail state, state=",state)
    return encoded_seq

# laplace source random sample
    # creating from Uniform distribution X~U[0,1] laplace distribution with lambda param
    # Using inverse Transform of cumulative function will create laplace distribution
    # Y = h(X) = {b_p*ln(2X) , 0<X<1/2 : -b_p*ln(2(1-X)) , 1/2<X<1}
def get_laplace_src_samples(n_samples, b_p):
    uniform_samples = np.random.rand(n_samples)
    laplace_samples = [ b_p*math.log(2*x) if 0<x<0.5 else -b_p*math.log(2*(1-x)) for x in uniform_samples]
    laplace_samples = np.array(laplace_samples)
    # x = np.linspace(-8, 8, 1000)
    # pdf = 0.5 * np.exp(-np.abs(x))
    # plt.figure(figsize=(10,5))
    # plt.hist(laplace_samples,bins=100, density=True, alpha=0.5)
    # plt.plot(x, pdf, linewidth=2)
    # plt.show()
    return laplace_samples

def get_gaussian_src_samples(n_samples, sigma_in):
    mu, sigma = 0, sigma_in
    gaussian_samples = np.random.normal(loc=mu, scale=sigma, size=n_samples)
    gaussian_samples = np.array(gaussian_samples)
    # x = np.linspace(-8, 8, 1000)
    # sigma_2 = np.power(sigma,2)
    # pdf = np.sqrt(1/(2*np.pi*sigma_2)) * np.exp(-np.power((x-mu),2)/(2*sigma_2))
    # plt.figure(figsize=(10,5))
    # plt.hist(gaussian_samples, bins=100, density=True, alpha=0.5)
    # plt.plot(x, pdf, linewidth=2)
    # plt.show()
    return gaussian_samples
    


# laplace samples with param b_p
def noisy_laplace_channel(tx_symbols, b_p):
    z = get_laplace_src_samples(len(tx_symbols), b_p)
    rx_symbols = tx_symbols + z
    # print("z laplace samples=",z)
    return rx_symbols

def noisy_gaussian_channel(tx_symbols, sigma):
    z = get_gaussian_src_samples(len(tx_symbols), sigma)
    rx_symbols = tx_symbols + z
    # print("z gaussian samples=",z)
    return rx_symbols



# L1 Metric Calculation
def l1_metric(r, v):
    sum = 0
    for i in range(len(r)):
        sum = sum + np.abs(r[i]-v[i])
    return sum

# L1 Metric Calculation - Optimized Method
def l1_metric_opt(r, v):
    return sum(np.abs(r-v))

# L2 Metric Calculation
def l2_metric(r,v):
    sum = 0
    for i in range(len(r)):
        sum = sum + np.square(r[i]-v[i])
    return sum

# L2 Metric Calculation - Optimized Method
def l2_metric_opt1(r, v):
    return ((r @ r) + (v @ v) - 2*(r @ v))

# Viterbi Algorithm
def viterbi_decoding_metric(rx_symbols, k, n, trellis_next_state, trellis_output, branch_metric, n_info_taps_and_term):
    time_units = int(n_info_taps_and_term/k)
    # s_opt      = np.zeros(n_info_taps_and_term+1, dtype=int)
    # u_opt      = np.zeros(n_info_taps_and_term, dtype=int)
    s_opt      = np.zeros(time_units+1, dtype=int)
    u_opt      = np.zeros(time_units, dtype=list)
    rx_opt_seq = np.zeros(len(rx_symbols))
    num_states = trellis_next_state.shape[0]
    num_inputs = np.power(2, k)
    # print("num_states=",num_states)
    trellis_array = [[Trellis_Cell(i,j) for j in range(time_units+1)] for i in range(num_states)]

    # Initialize accumulated metric - zero S0 and DeOptimize Other States to Cancel their start paths from t=0
    trellis_array[0][0].acumulative_metric = 0
    for st in range(1,num_states):
        trellis_array[st][0].acumulative_metric = math.inf

    for t in range(1, time_units+1):
        for state in range(num_states):
            accumulated_metric = np.ones(num_states)*math.inf
            # print("t={} state={}".format(t, state))
            for prev_state in range(num_states):
                
                for input_v in range(num_inputs): # would be fixed for k>1
                    input_vec_val = input_v
                    if(trellis_next_state[prev_state][input_vec_val] == state):
                        ts_rx_symbols      = rx_symbols[(t-1)*n:t*n] 
                        calc_branch_metric = branch_metric(ts_rx_symbols, trellis_output[prev_state][input_v])
                        lcl_accumulated_metric = trellis_array[prev_state][t-1].acumulative_metric +\
                                                         calc_branch_metric
                        if(lcl_accumulated_metric < accumulated_metric[prev_state]):
                            accumulated_metric[prev_state] = lcl_accumulated_metric
                        # print("ts_rx_symbols=", ts_rx_symbols)
                        # print("trellis_output[prev_state={}][bit={}]={}".format(prev_state, bit, trellis_output[prev_state][bit]))
                        # print("calc_branch_metric(prev_state={})(curr_state={})(bit={})={}".format(prev_state,state,bit,calc_branch_metric))
                        # print("accumulated_metric(prev_state={})={}".format(prev_state, accumulated_metric[prev_state]))
                        
            # find the min and apply on the trellis array
            opt_state                                  = np.argmin(accumulated_metric)
            trellis_array[state][t].prev_optimal_state = int(opt_state)
            trellis_array[state][t].acumulative_metric = accumulated_metric[opt_state]
            # print("opt_state[t={}]={}".format(t, opt_state))
    
    # starting from the end (S_t, t= N+m-1)
    s_opt[time_units] =  0 # last state is S0 after termination

    # Reconstructing u_opt and rx_ml_opt
    for t in range(time_units-1, -1, -1): # including t=0
        # print("t=",t)
        state    = s_opt[t+1]
        s_opt[t] = trellis_array[state][t+1].prev_optimal_state
        for u in range(num_inputs):
            if(trellis_next_state[s_opt[t]][u] == state):
                u_opt[t]                = decimal_to_bits(u, k)
                rx_opt_seq[t*n:(t+1)*n] = trellis_output[s_opt[t]][u]
                # print("rx_opt_seq[{}:{}]={}".format(t*n,(t+1)*n, trellis_output[s_opt[t]][u]))

    return u_opt, rx_opt_seq


##############  MAIN  #############

start_date      = datetime.now().strftime('%Y-%m-%d_%H-%M')
start_date_cmpt = datetime.now()
print(f"Start Simualtion Time:{start_date}")
### Choose Channel ####
channel = "laplace"
# channel = "gaussian"



# Coding the source with convolution code
num_bits_for_polynom = 0 # default value if not using it
######## Option 1 - k=1 #########
# G(D) = [D, 1+D, D+D^2] - filled in octal representation & reversed represenatation - D^0 is the MSB
G = np.array([2,6,3]) # bad represntation if don't know the number of bits all polynoms
num_bits_for_polynom = 3

# Alternative representation - G Will be 3D array - 1st dim k, 2nd dim n, 3rd dim - polynom powers
# G = np.array([[[0,1,0], [1,1,0], [0,1,1]]])
######## Option 1 - k=1 ########

####### Option 2 - k>1 #########
# G(D) = [1+D,   0, 1+D;
#           0, 1+D, 1+D] - filled in octal representation & reversed represenatation - D^0 is the MSB
# G = np.array([[3, 0, 3],   # bad represntation if don't know the number of bits all polynoms
            #   [0, 3, 3]])
            #   [0, 6, 6]])
# num_bits_for_polynom = 2
# Alternative representation - G Will be 3D array - 1st dim k, 2nd dim n, 3rd dim - polynom powers
# G = np.array([[[1,1], [0,0], [1,1]],
            #   [[0,0], [1,1], [1,1]]  ])
####### Option 2 - k>1 #########

####### Saving for DataFrame #######
G1       = np.array([[[0,1,0], [1,1,0], [0,1,1]]])
G1_octal = np.array([2,6,3])
num_bits_for_polynom_G1 = 3
G2 = np.array([[[1,1], [0,0], [1,1]],
              [[0,0], [1,1], [1,1]]  ])
G2_octal = np.array([[3, 0, 3], 
                     [0, 3, 3]]) 
num_bits_for_polynom_G2 = 2
####### Saving for DataFrame #######

k = 1      if (num_bits_for_polynom>0 and G.ndim==1) else G.shape[0]
n = G.size if (num_bits_for_polynom>0 and G.ndim==1) else G.shape[1]
R = k/n
print(f"k={k}, n={n}")
print("G=",G)
print("G(dims)=",G.shape)


########### Running on Eb_N0 Values ##############
# Eb_N0_ls     = np.linspace(0, 8.2, 16) # The value 8.2 -> 6.249*10^-7 checked manually (for gaussian channel)
if (channel == "laplace"):
    Eb_N0_ls     = np.linspace(0, 10, 16)
elif (channel == "gaussian"):
    Eb_N0_ls     = np.linspace(0, 8.4, 16)
else:
    exit("channel error")
avg_p_b_list = []
N0_ls        = []
b_p_ls       = []


started_sim = 0
# Eb/N0 param
for eb_n0 in Eb_N0_ls:
    Eb_N0 = eb_n0                     # in dB
    gamma = np.power(10, Eb_N0/10) # in linear
    # Calculate Symbol Energy - BPSK Symbols are -1,1 -> E_s(Not coded) = 1
    E_s = 1
    E_b = (1/R)*E_s
    N0  = E_b/gamma # define N0 of AWGN for defining later b of laplace
    N0_ls.append(N0)
    # laplace distribution b param - Laplace Noise Variance = 2*b^2 
    # (which will give same AWGN Noise Variance -> variance_laplace = N0/2)
    b_p   = math.sqrt(N0)/2
    b_p_ls.append(b_p)
    # Gaussian distribution has sigma^2 as variance which is sigma^2=N0/2
    sigma = math.sqrt(N0/2)
    


    # Channel Passing
    if channel == "laplace":
        # decoding_metric = l1_metric
        decoding_metric = l1_metric_opt  # Optimized Version
    elif channel == "gaussian":
        # decoding_metric = l2_metric
        decoding_metric = l2_metric_opt1 # Optimized Version
    else:
        exit("error channel")
    
    if started_sim == 0:
        print(f"laplace param: mu={0}, b      ={b_p}")
        print(f"gauss param:   mu={0}, sigma^2={N0}")
        print(f"Passed through {channel} channel decoding_metric will be - {decoding_metric.__name__}")
        started_sim = 1

    num_iters       = 500000
    min_iters       = 100     # For getting randomization for low SNR 
    num_errors      = 0       # summing error bits after decoding (if stop condition activated use it)
    num_msg_bits    = 0       # summing message bits (if stop condition activated use it)
    num_err_stop_th = 100  
    stop_condition_activated = 0
    P_b_list  = []

    for itr in range(1,num_iters+1):
        # np.random.seed(iter)
        T = 4000    # Info Sequence length  
        num_msg_bits = num_msg_bits + T
        ########## source randomization ##########
        p       = 0.5  # bernoulli param
        n_trial = 1    # transform binomial to bernoulli 
        info_seq = np.random.binomial(n=n_trial, p=p, size=T)
        info_seq = np.array(info_seq)
        # info_seq = np.array([1,0,1]) # Sanity Check

        ########## Calc Memory of the conv code ##########
        memory            = calc_memory(G, num_bits_for_polynom)
        constraint_length = memory + 1
        # print("memory=", memory)

        # Using this for finding the number of termination bits
        total_memory = max(memory)

        ########## Create Trellis ##########
        # in LSB convention ( D^0 is the MSB)
        sum_memory = sum(memory)
        num_states = np.power(2, sum_memory)
        num_inputs = np.power(2, k)
        trellis_next_state = np.zeros((num_states,num_inputs), dtype=int)
        trellis_outputs    = np.zeros((num_states,num_inputs, n), dtype=int)
        trellis_next_state, trellis_outputs = build_trellis_fsm(memory, G, constraint_length, num_bits_for_polynom)
        # print("trellis_states_fsm=\n", trellis_next_state)
        # print("trellis_outputs_fsm=\n", trellis_outputs)

        """
        # golden reference check of encoding conv code
        G_commpy = np.array([[2,6,3]])
        trellis = Trellis(memory=memory,
                    g_matrix=G_commpy,
                    polynomial_format='LSB')
        coded_ref  = conv_encode(info_seq, trellis, termination='term')
        print("golden check=", coded_ref)
        """

        # The number of term bits to be added will be = 
        # num steps cleaning longest register(max m) mult by k per all inputs = total_memory*k
        num_info_taps_and_term_bits = T + total_memory*k
        
        ########## Encoding the Information Sequence ##########
        encoded_seq = conv_encoding(info_seq, k, trellis_next_state, trellis_outputs, total_memory)

        ########## BPSK Modulation ##########
        mod_seq    = mod_bpsk(encoded_seq)
        tx_symbols = mod_seq

        trellis_outputs_bpsk = trellis_outputs # BPSK trellis_outputs for decoding algorithm
        for i in range(trellis_outputs.shape[0]):
            for j in range(trellis_outputs.shape[1]):
                trellis_outputs_bpsk[i][j] = mod_bpsk(trellis_outputs[i][j])

        ########## Channel Passing ##########
        if channel == "laplace":
            rx_symbols = noisy_laplace_channel(tx_symbols, b_p)
        elif channel == "gaussian":
            rx_symbols = noisy_gaussian_channel(tx_symbols, sigma)
        else:
            exit("error channel")
        # print(f"Passed through {channel} channel decoding_metric will be - {decoding_metric.__name__}")

        """
        # start for checking - G option 1 laplace
        rx_symbols = [0.1,0.2,-0.5,
                    -1,0.8,-0.5,
                    -0.9,1.2,1.1,
                    1,-2.2,0.8,
                    1.3,-0.8,0] """
        """ 
        # start for checking - G option 2 gaussian
        rx_symbols = [0.1,0.2,-0.5,
                    -1,0.8,-0.5,
                    -0.9,1.2,1.1] """    
        # print("rx_symbolos=", rx_symbols)

        time_units = num_info_taps_and_term_bits/k
        # print("Info+Term - T+m bits={num_info_taps_and_term_bits} time units={time_units}")

        ########## Decoding by Viterbi Algorithm ##########
        u_opt      = np.zeros(num_info_taps_and_term_bits)
        rx_opt_seq = np.zeros(len(rx_symbols))

        u_opt, rx_opt_seq = viterbi_decoding_metric(rx_symbols, k, n, trellis_next_state, trellis_outputs_bpsk, decoding_metric, num_info_taps_and_term_bits)
        # print("u_opt=", u_opt)
        u_opt = np.concatenate(u_opt)
        # print("u_opt_concatenated=", u_opt)
        rx_opt_seq = np.array([int(x) for x in rx_opt_seq])
        # print("rx_opt_seq=", rx_opt_seq)

        ########## BPSK Demodulation ##########
        c_opt_seq = demod_bpsk(rx_opt_seq)
        c_opt_seq = np.array([int(x) for x in c_opt_seq])
        # print("c_opt_seq",c_opt_seq)

        ########## Bit Error rate Calculation ##########
        # print("Bit Error rate Calculation")
        u_opt_info_only = u_opt[:T]
        # print(f'u_opt          ={u_opt}')
        # print(f'u_opt_info_only={u_opt_info_only}')
        # print(f'info_seq       ={info_seq}')
        error_vec = info_seq ^ u_opt_info_only
        # print(f'error_vec={error_vec}')
        
        ########## Error probability per bit ##########
        # Summarizing the errors from previous loops with the current iteration errors
        num_errors = num_errors + sum(error_vec) 
        P_b = (1/T)*sum(error_vec)
        if (itr % 500 == 0):
            print(f'Eb_N0={eb_n0} iter={itr}, Pb={P_b}, curr_iter_errors={sum(error_vec)}, num_errors={num_errors}')
        if((num_errors >= num_err_stop_th) and (itr >= min_iters) ): #### break condition (num_errors threshold)
            avg_p_b = num_errors/num_msg_bits
            print(f"Stop_condition at Eb_N0={Eb_N0}, iter={itr}, curr_iter_errors={sum(error_vec)}, P_b={P_b}, avg_P_b={avg_p_b}, num_errors={num_errors}")
            stop_condition_activated = 1
            break


    ########## Average Bit Error Probability ##########
    if(stop_condition_activated == 0):
        avg_p_b = num_errors/num_msg_bits
        # avg_p_b = sum(P_b_list)/len(P_b_list)
        print(f"Max num_iters={num_iters}: Eb_N0={Eb_N0}, Average P_b={avg_p_b}, num_errors={num_errors}")
    avg_p_b_list.append(avg_p_b)


Eb_N0_ls        = [float(x) for x in Eb_N0_ls]
N0_ls           = [float(x) for x in N0_ls]
avg_p_b_list_cl = [float(x) for x in avg_p_b_list]

print("Eb_N0_list=", Eb_N0_ls)
print("avg_p_b_list=", avg_p_b_list_cl)


# Theoretical P_b
# For G1

# Upper_bound(Z) =(1/k) * ((Z^5)/(1-Z-Z^5)^2 )
# k which is the number of inputs to encoder in each time-unit
# Gaussian Channel
Z_g    = np.array([np.exp(-1/(N0)) for N0 in N0_ls])
Z_g_5  = np.array([np.power(Z, 5)  for Z in Z_g])
# Laplace Channel
Z_lp   = np.array([(1+1/b)*np.exp(-1/b) for b in b_p_ls])       
Z_lp_5 = np.array([np.power(Z, 5) for Z in Z_lp])       

if(channel=="laplace"):
    Z, Z_5 = Z_lp, Z_lp_5
elif channel == "gaussian":
    Z, Z_5 = Z_g, Z_g_5
else:
    exit("error channel")    

Theor_P_b_G1 = (1/k) * (Z_5/np.power((1-Z-Z_5), 2))
Diff = Theor_P_b_G1 - avg_p_b_list_cl
print("N0 list=",N0_ls)
print("Theor_P_b_G1=",Theor_P_b_G1)
print("avg_p_b_list_cl=",avg_p_b_list_cl)
print("Diff=", Diff)


############ Approximation for High SNR - Eb_N0>>1  ############
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
    N0_approx_ls.append(N0)
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






end_date = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
end_date_cmpt = datetime.now()
print(f"End Simualtion Time:{end_date}")

np.savez(f"Simulation_results_{channel}_{end_date}.npz",
    Eb_N0_db         = Eb_N0_ls,
    channel          = channel,
    selected_G       = G,
    N0_ls            = N0_ls,
    E_b              = E_b,
    E_s              = E_s,
    b_ls             = b_p_ls,
    Theor_P_b_G1     = Theor_P_b_G1,
    P_b              = avg_p_b_list_cl,
    Theor_P_b_Diff   = Diff,
    G1               = G1,
    G1_octal         = G1_octal,
    n_bits_poly_G1   = num_bits_for_polynom_G1,
    G2               = G2,
    G2_octal         = G2_octal,
    n_bits_poly_G2   = num_bits_for_polynom_G2,
    Eb_N0_approx_ls  = Eb_N0_approx_ls,
    N0_approx_ls     = N0_approx_ls,
    b_p_approx_ls    = b_p_approx_ls,
    P_b_approx_Z_g   = P_b_approx_Z_g,
    P_b_approx_Z_lp  = P_b_approx_Z_lp
    )


print(f'sim_results_{channel}_{end_date}.npz Results Saved successfuly!')

df = pd.DataFrame({
    'Eb_N0_db'          : Eb_N0_ls,
    'channel'           : channel,
    'N0_ls'             : N0_ls,
    'b_ls'              : b_p_ls,
    'Theor_P_b_G1'      : Theor_P_b_G1,
    'P_b'               : avg_p_b_list_cl,
    'Theor-P_b_Diff_G1' : Diff}   )

df.to_csv(f'sim_results_{channel}_{end_date}.csv', index=False)
print(f'simulation_results_{channel}_{end_date}.csv Results Saved successfuly!')


duration = end_date_cmpt - start_date_cmpt
print("Simulation Duration:", duration)

# # Plotting
# plt.figure(figsize=(10,5))
# plt.plot(times, Vm - 70, linewidth=2, label='Vm')
# plt.plot(times, stim - 70, label = 'Stimuli (Scaled)', linewidth=2, color='sandybrown')
# plt.ylabel("Membrane Potential (mV)", fontsize=15)
# plt.xlabel('Time (msec)', fontsize=15)
# plt.xlim([90,160])
# plt.title("Hodgkin-Huxley Neuron Model", fontsize=15)
# plt.text(150, 0, r'$E_k$={}'.format(hh.EK)+'\n'+r'$E_{{Na}}={}$'.format(hh.ENa)+'\n'+r'$E_{{leak}}={}$'.format(hh.EKleak), fontsize=12,
#              bbox=dict(facecolor='white', alpha=0.5))
# plt.legend(loc=1)
# # plt.show()