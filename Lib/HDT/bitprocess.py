import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

from .constant import *
C = Constant()

##################################
## bit processing
##################################

def hdt_whitener(bits, seed=0b10000_00000000_10):
    whitened_bits = []
    K = 15
    lfsr = seed

    for bit in bits:
        whitened_bit = bit ^ (lfsr & 1)
        whitened_bits.append(whitened_bit)
        feedback = ((lfsr & 1) << 14) | ((lfsr & 1) << 13)
        lfsr = ((lfsr >> 1) ^ feedback) & ((1 << K) - 1)
    
    return np.array(whitened_bits, dtype=np.uint8)

def hdt_encoder(bits, seed=0b000000):
    G0 = 0b110101
    G1 = 0b101111
    K = 6
    shift_register = seed
    encoded_bits = []
    
    for bit in bits:
        shift_register = ((shift_register << 1) | bit) & ((1 << K) - 1)
        output_0 = bin(G0 & shift_register).count('1') % 2
        output_1 = bin(G1 & shift_register).count('1') % 2
        encoded_bits.extend([output_0, output_1])
    
    return np.array(encoded_bits, dtype=np.uint8)

def hdt_decoder(punctured_bits, rate):
    if rate == 1:
        puncturing_pattern = [1, 1]
    elif rate == 3/4:
        puncturing_pattern = [1, 1, 0, 1]
    elif rate == 4/6:
        puncturing_pattern = [1, 1, 0, 1, 0, 1]
    elif rate == 16/30:
        puncturing_pattern = [1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1]

    G0 = 0b110101
    G1 = 0b101111
    K = 6  # Constraint length
    num_states = 2 ** (K - 1)
    trellis = {}

    # Build the Trellis (State Transitions and Output)
    for state in range(num_states):
        for bit in [0, 1]:
            next_state = ((state << 1) | bit) & (num_states - 1)
            output_0 = bin(G0 & ((state << 1) | bit)).count('1') % 2
            output_1 = bin(G1 & ((state << 1) | bit)).count('1') % 2
            trellis[(state, bit)] = (next_state, (output_0, output_1))
    
    # Depuncturing: Insert `None` in missing bit positions
    pattern_length = len(puncturing_pattern)
    num_retained = sum(puncturing_pattern)  # How many bits are actually transmitted per pattern cycle

    # Prepare depunctured sequence with `None` for missing bits
    depunctured_bits = []
    pattern_index = 0
    for bit in punctured_bits:
        while pattern_index < pattern_length and puncturing_pattern[pattern_index] == 0:
            depunctured_bits.append(None)  # Insert missing bit marker
            pattern_index += 1
        depunctured_bits.append(bit)
        pattern_index = (pattern_index + 1) % pattern_length

    while pattern_index < pattern_length:  # Handle last missing bits in pattern
        if puncturing_pattern[pattern_index] == 0:
            depunctured_bits.append(None)
        pattern_index += 1

    num_steps = len(depunctured_bits) // 2  # Each step takes 2 bits
    path_metrics = np.full(num_states, np.inf)
    path_metrics[0] = 0  # Start from state 0
    paths = {state: [] for state in range(num_states)}

    # Viterbi Algorithm with Handling of Missing Bits
    for step in range(num_steps):
        received_pair = depunctured_bits[2 * step: 2 * step + 2]
        new_metrics = np.full(num_states, np.inf)
        new_paths = {state: [] for state in range(num_states)}

        for state in range(num_states):
            if path_metrics[state] == np.inf:
                continue

            for bit in [0, 1]:
                next_state, output_pair = trellis[(state, bit)]
                
                # Compute distance, ignoring missing (None) bits
                distance = 0
                for i in range(2):
                    if received_pair[i] is not None:  # Ignore missing bits
                        # distance += abs(output_pair[i] - received_pair[i])
                        distance += abs(int(output_pair[i]) - int(received_pair[i]))

                new_metric = path_metrics[state] + distance

                if new_metric < new_metrics[next_state]:
                    new_metrics[next_state] = new_metric
                    new_paths[next_state] = paths[state] + [bit]

        path_metrics = new_metrics
        paths = new_paths

    # Get the best path (minimum cost)
    best_state = np.argmin(path_metrics)
    return np.array(paths[best_state], dtype=np.uint8)

def hdt_puncturing(bits, rate):
    if rate == 1:
        return bits
    elif rate == 3/4:
        pattern = [1, 1, 0, 1]
    elif rate == 4/6:
        pattern = [1, 1, 0, 1, 0, 1]
    elif rate == 16/30:
        pattern = [1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1]
    
    pattern_length = len(pattern)
    bit_matrix = np.array(bits[:int(len(bits) // pattern_length) * pattern_length]).reshape(-1, pattern_length)
    punctured_bits = bit_matrix[:, np.array(pattern, dtype=bool)].ravel()

    return punctured_bits
    
def hdt_crc32_msb(data_bytes, seed=np.uint32(0x00000000)):
    crc_polinomial = np.uint32(0x04C11DB7)
    crc = seed
    BIT_REVERSE_TABLE = np.array([int('{:08b}'.format(i)[::-1], 2) for i in range(256)], dtype=np.uint8)

    for byte in data_bytes:
        crc ^= np.uint32(BIT_REVERSE_TABLE[byte]) << 24
        
        for _ in range(8):
            if crc & np.uint32(0x80000000):
                crc = (crc << 1) ^ crc_polinomial
            else:
                crc = crc << 1

    return np.array([(crc >> 24) & 0xFF, (crc >> 16) & 0xFF, (crc >> 8) & 0xFF, crc & 0xFF], dtype=np.uint8)

def hdt_crc32(data_bytes, seed=np.uint32(0x00000000)):  
    crc_polynomial = np.uint32(0xEDB88320)  # Bit-reversed polynomial
    crc = seed

    for byte in data_bytes:
        crc ^= np.uint32(byte)
        
        for _ in range(8):
            if crc & np.uint32(0x00000001):
                crc = (crc >> 1) ^ crc_polynomial
            else:
                crc = crc >> 1

    return np.array([(crc >> 24) & 0xFF, (crc >> 16) & 0xFF, (crc >> 8) & 0xFF, crc & 0xFF], dtype=np.uint8)