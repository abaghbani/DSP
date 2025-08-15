import numpy as np

def generate_primes(limit):
    '''
    Sieve of Eratosthenes algoritm
    '''
    sieve = np.ones(limit + 1, dtype=bool)  # Create a boolean array
    sieve[0:2] = False  # 0 and 1 are not primes
    for i in range(2, int(np.sqrt(limit)) + 1):
        if sieve[i]:
            sieve[i * i : limit + 1 : i] = False  # Mark multiples as non-prime
    primes = np.nonzero(sieve)[0]  # Indices of True values are primes
    return primes
