import numpy as np

def ChromaticTone(n_key):

    # key number could be 0 to 87 (key_0:A0:27.5Hz,  key_48:A4:440Hz,  key_87:C8:4186.0Hz)

    return 440.0 * (2**((n_key-48)/12))