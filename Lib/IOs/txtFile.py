import numpy as np

def readtxtfile(filename):
    with open(filename, "r") as file:
        content = file.read()
    hex_values = [int(x.strip(), 16) for x in content.split(",") if x.strip()]

    return np.array(hex_values, dtype=np.uint8)

def readtxtfile(filename, line_number=0, format='int'):
    with open(filename, "r") as file:
        lines = file.readlines()
    if format == 'int':
        if line_number > 0:
            ret_val = np.array([int(x.strip(), 16) for x in lines[line_number].split(",") if x.strip()], dtype=np.uint8)
        else:
            hex_values = [[int(x.strip(), 16) for x in line.split(",") if x.strip()] for line in lines]
            max_length = max(len(row) for row in hex_values)
            ret_val = np.array([row + [0] * (max_length - len(row)) for row in hex_values], dtype=np.uint8)
    elif format == 'float':
        if line_number > 0:
            ret_val = np.array([float(x.strip()) for x in lines[line_number].split(",") if x.strip()])
        else:
            ret_val = np.array([[float(x.strip()) for x in line.split(",") if x.strip()] for line in lines])
    return ret_val
