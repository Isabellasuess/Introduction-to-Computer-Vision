import numpy as np

# Helpers
def round_half_up(x):
    return np.floor(x + 0.5).astype(int)

def clip(matrix):
    return np.clip(matrix, 0, 255).astype(np.uint8)

# Arithmetic Operations
def add(a, b):
    return clip(a.astype(int) + b.astype(int))

def subtract(a, b):
    return clip(a.astype(int) - b.astype(int))

def multiply(a, b):
    return clip(a.astype(int) * b.astype(int))

def divide(a, b):
    safe_b = np.where(b == 0, 1, b)
    result = round_half_up(a.astype(float) / safe_b)
    result = np.where(b == 0, 255, result)
    return clip(result)

def print_grid(title, matrix):
    print(f"\n{title}")
    for row in matrix:
        print(" ".join(f"{v:3d}" for v in row))

# Two sample 4x4 matrices with values 0-255
A = np.array([
    [136,  38, 217,  22],
    [205, 251,  33, 198],
    [193, 255, 145, 167],
    [ 97,  86,  90, 112],
], dtype=np.uint8)

B = np.array([
    [ 36,  22, 218, 110],
    [194,  18, 205, 219],
    [141, 136,   0,  22],
    [ 14, 182, 134, 178],
], dtype=np.uint8)

print_grid("Matrix A", A)
print_grid("Matrix B", B)
print_grid("A + B", add(A, B))
print_grid("A - B", subtract(A, B))
print_grid("A * B", multiply(A, B))
print_grid("A / B", divide(A, B))