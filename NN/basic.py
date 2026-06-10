import numpy as np

x = np.array([1, 2])

weights = np.array([0.5, 0.3])

bias = 0.1

z = np.dot(x, weights) + bias

output = max(0, z)  # ReLU

print(output)