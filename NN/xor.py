import numpy as np
from sklearn.neural_network import MLPClassifier

# Inputs
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

# XOR Outputs
y = np.array([0, 1, 1, 0])

# Create MLP
mlp = MLPClassifier(
    hidden_layer_sizes=(2,),
    activation='relu',
    max_iter=5000,
    random_state=42
)

# Train
mlp.fit(X, y)

# Test
print("Predictions:")
for x in X:
    print(f"{x} -> {mlp.predict([x])[0]}")