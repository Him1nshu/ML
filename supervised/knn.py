import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

x = [1, 2, 3, 4]
y = [6, 2, 3, 6]
cl = [0, 1, 1, 0]

plt.scatter(x, y, c=cl)
plt.show()

data = list(zip(x, y))
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(data, cl)

nx, ny = 5, 5
npp = [[nx, ny]]

plt.scatter(x + [nx], y + [ny], c=cl + [knn.predict(npp)])
plt.show()