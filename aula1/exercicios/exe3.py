import matplotlib.pyplot as plt
import numpy as np


cov = [
        [[1.0, 0.0], [0.0, 1.0]],
        [[0.2, 0.0], [0.0, 0.2]],
        [[2.0, 0.0], [0.0, 2.0]],
        [[0.2, 0.0], [0.0, 2.0]],
        [[2.0, 0.0], [0.0, 0.2]],
        [[0.3, 0.5], [0.5, 2.0]],
        [[0.3, -.5], [-.5, 2.0]]
]

medias = [[  0,   0],
          [ -1,  -2],
          [ -3,  -3]]

for cl in medias:
    for s in cov:
        x, y = np.random.multivariate_normal(cl, s, 500).T
        plt.scatter(x, y)

plt.show()
