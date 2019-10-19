import matplotlib.pyplot as plt
import numpy as np


mu1, mu2 = [1, 1], [3, 3]
mu1 = np.array(mu1)
mu2 = np.array(mu2)

sigma = 0.2

co_mat1 = np.array([[1, 0], [0,1]])
co_mat_rot = np.array([[1, 1], [-1,-0.8]])

x1, y1 = np.random.multivariate_normal(mu1, co_mat1 * sigma, 5000).T
x2, y2 = np.random.multivariate_normal(mu2, co_mat_rot * sigma, 5000).T

plt.scatter(x1, y1, s=1)
plt.scatter(x2, y2, s=1)
plt.show()