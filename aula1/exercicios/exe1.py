#%%
import matplotlib.pyplot as plt
import numpy as np

#%%
cov = np.array([[1.2, 0.4], [0.4, 1.8]])

medias = [[0.5, 0.5],
          [2.0, 2.0],
          [1.3, 1.8]]

#%%
for cl in medias:
    x, y = np.random.multivariate_normal(cl, cov, 5000).T
    plt.scatter(x, y)

plt.show()


#%%
