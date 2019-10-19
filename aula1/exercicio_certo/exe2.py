#%%
import matplotlib.pyplot as plt
import numpy as np

#%%
# Parte A
cov = np.array([[1.2, 0.4], [0.4, 1.8]])

medias = [[0.5, 0.5],
          [2.0, 2.0],
          [1.3, 1.8]]

#%%
for cl in medias:
    x, y = np.random.multivariate_normal(cl, cov, 5000).T
    plt.scatter(x, y)

plt.show()

# Parte B
# Traçando uma reta entre as médias de cada distribuição, encontrando o ponto central dessa reta
# e traçando uma reta perpendicular a essa reta nesse ponto

# Parte C
# Se as médias tiverem uma distância menor que a distância das matrizes de variância ao ponto zero,
# então as distribuições serão bem sobrepostas e fica mais difícil de separá-las
#