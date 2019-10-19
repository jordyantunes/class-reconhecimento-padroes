from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
from scipy.optimize import minimize
from matplotlib import cm
import numpy as np
import math

# definicao da funcao
def himmelblau(X):
    x = X[0]
    y = X[1]
    a = x*x + y - 11
    b = x + y*y - 7
    return a*a + b*b

func = np.vectorize(himmelblau)

# geracao de valores de x, y, x
x = y = np.linspace(-6, 6)
X, Y = np.meshgrid(x, y)
zs = np.array(func(np.ravel(X), np.ravel(Y)))
Z = zs.reshape(X.shape)

"""
----------------------------------------
EXERCICIO 1
----------------------------------------
"""
# Surface plot
ax = plt.subplot(projection='3d')
ax.plot_surface(X, Y, Z, cmap=cm.coolwarm_r)
plt.show()

# Contour plot
ax = plt.subplot()
ax.contour(X, Y, Z, cmap=cm.coolwarm_r, levels=[x for x in range(-1000, 5000, 75)])
ax.set_xlim(-6, 6)
ax.set_ylim(-6, 6)
plt.show()


valores_iniciais = [
    [-3, 5],
    [0, 0],
    [2, 1],
    [-4, -4],
    [4, -2]
]

respostas = [
    [3.0, 2.0],
    [-2.8, 3.13],
    [-3.77, -3.28],
    [-3.58, -1.84]
]

for ini in valores_iniciais:
    r = minimize(himmelblau, ini)
    print("Inicial x:{}, y:{} -> min x: {:.2f}, min y: {:.2f}".format(ini[0], ini[1], r.x[0], r.x[1]))


"""
----------------------------------------
Dependendo do ponto inicial, a descida de gradiente leva o algoritmo de otimização
a pontos de mínimos locais diferentes. Isso pode ser contornado mexendo no peso de cada
"passo" do algoritmo ou testando vários pontos iniciais diferentes
----------------------------------------
"""