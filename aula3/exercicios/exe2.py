from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
from scipy.optimize import minimize
import numpy as np


def fun(x, y):
    return ((0.25*x) ** 2) + (y ** 2)

def fun_x(x):
    return ((0.25*x[0]) ** 2) + (x[1] ** 2)

fun_vec = np.vectorize(fun)

# geracao de valores de x, y, x
x = y = np.linspace(-1, 1)
X, Y = np.meshgrid(x, y)
zs = np.array(fun_vec(np.ravel(X), np.ravel(Y)))
Z = zs.reshape(X.shape)

ax = plt.subplot(projection='3d')
ax.plot_surface(X, Y, Z)
plt.show()


r = minimize(fun_x, [2, 1], method='SLSQP', constraints=(
    {'type': 'eq', 'fun': lambda x: 5 - x[0] - x[1]},
    {'type': 'ineq', 'fun': lambda x: x[0] + 0.3*x[1] - 3}
))

