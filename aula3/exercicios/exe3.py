from numpy import array, dot
from qpsolvers import solve_qp

P = array([[-5./4., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
q = array([-2., 1., 0])
G = array([[1.,  0., 1.],
           [0., 1., 10.],
           [ 1., 1., 0.]])
h = array([0., 0., 1.])
A = array([0., 0., 0.])
b = array([0.])

x = solve_qp(P, q, G, h, solver='quadprog')
print("QP solution: x = {}".format(x))