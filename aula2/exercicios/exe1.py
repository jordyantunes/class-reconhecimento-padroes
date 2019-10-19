from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import distance
import matplotlib.pyplot as plt
from numpy.linalg import inv
from functools import reduce
from itertools import chain
import numpy as np

cov = [[0.8, 0.0, 0.0], [0.0, 0.8, 0.0], [0.0, 0.0, 0.8]]

medias = {'c1': [ 0, 0, 0],
          'c2': [ 1, 2, 2],   
          'c3': [ 3, 3, 4]}

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


treinamento = {}
teste = {}
media_ml = {}
cov_ml = {}

# exercicio a
for cl, m in medias.items():
    treinamento[cl] = np.random.multivariate_normal(m, cov, 1000)
    teste[cl] = np.random.multivariate_normal(m, cov, 1000)

    media_ml[cl] = np.mean(treinamento[cl], axis=0)
    cov_ml[cl] = np.cov(treinamento[cl], rowvar=False)

    # print("{}: ".format(cl), media_ml[cl])
    # print("{}: ".format(cl), cov_ml[cl])

    # x, y, z = treinamento[cl].T
    # ax.scatter(x, y, z, alpha=0.5)
    # plt.show()

cov_final = (cov_ml['c1'] + cov_ml['c2'] + cov_ml['c3'])/3
cov_final
cov_final_inv = inv(cov_final)

amostra_testes = np.concatenate(list(teste.values()))
labels = list(chain.from_iterable([[k for _ in range(1000)] for k in teste.keys()]))

# item b
def classificar_euclideano(ponto):
    global media_ml
    return min([(k, distance.euclidean(ponto, m)) for k, m in media_ml.items()], key=lambda v: v[1])[0]

classificado = np.apply_along_axis(classificar_euclideano, 1, amostra_testes)
acc = accuracy_score(labels, classificado)
print("Acurácia Euclidiano", acc)


# item c
def classificar_mahalanobis(ponto):
    global cov_final_inv
    return min([(k, distance.mahalanobis(ponto, m, cov_final_inv)) for k, m in media_ml.items()], key=lambda v: v[1])[0]


classificado = np.apply_along_axis(classificar_mahalanobis, 1, amostra_testes)
acc = accuracy_score(labels, classificado)
print("Acurácia Mahalanobis", acc)

amostra_treinamento = np.concatenate([t for t in treinamento.values()])
labels_treinamento  = np.array(list(chain.from_iterable([[k for _ in range(1000)] for k in treinamento.keys()])))

# d
gnb = GaussianNB()
gnb.fit(amostra_treinamento, labels_treinamento)
classificado = gnb.predict(amostra_testes)
acc = accuracy_score(labels, classificado)
print("Acurácia Bayes", acc)