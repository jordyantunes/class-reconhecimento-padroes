from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt
from itertools import chain
from sklearn.svm import SVC
import numpy as np

cov = [[0.2, 0.0], [0.0, 0.2]]

medias = {-1: [ 0, 0],
          1: [ 1.5, 1.5]}


treinamento = {
    classe: np.random.multivariate_normal(media, cov, 200)
    for classe, media in medias.items()
}

train_values = np.array(list(chain.from_iterable(treinamento.values())))
train_labels = np.array(list(chain.from_iterable([[k for _ in range(200)] for k in treinamento.keys()])))


teste = {
    classe: np.random.multivariate_normal(media, cov, 200)
    for classe, media in medias.items()
}

test_values = np.array(list(chain.from_iterable(teste.values())))
test_labels = np.array(list(chain.from_iterable([[k for _ in range(200)] for k in teste.keys()])))

cs = [0.1, 0.2, 0.5, 1, 2, 20]
tol = 0.001

for i, c in enumerate(cs):
    clf = SVC(C=c, tol=tol, kernel='linear')
    clf.fit(train_values, train_labels)
    z = clf.predict(test_values)
    acc = accuracy_score(test_labels, z)
    qtde_support_vectors = clf.support_vectors_.shape[0]

    w = clf.coef_[0]
    a = -w[0] / w[1]
    xx = np.linspace(-5, 5)
    yy = a * xx - (clf.intercept_[0]) / w[1]

    margin = 1 / np.sqrt(np.sum(clf.coef_ ** 2))
    yy_down = yy - np.sqrt(1 + a ** 2) * margin
    yy_up = yy + np.sqrt(1 + a ** 2) * margin

    print("Acurácia: {}, vetores suporte: {}, margem: {}".format(acc, qtde_support_vectors, margin))

    margin = 1 / np.sqrt(np.sum(clf.coef_ ** 2))
    yy_down = yy - np.sqrt(1 + a ** 2) * margin
    yy_up = yy + np.sqrt(1 + a ** 2) * margin

    # plot the line, the points, and the nearest vectors to the plane
    plt.figure(i, figsize=(4, 3))
    plt.clf()
    plt.plot(xx, yy, 'k-')
    plt.plot(xx, yy_down, 'k--')
    plt.plot(xx, yy_up, 'k--')

    plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=80,
                facecolors='none', zorder=10, edgecolors='k')
    # cores = np.where(test_labels == 'c1', 20, -20)
    plt.scatter(test_values[:, 0], test_values[:, 1], c=test_labels, 
                zorder=10, cmap=plt.cm.Paired,
                edgecolors='k')

    plt.axis('tight')
    x_min = -2
    x_max = 4
    y_min = -2
    y_max = 2

    XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
    Z = clf.predict(np.c_[XX.ravel(), YY.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(XX.shape)
    plt.figure(i, figsize=(4, 3))
    plt.pcolormesh(XX, YY, Z, cmap=plt.cm.Paired)

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    plt.xticks(())
    plt.yticks(())

    # ax = plt.gca()
    # ax.set_xlim(x_min, x_max)
    # ax.set_ylim(y_min, y_max)

plt.show()
