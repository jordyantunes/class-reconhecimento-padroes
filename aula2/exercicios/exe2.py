from numpy.linalg import inv
from sklearn import mixture
import numpy as np

grupos = ['c1', 'c2', 'c3']
prob = {'c1': 0.4, 'c2': 0.4, 'c3': 0.2} 
cov = {'c1': 0.1 * np.identity(2), 'c2': 0.2 * np.identity(2), 'c3': 0.3 * np.identity(2)}

medias = {'c1': [ 1, 1],
          'c2': [ 3, 3],   
          'c3': [ 2, 6]}


treinamento = {}

for g, c, m, p in zip(grupos, cov.values(), medias.values(), prob.values()):
    treinamento[g] = np.random.multivariate_normal(m, c, int(1000 * p))

# exercicio a
gauss = mixture.GaussianMixture(n_components=3,
                                means_init=[[0, 2], [5,2], [5, 5]],
                                precisions_init=[inv(c) for c in cov.values()],
                                weights_init=[w for w in prob.values()])

gauss.fit(np.concatenate(list(treinamento.values())))

print(gauss.means_)
print(gauss.covariances_)

# exercicio b
# com os valores passados, o modelo não funciona. 
# As médias e covariâncias encontradas são todas bem diferentes
gauss = mixture.GaussianMixture(n_components=3,
                                means_init=[[0, 6], [10,1], [3, 1]],
                                precisions_init=[inv(c) for c in cov.values()],
                                weights_init=[w for w in prob.values()]
                                )

gauss.fit(np.concatenate(list(treinamento.values())))

print(gauss.means_)
print(gauss.covariances_)