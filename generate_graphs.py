import numpy as np

import random

def init_uf(n):
    return [-1] * n

def find(uf, x):
    if uf[x] < 0:
        return x
    cx = find(uf, uf[x])
    uf[x] = cx
    return cx

def union(uf, x, y):
    cx = find(uf, x)
    cy = find(uf, y)
    if cx != cy:
        if -uf[cx] > -uf[cy]:
            uf[cx] += uf[cy]
            uf[cy] = cx
        else:
            uf[cy] += uf[cx]
            uf[cx] = cy

def random_graph(n):
    uf = init_uf(n)
    g = np.zeros((n, n), dtype=int)
    ind = 0
    c = False
    while not c:
        i = random.randint(0, n-2)
        j = random.randint(i+1, n-1)
        if not g[i, j]:
            g[i, j] = g[j, i] = 1
            union(uf, i, j)

        ind += 1
        if ind >= 10*n:
            ind = 0
            c = True
            cb = find(uf, 0)
            for i in range(1, n):
                if find(uf, i) != cb:
                    c = False
                    break
    return g

import networkx as nx
import matplotlib.pyplot as plt

def afficher_graphe_adj(adj_matrix: np.ndarray):
    """
    Affiche le graphe correspondant à la matrice d'adjacence passée en argument.
    
    Paramètres :
      - adj_matrix : numpy.ndarray
          La matrice d'adjacence représentant le graphe (matrice carrée).
    """
    # Crée le graphe à partir de la matrice (les nœuds seront numérotés de 0 à n-1)
    G = nx.from_numpy_array(adj_matrix)
    
    # On calcule une disposition des nœuds (layout) avec spring_layout pour une bonne visualisation
    pos = nx.kamada_kawai_layout(G)  # seed pour reproductibilité
    #pos = nx.spring_layout(G, seed=42)  # seed pour reproductibilité
    #pos = nx.shell_layout(G)  # seed pour reproductibilité

    # Dessin du graphe
    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray',
            node_size=500, font_size=10)
    plt.show()


def floyd_warshall(adj):
    """
    Calcule les distances les plus courtes entre chaque paire de sommets
    d'un graphe représenté par une matrice d'adjacence.
    
    Pour un graphe non pondéré, on considère que chaque arête a un poids de 1.
    Les entrées à 0 (hors diagonale) indiquent l'absence d'arête et sont remplacées
    par np.inf.
    
    Paramètres
    ----------
    adj : numpy.ndarray
        Matrice d'adjacence (de forme (n, n)). Pour i != j, 
        adj[i, j] > 0 signifie la présence d'une arête (ou le poids de celle-ci),
        et 0 signifie l'absence d'arête.
    
    Renvoie
    -------
    D : numpy.ndarray
        La matrice des distances les plus courtes entre chaque paire de sommets.
    """
    n = adj.shape[0]
    # Initialiser la matrice des distances :
    # sur la diagonale, la distance est 0
    # en dehors, si aucune arête n'existe (adj == 0), on affecte l'infini
    D = np.where(adj == 0, np.inf, adj).astype(float)
    np.fill_diagonal(D, 0)
    
    # Algorithme de Floyd–Warshall
    for k in range(n):
        # Utilisation de la diffusion (broadcasting) pour mettre à jour toutes les paires (i,j)
        D = np.minimum(D, np.add.outer(D[:, k], D[k, :]))
    
    return D

def compute_distortion(A_orig, A_tree, mode="average"):
    """
    Calcule la distorsion d'un sous-graphe (par exemple, un arbre couvrant) par rapport au
    graphe original. On suppose que les deux graphes sont représentés par des matrices d'adjacence.
    
    La distorsion est définie comme le rapport des distances dans le sous-graphe et celles 
    dans le graphe original, pris en moyenne (ou au pire cas).
    
    Paramètres
    ----------
    A_orig : numpy.ndarray
        Matrice d'adjacence du graphe original.
    A_tree : numpy.ndarray
        Matrice d'adjacence du sous-graphe (par exemple, un arbre couvrant).
    mode : str, optionnel
        "average" pour la distorsion moyenne
        "max" pour la distorsion en pire cas.
    
    Renvoie
    -------
    float
        La distorsion calculée.
    """
    # Calcul des distances dans le graphe original et dans l'arbre/sous-graphe
    D_orig = floyd_warshall(A_orig)
    D_tree = floyd_warshall(A_tree)
    
    n = A_orig.shape[0]
    ratios = []
    for i in range(n):
        for j in range(i + 1, n):
            # Dans un graphe connexe, D_orig[i,j] doit être > 0.
            # On calcule le rapport pour chaque paire.
            if D_orig[i, j] > 0:
                ratios.append(D_tree[i, j] / D_orig[i, j])
    
    ratios = np.array(ratios)
    
    if mode == "average":
        return ratios.mean()
    elif mode =="max":
        return ratios.max()
    else:
        raise ValueError("Mode non supporté. Utilisez 'average' ou 'max'.")



def resolve(g):
    cmin = 99999999999
    ct = None
    def explore(i, j, n, cur_n, tree, g):
        nonlocal cmin, ct
        if cur_n == n-1:
            dist = compute_distortion(g, tree, mode="average")
            cmin = min(dist, cmin)
            if cmin == dist:
                ct = tree.copy()
            return
        for k in range(j+1, n):
            if g[i, k]:
                tree[i, k] = tree[k, i] = 1
                explore(i, k, n, cur_n + 1, tree, g)
                tree[i, k] = tree[k, i] = 0
                explore(i, k, n, cur_n, tree, g)
                return

        if i < n-1:
            explore(i+1, i+1, n, cur_n, tree, g)
    explore(0, -1, g.shape[0], 0, g * 0, g)
    return cmin, ct

if __name__ == '__main__':
    random.seed(1212)
    n = 100
    with open("test.graph", "w") as file:
        for j in range(400):
            print(j)
            g = random_graph(n)
            
            
            (disto, t) = (1.0, None) #resolve(g)

            oriented_edges = []
            edges = []
            out_edges = [list() for _ in range(n)]
            in_edges = [list() for _ in range(n)]
            for i in range(n):
                for j in range(n):
                    if g[i, j]:
                        if i != j:
                            oriented_edges.append((i, j))
                            if i < j:
                                edges.append((i, j))
                        out_edges[i].append((i, j))
                        in_edges[i].append((j, i))
            print(disto, end="|", file=file)
            print(n, end="|", file=file)
            print(*[f"{i},{j}" for (i, j) in edges], sep=";", end="::", file=file)
        




