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
    pos = nx.circular_layout(G)  # seed pour reproductibilité
    #pos = nx.spring_layout(G, seed=42)  # seed pour reproductibilité
    #pos = nx.shell_layout(G)  # seed pour reproductibilité

    # Dessin du graphe
    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray',
            node_size=500, font_size=10)
    plt.show()


import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def afficher_graphe_adj(adj_matrix: np.ndarray):
    """
    Affiche le graphe correspondant à la matrice d'adjacence passée en argument,
    avec des arêtes dont l'épaisseur est proportionnelle à leur poids.

    Paramètres :
      - adj_matrix : numpy.ndarray
          La matrice d'adjacence représentant le graphe (matrice carrée).
    """
    # Créer le graphe à partir de la matrice (les nœuds seront numérotés de 0 à n-1)
    G = nx.from_numpy_array(adj_matrix)

    # Calculer une disposition des nœuds pour une bonne visualisation
    pos = nx.circular_layout(G)
    # pos = nx.spring_layout(G, seed=42)
    # pos = nx.shell_layout(G)

    # Calculer l'épaisseur des arêtes à partir de leur poids
    # Si le poids n'est pas défini, on considère une valeur par défaut égale à 1
    widths = [G[u][v].get('weight', 1) for u, v in G.edges()]

    # Ajuster l'échelle de l'épaisseur pour une meilleure visibilité
    facteur_echelle = 2  # Vous pouvez modifier ce facteur selon vos préférences
    widths = [facteur_echelle * w for w in widths]

    # Afficher le graphe en passant la liste des épaisseurs pour les arêtes
    nx.draw(G, pos, with_labels=True,
            node_color='lightblue', edge_color='gray',
            node_size=500, font_size=10, width=widths)
    plt.show()

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def afficher_graphe_adj(adj_matrix: np.ndarray, edges_highlight=None):
    """
    Affiche le graphe correspondant à la matrice d'adjacence passée en argument,
    avec :
      - des arêtes dont l'épaisseur est proportionnelle à leur poids,
      - un surlignage (couleur différente) sur certaines arêtes de la liste edges_highlight.

    Paramètres :
      - adj_matrix : numpy.ndarray
          La matrice d'adjacence représentant le graphe (matrice carrée).
      - edges_highlight : list of tuple (optionnel)
          Liste d'arêtes à mettre en valeur.
          Chaque arête doit être représentée par un tuple (u, v) où u et v sont des nœuds.
          Pour un graphe non orienté, (u, v) et (v, u) seront considérés comme identiques.
    """
    # Créer le graphe à partir de la matrice
    G = nx.from_numpy_array(adj_matrix)
    
    # Détermination de la disposition (layout) des nœuds
    pos = nx.spring_layout(G, seed=12)
    # Vous pouvez essayer circular_layout, shell_layout, etc.

    # Facteur d'échelle pour l'épaisseur des arêtes
    facteur_echelle = 2

    # Si aucune arête à mettre en valeur n'est précisée, on crée une liste vide
    if edges_highlight is None:
        edges_highlight = []
    
    # Pour un graphe non dirigé, on normalise les arêtes en triant chaque tuple
    highlight_set = {tuple(sorted(edge)) for edge in edges_highlight}

    # Séparer les arêtes entre celles à mettre en couleur et les autres
    edges_autres = []
    widths_autres = []
    edges_en_haut = []
    widths_en_haut = []

    for u, v in G.edges():
        # Récupérer le poids de l'arête ou 1 par défaut
        poids = G[u][v].get('weight', 1)
        largeur = poids * facteur_echelle
        # Pour un graphe non orienté, on vérifie dans le set avec tuple trié
        if tuple(sorted((u, v))) in highlight_set:
            edges_en_haut.append((u, v))
            widths_en_haut.append(largeur)
        else:
            edges_autres.append((u, v))
            widths_autres.append(largeur)
    
    # Dessiner les nœuds et les labels
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=500)
    nx.draw_networkx_labels(G, pos, font_size=10)
    
    # Dessiner d'abord les arêtes "classiques" en gris
    nx.draw_networkx_edges(G, pos, edgelist=edges_autres, width=widths_autres, edge_color='gray')
    # Puis dessiner par-dessus les arêtes à mettre en évidence en rouge
    nx.draw_networkx_edges(G, pos, edgelist=edges_en_haut, width=widths_en_haut, edge_color='red')
    
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
    n = 50
    with open("test.graph", "w") as file:
        for j in range(400):
            print(j)
            g = random_graph(n)
            #afficher_graphe_adj(g)

            gr = nx.from_numpy_array(g)

            import time

            #print(dict(nx.edge_betweenness_centrality(nx.from_numpy_array(g))))
            
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
        
tau = [-1.0, -1.0, -1.0, -1.0, 1.8307503491130612, -1.0, 0.8781289773971983, 1.8159029306608532, -1.0, -1.0, 0.8337121232265541, -1.0, 0.10000000000002274, 0.9357809567622959, -1.0, -1.0, -1.0, 1.8231358805112732, 0.4630104816217597, -1.0, -1.0, -1.0, 3.068528074287574, 0.39693026626077427, 0.10000000000002274, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.8231358805112732, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.2800511522589346, 2.3076420159681676, 3.254405139452447, -1.0, 1.4390917579856795, 1.3882846438545593, -1.0, 0.4630104816217597, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 2.14416590821575, -1.0, -1.0, -1.0, 3.7232768781629524, -1.0, 1.8307503491130612, -1.0, -1.0, -1.0, -1.0, 0.44333129563210133, 2.6728680641139717, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.813172316761992, -1.0, -1.0, -1.0, -1.0, 0.44333129563210133, -1.0, 1.4421014895183741, -1.0, -1.0, -1.0, -1.0, -1.0, 4.097960466523611, -1.0, 1.8340248302675377, 0.8781289773971983, -1.0, -1.0, -1.0, 2.6728680641139717, 1.4421014895183741, -1.0, -1.0, 2.192663301317709, -1.0, -1.0, 3.1448275163440003, 0.8302925177685113, 0.9601926189736743, -1.0, 1.8159029306608532, 3.068528074287574, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 0.10000000000002274, -1.0, -1.0, 1.8220481497936052, 1.8633990501812303, -1.0, -1.0, 0.39693026626077427, -1.0, -1.0, -1.0, -1.0, 2.192663301317709, -1.0, -1.0, 0.9313593110898117, -1.0, -1.0, 0.9515677990303903, 1.3771559762382315, -1.0, -1.0, 0.10000000000002274, 1.2800511522589346, 2.14416590821575, -1.0, -1.0, -1.0, 0.10000000000002274, 0.9313593110898117, -1.0, 0.8226111398698944, -1.0, 0.94612888212744, 0.48784364087832954, 2.7142791742511574, 0.8337121232265541, -1.0, 2.3076420159681676, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 0.8226111398698944, -1.0, -1.0, 1.396039860533306, -1.0, 0.4478337956889477, -1.0, -1.0, 3.254405139452447, -1.0, -1.0, -1.0, 3.1448275163440003, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 0.10000000000002274, -1.0, -1.0, -1.0, -1.0, 4.097960466523611, 0.8302925177685113, 1.8220481497936052, 0.9515677990303903, 0.94612888212744, 1.396039860533306, -1.0, -1.0, 0.444045801247294, 0.10000000000002274, 0.9357809567622959, -1.0, 1.4390917579856795, 3.7232768781629524, -1.0, -1.0, 0.9601926189736743, 1.8633990501812303, 1.3771559762382315, 0.48784364087832954, -1.0, -1.0, 0.444045801247294, -1.0, 2.2087993517168636, -1.0, -1.0, 1.3882846438545593, -1.0, 1.813172316761992, 1.8340248302675377, -1.0, -1.0, -1.0, 2.7142791742511574, 0.4478337956889477, -1.0, 0.10000000000002274, 2.2087993517168636, -1.0]

tau = np.array(tau).reshape((n, n))
tau += 1

edges = [[0, 13], [1, 8], [2, 11], [3, 9], [4, 14], [5, 14], [6, 12], [6, 13], [6, 11], [7, 12], [8, 12], [9, 12], [10, 12], [12, 14]]


nx.from_numpy_array(tau)
afficher_graphe_adj(tau, edges)


