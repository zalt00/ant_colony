
import networkx as nx
from networkx.algorithms import approximation
edges = []
v = dict()
i = 0
with open("data/wiki_sparse.txt") as file:
    for line in file:
        if line and "#" not in line:
            e = [int(a) for a in line.split()]
            for a in e:
                if a not in v:
                    v[a] = i
                    i += 1
            edges.append([v[a] for a in e])

G = nx.Graph(edges).to_undirected()

(tw, G2) = approximation.treewidth_min_degree(G)

print(tw, G2)
print(len(G2.edges))
for node in G.nodes:
    if not isinstance(node, int):
        print(node)


