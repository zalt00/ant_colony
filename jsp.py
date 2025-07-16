import networkx as nx

# 1. Charger le graphe d’origine
G = nx.Graph()
with open("./data/social_network/com-dblp.ungraph.txt", "r") as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) != 2:
            continue
        u, v = map(int, parts)
        G.add_edge(u, v)

# 2. Filtrer les composantes de taille >= 50
ccs = list(nx.connected_components(G))
large_ccs = [c for c in ccs if len(c) >= 50]
nodes_to_keep = set().union(*large_ccs)

H = G.subgraph(nodes_to_keep).copy()

# 3. Renuméroter les nœuds de 0 à n-1
mapping = {old: new for new, old in enumerate(sorted(H.nodes()))}
H = nx.relabel_nodes(H, mapping)

# 4. Sauvegarder le graphe filtré dans un nouveau fichier
output_path = "./data/social_network/com-dblp.ungraph-filtered.txt"
with open(output_path, "w") as f:
    for u, v in H.edges():
        f.write(f"{u} {v}\n")

# 5. Afficher un résumé
print(f"Graph filtré : {H.number_of_nodes()} nœuds, {H.number_of_edges()} arêtes")
