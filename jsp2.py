import networkx as nx

# 1. Créer un graphe non orienté
G = nx.Graph()

# 2. Charger les arêtes depuis le fichier
with open("./data/social_network/com-dblp.ungraph.txt", "r") as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) != 2:
            continue
        u, v = parts
        # conversion en int si vos nœuds sont des entiers
        G.add_edge(int(u), int(v))

# 3. Calculer les composantes connexes
n_cc = nx.number_connected_components(G)
cc_generator = nx.connected_components(G)

# 4. Calculer la taille de chaque composante
sizes = [len(c) for c in cc_generator]
sizes.sort(reverse=True)  # optionnel : du plus grand au plus petit

# 5. Afficher les résultats
print(f"Nombre de composantes connexes : {n_cc}")
print("Tailles de chaque composante :")
for i, size in enumerate(sizes, start=1):
    print(f"  Composante {i} : {size} nœuds")
